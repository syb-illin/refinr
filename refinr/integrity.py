"""
Contrôle qualité "pré-mastering" niveau professionnel : détecte tout ce
qu'un ingénieur mastering vérifierait avant même de toucher un EQ — pas
des corrections, des CONSTATS factuels sur l'intégrité du signal.

Complète `analysis.py` (qui pilote les décisions d'EQ) par des vérifications
que l'analyse spectrale/dynamique globale ne fait PAS :
  - échantillons invalides (NaN/Inf) — corruption de fichier, jamais silencieux
  - offset DC (pollution basse fréquence invisible à l'oreille sur un système
    mal calibré, mange du headroom, cause des clics au montage/loop)
  - silence en tête/queue (souvent laissé par les exports IA — Suno inclus)
  - déséquilibre de balance L/R (mixage asymétrique non voulu)
  - écrêtage DUR (runs d'échantillons consécutifs collés au plafond — signature
    d'un vrai clipping numérique, distinct des pics isolés proches de 0dBFS déjà
    couverts par `analysis.DynamicsProfile.clipping_ratio`)
  - plancher de bruit (hiss/hum anormal)
  - rolloff spectral anormalement bas (signature classique d'une source
    transcodée en lossy — mp3 128-192kbps coupe typiquement vers 16-19kHz —
    ou d'un export IA bandlimité)
  - corrélation stéréo PAR BANDE (pas juste un score global) : un problème de
    phase localisé à une seule zone du spectre (ex: le haut-médium, signature
    rapportée pour les artefacts de génération IA) peut être noyé dans une
    corrélation large-bande qui paraît saine.
  - repli MONO (mono fold-down) : perte de niveau au sommage (L+R)/2 par
    rapport à la moyenne RMS des deux canaux. Un signal stéréo non corrélé
    perd naturellement ~3dB à ce sommage (rien d'anormal) ; un déficit
    nettement supérieur signale une vraie annulation de phase, un problème
    réel de compatibilité mono (lecture téléphone/enceinte Bluetooth/club en
    mono partiel) — jusqu'ici jamais constaté explicitement (voir l'ancien
    commentaire "future mono fold-down check, backlog" dans stereo_width.py).

Rien de tout ça n'est corrigé automatiquement ici — ce module CONSTATE,
`chain.py` décide quoi en faire (avertissement dans le rapport au minimum,
correction EQ ciblée pour ce qui est déjà actionnable comme le High Pass
anti-DC/sub déjà présent dans `proq4_control.decide_bands`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import csd, welch

from .audio_io import AudioBuffer

SILENCE_THRESHOLD_DBFS = -60.0
DC_OFFSET_ISSUE_THRESHOLD_DBFS = -50.0
CHANNEL_BALANCE_ISSUE_DB = 1.0
HARD_CLIP_THRESHOLD_LINEAR = 10 ** (-0.1 / 20.0)  # à moins de 0.1dB de 0dBFS
HARD_CLIP_MIN_RUN_SAMPLES = 3  # runs plus courts = pics isolés, pas du clipping
ROLLOFF_ENERGY_FRACTION = 0.99
LOSSY_ROLLOFF_HZ_THRESHOLD = 16000.0
LOSSY_CHECK_MIN_SAMPLE_RATE = 40000  # inutile de vérifier sur du 22kHz déjà bas
LOCALIZED_PHASE_ISSUE_CORRELATION = 0.3

# Perte (dB) au repli mono au-delà de laquelle on considère qu'il y a une
# vraie annulation de phase, pas juste la perte "normale" d'un signal stéréo
# non corrélé. Théorie : pour deux canaux indépendants de même RMS sigma,
# mono=(L+R)/2 a une RMS de sigma/sqrt(2) (~-3dB) par rapport à la moyenne
# RMS des deux canaux (=sigma) — c'est la ligne de base attendue, pas un
# défaut. On fixe le seuil à 2x cette perte théorique (6dB) pour ne signaler
# que les cas où quelque chose s'annule réellement (contenu hors-phase),
# pas la largeur stéréo normale d'un mix.
MONO_FOLD_LOSS_ISSUE_DB = 6.0

# Mêmes bandes que analysis.BANDS_HZ (dupliqué volontairement : ce module
# reste indépendant d'analysis.py, testable isolément).
BANDS_HZ = {
    "sub": (20, 60),
    "low": (60, 250),
    "low_mid": (250, 800),
    "mid": (800, 2500),
    "high_mid": (2500, 6000),
    "high": (6000, 12000),
    "air": (12000, 20000),
}


def _linear_to_dbfs(value: float) -> float:
    return 20.0 * np.log10(value) if value > 0 else -np.inf


@dataclasses.dataclass
class IntegrityReport:
    has_nan: bool
    has_inf: bool

    dc_offset_dbfs: dict[str, float]  # par canal ("left"/"right" ou "mono")
    dc_offset_issue: bool

    leading_silence_sec: float
    trailing_silence_sec: float

    channel_balance_db: float | None  # None si mono
    channel_balance_issue: bool

    clip_event_count: int  # nombre de runs de clipping DUR (>= HARD_CLIP_MIN_RUN_SAMPLES)
    longest_clip_run_ms: float

    noise_floor_dbfs: float

    spectral_rolloff_hz: float
    lossy_source_suspected: bool

    band_stereo_correlation: dict[str, float]
    localized_phase_issue_bands: list[str]  # bandes avec corrélation < seuil, isolément du score large-bande

    mono_fold_loss_db: float  # perte au repli (L+R)/2 vs moyenne RMS des canaux, voir MONO_FOLD_LOSS_ISSUE_DB
    mono_fold_issue: bool

    def issue_tags(self) -> list[str]:
        """Tags courts, dans le même esprit que FileAnalysis.summary_tags()."""
        tags = []
        if self.has_nan or self.has_inf:
            tags.append("corrupted_samples")
        if self.dc_offset_issue:
            tags.append("dc_offset_detected")
        if self.leading_silence_sec > 0.5:
            tags.append("leading_silence")
        if self.trailing_silence_sec > 0.5:
            tags.append("trailing_silence")
        if self.channel_balance_issue:
            tags.append("channel_balance_off")
        if self.clip_event_count > 0:
            tags.append("hard_clipping_detected")
        if self.lossy_source_suspected:
            tags.append("lossy_source_suspected")
        if self.localized_phase_issue_bands:
            tags.append("localized_phase_issue")
        if self.mono_fold_issue:
            tags.append("mono_fold_down_issue")
        return tags


def _detect_edge_silence(mono: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Durée de silence (RMS glissant < SILENCE_THRESHOLD_DBFS) en tête et en queue."""
    if mono.size == 0:
        return 0.0, 0.0

    win = max(1, int(0.05 * sample_rate))  # fenêtres de 50ms
    n_windows = mono.size // win
    if n_windows == 0:
        return 0.0, 0.0

    trimmed = mono[: n_windows * win].reshape(n_windows, win)
    rms_per_window = np.sqrt(np.mean(trimmed**2, axis=1)) + 1e-12
    dbfs_per_window = 20.0 * np.log10(rms_per_window)

    is_silent = dbfs_per_window < SILENCE_THRESHOLD_DBFS

    leading = 0
    for v in is_silent:
        if not v:
            break
        leading += 1

    trailing = 0
    for v in is_silent[::-1]:
        if not v:
            break
        trailing += 1

    win_sec = win / float(sample_rate)
    return leading * win_sec, trailing * win_sec


def _count_hard_clip_runs(stereo: np.ndarray) -> tuple[int, float, int]:
    """
    Compte les runs d'échantillons consécutifs collés au plafond numérique
    (signature d'un vrai écrêtage, par opposition à un pic isolé proche de
    0dBFS qui peut être parfaitement légitime musicalement).
    Retourne (nombre de runs, durée du plus long run en échantillons, sample_rate-agnostique).
    """
    per_sample_peak = np.max(np.abs(stereo), axis=1) if stereo.ndim > 1 else np.abs(stereo)
    clipped = per_sample_peak >= HARD_CLIP_THRESHOLD_LINEAR

    if not np.any(clipped):
        return 0, 0.0, 0

    # Encode les runs de True consécutifs via les indices de transition.
    diff = np.diff(np.concatenate(([0], clipped.astype(np.int8), [0])))
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    run_lengths = run_ends - run_starts

    significant = run_lengths[run_lengths >= HARD_CLIP_MIN_RUN_SAMPLES]
    longest = int(run_lengths.max()) if run_lengths.size else 0
    return int(significant.size), 0.0, longest  # ms calculé par l'appelant (a besoin du sample_rate)


def _noise_floor_dbfs(mono: np.ndarray, sample_rate: int) -> float:
    """RMS de la fenêtre de 200ms la plus silencieuse — estimation grossière
    du plancher de bruit (hiss/hum), pas une mesure de bruit certifiée."""
    win = max(1, int(0.2 * sample_rate))
    n_windows = mono.size // win
    if n_windows == 0:
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        return _linear_to_dbfs(rms)

    trimmed = mono[: n_windows * win].reshape(n_windows, win)
    rms_per_window = np.sqrt(np.mean(trimmed**2, axis=1))
    quietest = float(np.min(rms_per_window))
    return _linear_to_dbfs(quietest)


def _spectral_rolloff_hz(mono: np.ndarray, sample_rate: int, fraction: float = ROLLOFF_ENERGY_FRACTION) -> float:
    freqs, psd = welch(mono, fs=sample_rate, nperseg=min(8192, max(256, len(mono))))
    cumulative = np.cumsum(psd)
    total = cumulative[-1] if cumulative.size else 0.0
    if total <= 0:
        return float(sample_rate) / 2.0
    idx = np.searchsorted(cumulative, fraction * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _band_stereo_correlation(stereo: np.ndarray, sample_rate: int) -> dict[str, float]:
    """
    Corrélation stéréo PAR BANDE de fréquence, via cross-spectral density
    (scipy.signal.csd) plutôt qu'une corrélation de Pearson large-bande —
    permet de repérer un problème de phase localisé à une seule zone du
    spectre (ex: uniquement le haut-médium) qui serait noyé dans un score
    global sain. Corrélation reconstruite depuis la cohérence + le signe de
    la partie réelle du cross-spectre (approximation standard : cohérence
    signée), bornée à [-1, 1].
    """
    left, right = stereo[:, 0], stereo[:, 1]
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return dict.fromkeys(BANDS_HZ, 1.0)

    nperseg = min(8192, max(256, len(left)))
    freqs, pxy = csd(left, right, fs=sample_rate, nperseg=nperseg)
    _, pxx = welch(left, fs=sample_rate, nperseg=nperseg)
    _, pyy = welch(right, fs=sample_rate, nperseg=nperseg)

    denom = np.sqrt(np.maximum(pxx * pyy, 1e-30))
    coherence_signed = np.real(pxy) / denom
    coherence_signed = np.clip(coherence_signed, -1.0, 1.0)

    # Seuil RELATIF (pas une constante absolue) : une bande qui ne porte
    # qu'une fraction négligeable de l'énergie totale du signal (ex: la
    # bande "air" sur une source déjà passée au travers d'un filtre passe-
    # bas) donne une corrélation non significative/bruitée, pas un vrai
    # problème de phase — on l'exclut plutôt que de risquer un faux positif.
    total_weight = float(np.sum(pxx) + np.sum(pyy)) + 1e-30
    negligible_fraction = 0.005

    result: dict[str, float] = {}
    for name, (lo, hi) in BANDS_HZ.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            result[name] = 1.0
            continue
        weights = pxx[mask] + pyy[mask]
        band_weight = float(np.sum(weights))
        if band_weight / total_weight < negligible_fraction:
            result[name] = 1.0  # bande négligeable en énergie : pas de constat de phase fiable possible
            continue
        result[name] = float(np.average(coherence_signed[mask], weights=weights))
    return result


def _mono_fold_down_loss_db(stereo: np.ndarray) -> float:
    """
    Perte de niveau (dB) au repli mono (L+R)/2, par rapport à la moyenne RMS
    des deux canaux stéréo — voir MONO_FOLD_LOSS_ISSUE_DB pour l'explication
    de la ligne de base ~3dB "normale". 0.0 pour une source mono ou un canal
    quasi silencieux (rien à mesurer). Une valeur élevée (proche de
    l'annulation totale) est plafonnée à 120.0 plutôt que de tendre vers
    l'infini.
    """
    if stereo.ndim < 2 or stereo.shape[1] < 2:
        return 0.0
    left, right = stereo[:, 0], stereo[:, 1]
    rms_left = float(np.sqrt(np.mean(left**2)))
    rms_right = float(np.sqrt(np.mean(right**2)))
    reference_rms = (rms_left + rms_right) / 2.0
    if reference_rms < 1e-9:
        return 0.0
    mono = (left + right) / 2.0
    mono_rms = float(np.sqrt(np.mean(mono**2)))
    if mono_rms < 1e-9:
        return 120.0  # annulation quasi totale
    return min(120.0, 20.0 * np.log10(reference_rms / mono_rms))


def check_integrity(buffer: AudioBuffer) -> IntegrityReport:
    samples = buffer.samples
    stereo = buffer.as_stereo()
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples

    has_nan = bool(np.any(np.isnan(samples)))
    has_inf = bool(np.any(np.isinf(samples)))

    # Les checks suivants supposent des échantillons finis : on les
    # neutralise en 0.0 uniquement pour le calcul (le has_nan/has_inf
    # ci-dessus reste la source de vérité sur la corruption elle-même).
    safe_stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)
    safe_mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    if samples.ndim > 1 and samples.shape[1] >= 2:
        dc_offset_dbfs = {
            "left": _linear_to_dbfs(abs(float(np.mean(safe_stereo[:, 0])))),
            "right": _linear_to_dbfs(abs(float(np.mean(safe_stereo[:, 1])))),
        }
    else:
        dc_offset_dbfs = {"mono": _linear_to_dbfs(abs(float(np.mean(safe_mono))))}
    dc_offset_issue = any(v > DC_OFFSET_ISSUE_THRESHOLD_DBFS for v in dc_offset_dbfs.values())

    leading_sec, trailing_sec = _detect_edge_silence(safe_mono, buffer.sample_rate)

    channel_balance_db: float | None = None
    channel_balance_issue = False
    if samples.ndim > 1 and samples.shape[1] >= 2:
        rms_left = float(np.sqrt(np.mean(safe_stereo[:, 0] ** 2))) + 1e-12
        rms_right = float(np.sqrt(np.mean(safe_stereo[:, 1] ** 2))) + 1e-12
        channel_balance_db = 20.0 * float(np.log10(rms_right / rms_left))
        channel_balance_issue = abs(channel_balance_db) > CHANNEL_BALANCE_ISSUE_DB

    clip_event_count, _, longest_run_samples = _count_hard_clip_runs(safe_stereo)
    longest_clip_run_ms = 1000.0 * longest_run_samples / float(buffer.sample_rate)

    noise_floor = _noise_floor_dbfs(safe_mono, buffer.sample_rate)

    rolloff_hz = _spectral_rolloff_hz(safe_mono, buffer.sample_rate)
    lossy_suspected = buffer.sample_rate >= LOSSY_CHECK_MIN_SAMPLE_RATE and rolloff_hz < LOSSY_ROLLOFF_HZ_THRESHOLD

    band_corr = _band_stereo_correlation(safe_stereo, buffer.sample_rate)
    localized_issue_bands = [name for name, corr in band_corr.items() if corr < LOCALIZED_PHASE_ISSUE_CORRELATION]

    mono_fold_loss_db = _mono_fold_down_loss_db(safe_stereo) if samples.ndim > 1 and samples.shape[1] >= 2 else 0.0
    mono_fold_issue = mono_fold_loss_db > MONO_FOLD_LOSS_ISSUE_DB

    return IntegrityReport(
        has_nan=has_nan,
        has_inf=has_inf,
        dc_offset_dbfs={k: round(v, 2) if np.isfinite(v) else -120.0 for k, v in dc_offset_dbfs.items()},
        dc_offset_issue=dc_offset_issue,
        leading_silence_sec=round(leading_sec, 2),
        trailing_silence_sec=round(trailing_sec, 2),
        channel_balance_db=round(channel_balance_db, 2) if channel_balance_db is not None else None,
        channel_balance_issue=channel_balance_issue,
        clip_event_count=clip_event_count,
        longest_clip_run_ms=round(longest_clip_run_ms, 2),
        noise_floor_dbfs=round(noise_floor, 1) if np.isfinite(noise_floor) else -120.0,
        spectral_rolloff_hz=round(rolloff_hz, 0),
        lossy_source_suspected=lossy_suspected,
        band_stereo_correlation={k: round(v, 3) for k, v in band_corr.items()},
        localized_phase_issue_bands=localized_issue_bands,
        mono_fold_loss_db=round(mono_fold_loss_db, 2),
        mono_fold_issue=mono_fold_issue,
    )
