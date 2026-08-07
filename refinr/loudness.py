"""
Mesure de loudness (ITU-R BS.1770-4 via pyloudnorm) et true peak (suréchantillonnage),
+ gain staging automatique en entrée de chaîne.

Beaucoup de sources sortent des fichiers très chauds (proches de 0 dBFS,
parfois déjà écrêtés numériquement). Avant de toucher aux plugins de retraitement
(EQ/saturation/tape), on ramène systématiquement le niveau à une loudness
d'entrée cible (-18 LUFS intégré par défaut) pour que les plugins réagissent
comme sur un signal "normal" et ne saturent pas de façon incontrôlée.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

from .audio_io import AudioBuffer

# Cible par défaut de gain-staging avant la chaîne de plugins.
DEFAULT_GAIN_STAGING_LUFS = -18.0

# Facteur de suréchantillonnage pour l'estimation du true peak (BS.1770-4 Annexe 2).
TRUE_PEAK_OVERSAMPLE = 4


@dataclasses.dataclass
class LoudnessMeasurement:
    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float | None
    sample_peak_dbfs: float


def _meter(sample_rate: int) -> pyln.Meter:
    return pyln.Meter(sample_rate)


def measure_true_peak_dbtp(samples: np.ndarray, sample_rate: int) -> float:
    """
    Estime le true peak par suréchantillonnage (méthode BS.1770-4).
    samples: (n_frames,) ou (n_frames, n_channels)
    """
    if samples.size == 0:
        return -np.inf
    mono_or_multi = samples if samples.ndim > 1 else samples[:, None]
    oversampled = resample_poly(mono_or_multi, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    peak = float(np.max(np.abs(oversampled)))
    if peak <= 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def measure_loudness_range(samples: np.ndarray, sample_rate: int) -> float | None:
    """
    Estimation simplifiée de la loudness range (LRA) façon EBU R128 :
    percentile 95 - percentile 10 des mesures de loudness court-terme (3s),
    après gating relatif à -20 LU sous la loudness intégrée. Ce n'est pas une
    implémentation certifiée EBU R128 complète, mais une approximation utile
    pour le pilotage de la sélection de preset.
    """
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    win = int(3.0 * sample_rate)
    hop = int(1.0 * sample_rate)
    if mono.size < win:
        return None

    meter = _meter(sample_rate)
    st_loudness = []
    for start in range(0, mono.size - win, hop):
        chunk = mono[start : start + win]
        try:
            lufs = meter.integrated_loudness(chunk)
        except Exception:
            continue
        if np.isfinite(lufs):
            st_loudness.append(lufs)

    if len(st_loudness) < 4:
        return None

    arr = np.array(st_loudness)
    relative_gate = np.mean(arr) - 20.0
    gated = arr[arr > relative_gate]
    if gated.size < 2:
        gated = arr
    lo, hi = np.percentile(gated, [10, 95])
    return float(hi - lo)


def measure(buffer: AudioBuffer) -> LoudnessMeasurement:
    meter = _meter(buffer.sample_rate)
    integrated = meter.integrated_loudness(buffer.samples)
    true_peak = measure_true_peak_dbtp(buffer.samples, buffer.sample_rate)
    lra = measure_loudness_range(buffer.as_stereo(), buffer.sample_rate)
    return LoudnessMeasurement(
        integrated_lufs=float(integrated),
        true_peak_dbtp=true_peak,
        loudness_range_lu=lra,
        sample_peak_dbfs=buffer.peak_dbfs(),
    )


def gain_to_target_lufs(
    buffer: AudioBuffer,
    target_lufs: float = DEFAULT_GAIN_STAGING_LUFS,
    ceiling_dbtp: float = -1.0,
) -> tuple[AudioBuffer, float]:
    """
    Applique un gain (linéaire, simple trim) pour amener la loudness intégrée
    à `target_lufs`, puis vérifie que le true peak résultant ne dépasse pas
    `ceiling_dbtp` ; si c'est le cas, on réduit le gain pour respecter le
    plafond de peak plutôt que la cible LUFS exacte (on privilégie la
    sécurité anti-clip du plugin en aval).

    Retourne (buffer_gainé, gain_db_appliqué).
    """
    measurement = measure(buffer)
    if not np.isfinite(measurement.integrated_lufs):
        # Signal trop silencieux / non mesurable : on ne touche pas au gain.
        return buffer, 0.0

    gain_db = target_lufs - measurement.integrated_lufs
    projected_peak = measurement.true_peak_dbtp + gain_db
    if projected_peak > ceiling_dbtp:
        gain_db -= projected_peak - ceiling_dbtp

    gain_linear = 10.0 ** (gain_db / 20.0)
    gained_samples = (buffer.samples * gain_linear).astype(np.float32)
    gained_buffer = AudioBuffer(
        samples=gained_samples,
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
    )
    return gained_buffer, gain_db


def apply_gain_db(buffer: AudioBuffer, gain_db: float) -> AudioBuffer:
    gain_linear = 10.0 ** (gain_db / 20.0)
    return AudioBuffer(
        samples=(buffer.samples * gain_linear).astype(np.float32),
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
    )


def limit_true_peak(
    buffer: AudioBuffer,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
) -> AudioBuffer:
    """
    Limiteur "brickwall" simple à lookahead, gain lié stéréo (même gain
    appliqué à tous les canaux pour préserver l'image stéréo), pour garantir
    le plafond de true peak du profil de destination en fin de chaîne.

    Ce n'est pas un remplacement d'un vrai limiter mastering (pas de
    ISP-lookahead multi-bande, pas de saturation contrôlée), c'est un
    filet de sécurité final après la chaîne de plugins AU. Si tu préfères,
    tu peux le désactiver et faire le leveling final uniquement via un de
    tes plugins AU (ex: si tu ajoutes un limiter à ta bibliothèque de
    presets 'leveling').
    """
    stereo = buffer.as_stereo() if buffer.samples.ndim > 1 else buffer.samples[:, None]
    ceiling_linear = 10.0 ** (ceiling_dbtp / 20.0)

    # Suréchantillonnage pour estimer les true peaks inter-échantillons.
    oversampled = resample_poly(stereo, TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    link = np.max(np.abs(oversampled), axis=1)  # enveloppe liée stéréo, sur-échantillonnée

    needed_gain = np.ones_like(link)
    over = link > ceiling_linear
    needed_gain[over] = ceiling_linear / (link[over] + 1e-12)

    lookahead_samples = max(1, int(lookahead_ms * 1e-3 * buffer.sample_rate * TRUE_PEAK_OVERSAMPLE))
    from scipy.ndimage import minimum_filter1d

    # Lookahead : la réduction de gain doit commencer AVANT le pic -> fenêtre
    # glissante vers l'avant (origin négatif = regarde vers le futur).
    origin = -(lookahead_samples // 2) if lookahead_samples % 2 else -(lookahead_samples // 2 - 1)
    gain_env = minimum_filter1d(needed_gain, size=lookahead_samples, origin=origin, mode="nearest")

    # Lissage release (la remontée de gain vers 1.0 est progressive ; la
    # descente, elle, doit rester instantanée pour ne pas laisser passer de pic).
    release_coeff = np.exp(-1.0 / (release_ms * 1e-3 * buffer.sample_rate * TRUE_PEAK_OVERSAMPLE))
    smoothed = np.empty_like(gain_env)
    current = 1.0
    for i, g in enumerate(gain_env):
        if g < current:
            current = g
        else:
            current = release_coeff * current + (1 - release_coeff) * g
        smoothed[i] = current

    # Retour à la fréquence d'échantillonnage d'origine.
    gain_at_original_rate = smoothed[::TRUE_PEAK_OVERSAMPLE][: stereo.shape[0]]
    if gain_at_original_rate.shape[0] < stereo.shape[0]:
        pad = stereo.shape[0] - gain_at_original_rate.shape[0]
        gain_at_original_rate = np.pad(gain_at_original_rate, (0, pad), mode="edge")

    limited = stereo * gain_at_original_rate[:, None]

    if buffer.samples.ndim == 1:
        out_samples = limited[:, 0].astype(np.float32)
    else:
        out_samples = limited.astype(np.float32)

    return AudioBuffer(samples=out_samples, sample_rate=buffer.sample_rate, source_path=buffer.source_path)
