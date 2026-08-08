"""
Analyse par fichier pour piloter une sélection de preset spécifique
(jamais un traitement générique appliqué à tous les WAV).

On extrait un jeu de "features" par fichier :
- balance spectrale (graves/médiums/aigus, centroïde spectral)
- dynamique (crest factor, loudness range)
- niveau de saturation/écrêtage déjà présent dans la source
- présence de sifflantes / dureté dans l'aigu (proxy simple, pas un
  détecteur de sifflantes dédié)
- stéréo width approximatif
- intégrité du signal et métriques alignées avec `config/suno_artifacts_kb.md`
  (voir `integrity.py`) : rien de ce qui est documenté dans la KB n'est
  laissé sans mesure explicite, même quand `suno_mode` est désactivé —
  l'analyse mesure TOUJOURS, la correction (elle) reste opt-in.

Ces features servent ensuite de clés d'entrée à `preset_mapping.py` pour
choisir, parmi les presets AU fournis par l'utilisateur, ceux qui
correspondent le mieux à CE fichier précis.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import welch

from .audio_io import AudioBuffer
from .integrity import IntegrityReport, check_integrity
from .loudness import LoudnessMeasurement, measure

# Bandes de fréquence utilisées pour la balance spectrale (Hz).
BANDS_HZ = {
    "sub": (20, 60),
    "low": (60, 250),
    "low_mid": (250, 800),
    "mid": (800, 2500),
    "high_mid": (2500, 6000),
    "high": (6000, 12000),
    "air": (12000, 20000),
}

# Bandes précises alignées avec config/suno_artifacts_kb.md — distinctes des
# 7 bandes générales ci-dessus (qui servent au tilt/balance global), pour
# mesurer explicitement les zones documentées comme problématiques sur les
# générateurs IA type Suno. Densité d'énergie (par Hz), pas énergie brute,
# pour rester comparable entre bandes de largeurs différentes.
KB_BANDS_HZ = {
    "mud_300hz": (250.0, 500.0),  # KB section 1 : boue bas-médium
    "metallic_4k": (3500.0, 5000.0),  # KB section 3 : buzz métallique vocal
    "hf_fizz_14k": (14000.0, 20000.0),  # KB section 2 : bruit de synthèse HF
}
# Seuil (dB, densité relative à la moyenne du spectre mesuré) à partir
# duquel une zone KB est jugée "notablement élevée" — cohérent en ordre de
# grandeur avec les autres seuils du projet (tilt >1.0dB/oct, crest <8dB).
# Ajustable ; pas calibré sur un corpus Suno réel (voir limites dans la KB).
KB_BAND_ELEVATED_THRESHOLD_DB = 3.0

# --- Score de suspicion "généré par IA" (ai_score) --------------------------
# Note /10, UNE décimale, PLUS HAUT = PIRE (plus de signes d'artefacts de
# génération IA type Suno). Ce n'est PAS un détecteur IA/non-IA au sens
# classifieur ML — c'est un agrégat pondéré des signatures documentées dans
# config/suno_artifacts_kb.md et déjà mesurées ailleurs dans ce module et
# dans integrity.py. Chaque composante est plafonnée individuellement puis
# sommée, le tout borné à [0, 10]. Non calibré sur un corpus Suno réel (même
# réserve que KB_BAND_ELEVATED_THRESHOLD_DB) : à interpréter comme un
# indicateur relatif d'alerte, pas une probabilité.
AI_SCORE_MAX = 10.0
AI_SCORE_BAND_WEIGHT = 2.5  # poids max par bande KB élevée (mud, metallic, fizz)
AI_SCORE_BAND_SPAN_DB = 3.0  # dB au-dessus du seuil pour atteindre le poids max de la bande
AI_SCORE_PHASE_WEIGHT = 1.25  # poids si des bandes montrent un problème de phase localisé
AI_SCORE_DYNAMICS_WEIGHT = 1.25  # poids si dynamique effondrée (crest factor très bas)
AI_SCORE_DYNAMICS_CREST_DB = 6.0  # en-dessous de ce crest factor, dynamique jugée effondrée

# --- Heuristique "contenu dominé par le bas du spectre" (low_end_dominant) --
# PAS une détection d'instrument (aucun classifieur, aucun modèle entraîné) :
# juste un test de concentration d'énergie spectrale. Un mix complet, même
# très sombre (tag "dark" ci-dessus, basé sur le tilt), garde quasi toujours
# une énergie significative en médium/aigu (voix, cymbales, présence...) ;
# un enregistrement d'un seul élément grave (bass DI, sub, kick isolé...) n'en
# a presque pas. Sert à router le preset TAPE "bass_di" (voir
# config/presets/tape/j37_bass_di.meta.yaml) vers ce type de contenu SANS
# jamais l'appliquer à un mix complet sombre par erreur. À revoir si un jour
# une vraie détection d'instrument (classifieur audio) remplace ce proxy.
LOW_END_DOMINANT_LOW_FRACTION_MIN = 0.85  # part (linéaire) de l'énergie totale dans sub+low+low_mid
LOW_END_DOMINANT_HIGH_FRACTION_MAX = 0.15  # part max dans mid+high_mid+high+air


@dataclasses.dataclass
class SpectralProfile:
    band_energy_db: dict[str, float]  # énergie relative par bande, en dB (0 = référence globale)
    spectral_centroid_hz: float
    tilt_db_per_octave: float  # pente spectrale globale (positif = brillant, négatif = sombre)
    kb_band_density_db: dict[str, float]  # densité relative par bande KB_BANDS_HZ, en dB


@dataclasses.dataclass
class DynamicsProfile:
    crest_factor_db: float  # peak - RMS, indicateur de compression déjà présente
    clipping_ratio: float  # fraction d'échantillons proches de 0 dBFS (proxy d'écrêtage source)
    loudness_range_lu: float | None
    stereo_correlation: float  # -1..1, 1 = mono parfait, <0 = phase problématique


@dataclasses.dataclass
class FileAnalysis:
    loudness: LoudnessMeasurement
    spectral: SpectralProfile
    dynamics: DynamicsProfile
    integrity: IntegrityReport
    ai_score: float  # 0.0-10.0, une décimale, PLUS HAUT = PIRE (voir compute_ai_score)

    def summary_tags(self) -> list[str]:
        """Tags courts lisibles, utilisés dans le rapport et par le mapping de presets."""
        tags = []

        if self.dynamics.clipping_ratio > 0.001:
            tags.append("clipping_detected")
        if self.dynamics.crest_factor_db < 8:
            tags.append("already_compressed")
        elif self.dynamics.crest_factor_db > 16:
            tags.append("very_dynamic")

        tilt = self.spectral.tilt_db_per_octave
        if tilt > 1.0:
            tags.append("bright")
        elif tilt < -2.5:
            tags.append("dark")
        else:
            tags.append("balanced_tonal")

        if self.dynamics.stereo_correlation < 0.2:
            tags.append("wide_stereo")
        elif self.dynamics.stereo_correlation > 0.9:
            tags.append("narrow_mono_like")

        for name, density_db in self.spectral.kb_band_density_db.items():
            if density_db > KB_BAND_ELEVATED_THRESHOLD_DB:
                tags.append(f"kb_{name}_elevated")

        if self._is_low_end_dominant():
            tags.append("low_end_dominant")

        tags.extend(self.integrity.issue_tags())

        return tags

    def _is_low_end_dominant(self) -> bool:
        """Voir LOW_END_DOMINANT_* — heuristique de concentration spectrale,
        pas une détection d'instrument."""
        fractions = {name: 10.0 ** (db / 10.0) for name, db in self.spectral.band_energy_db.items()}
        low_fraction = fractions.get("sub", 0.0) + fractions.get("low", 0.0) + fractions.get("low_mid", 0.0)
        high_fraction = (
            fractions.get("mid", 0.0)
            + fractions.get("high_mid", 0.0)
            + fractions.get("high", 0.0)
            + fractions.get("air", 0.0)
        )
        return low_fraction > LOW_END_DOMINANT_LOW_FRACTION_MIN and high_fraction < LOW_END_DOMINANT_HIGH_FRACTION_MAX


def _band_energy(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(psd[mask]))


def _band_density_relative_db(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """
    Densité d'énergie moyenne (par bin) dans [lo, hi) relative à la densité
    moyenne sur tout le spectre mesuré — en dB. Positif = zone au-dessus de
    la moyenne (pic notable), négatif = en-dessous (creux). Contrairement à
    `band_energy_db` (énergie brute sommée, qui favorise les bandes larges),
    ceci reste comparable entre bandes étroites (ex: metallic_4k, 1.5kHz de
    large) et larges (ex: hf_fizz_14k, 6kHz de large).
    """
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return -120.0
    band_mean = float(np.mean(psd[mask]))
    overall_mean = float(np.mean(psd)) + 1e-20
    ratio = (band_mean + 1e-20) / overall_mean
    return 10.0 * np.log10(ratio) if ratio > 0 else -120.0


def _spectral_profile(mono: np.ndarray, sample_rate: int) -> SpectralProfile:
    freqs, psd = welch(mono, fs=sample_rate, nperseg=min(8192, len(mono)))
    total_energy = float(np.sum(psd)) + 1e-20

    band_energy_db: dict[str, float] = {}
    for name, (lo, hi) in BANDS_HZ.items():
        e = _band_energy(freqs, psd, lo, hi)
        ratio = (e + 1e-20) / total_energy
        band_energy_db[name] = 10.0 * np.log10(ratio) if ratio > 0 else -120.0

    nyquist = sample_rate / 2.0
    kb_band_density_db: dict[str, float] = {}
    for name, (lo, hi) in KB_BANDS_HZ.items():
        if lo >= nyquist:
            continue  # bande hors du spectre mesurable à ce sample rate
        kb_band_density_db[name] = round(_band_density_relative_db(freqs, psd, lo, min(hi, nyquist)), 2)

    centroid = float(np.sum(freqs * psd) / total_energy)

    # Pente spectrale : régression log-log de la PSD sur les fréquences audibles.
    audible = (freqs >= 40) & (freqs <= 16000) & (psd > 0)
    if np.count_nonzero(audible) > 8:
        log_f = np.log2(freqs[audible])
        log_p = 10.0 * np.log10(psd[audible] + 1e-20)
        slope, _ = np.polyfit(log_f, log_p, 1)
        tilt = float(slope)
    else:
        tilt = 0.0

    return SpectralProfile(
        band_energy_db=band_energy_db,
        spectral_centroid_hz=centroid,
        tilt_db_per_octave=tilt,
        kb_band_density_db=kb_band_density_db,
    )


def _dynamics_profile(buffer: AudioBuffer, loudness: LoudnessMeasurement) -> DynamicsProfile:
    samples = buffer.samples
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples

    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    peak = float(np.max(np.abs(mono))) + 1e-12
    crest_factor_db = 20.0 * np.log10(peak / rms)

    # Proxy d'écrêtage : fraction des échantillons à moins de 0.3 dB de 0 dBFS.
    # Le clipping se produit par canal, pas sur le downmix mono : on prend le
    # max absolu inter-canaux échantillon par échantillon.
    stereo_for_clip = buffer.as_stereo()
    per_sample_peak = np.max(np.abs(stereo_for_clip), axis=1)
    threshold = 10 ** (-0.3 / 20.0)
    clipping_ratio = float(np.mean(per_sample_peak >= threshold))

    stereo = buffer.as_stereo()
    left, right = stereo[:, 0], stereo[:, 1]
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-9 and np.std(right) > 1e-9 else 1.0

    return DynamicsProfile(
        crest_factor_db=crest_factor_db,
        clipping_ratio=clipping_ratio,
        loudness_range_lu=loudness.loudness_range_lu,
        stereo_correlation=corr,
    )


def compute_ai_score(spectral: SpectralProfile, dynamics: DynamicsProfile, integrity: IntegrityReport) -> float:
    """
    Score de suspicion "généré par IA" sur 10, une décimale, PLUS HAUT = PIRE.
    Agrège les signatures de config/suno_artifacts_kb.md déjà mesurées ailleurs :
      - bandes KB élevées (boue 250-500Hz, métallique 3.5-5kHz, fizz HF 14-20kHz)
        jusqu'à AI_SCORE_BAND_WEIGHT chacune, proportionnel au dépassement du seuil
      - problème de phase localisé (integrity.localized_phase_issue_bands non vide)
      - dynamique effondrée (crest factor sous AI_SCORE_DYNAMICS_CREST_DB)
    Ne remplace pas une lecture des tags/rapport détaillé — sert d'indicateur
    d'alerte synthétique, pas de verdict binaire IA/non-IA.
    """
    score = 0.0

    for name in ("mud_300hz", "metallic_4k", "hf_fizz_14k"):
        density_db = spectral.kb_band_density_db.get(name)
        if density_db is None:
            continue
        excess = density_db - KB_BAND_ELEVATED_THRESHOLD_DB
        if excess > 0:
            fraction = min(excess / AI_SCORE_BAND_SPAN_DB, 1.0)
            score += fraction * AI_SCORE_BAND_WEIGHT

    if integrity.localized_phase_issue_bands:
        score += AI_SCORE_PHASE_WEIGHT

    if dynamics.crest_factor_db < AI_SCORE_DYNAMICS_CREST_DB:
        score += AI_SCORE_DYNAMICS_WEIGHT

    return round(min(score, AI_SCORE_MAX), 1)


def analyze(buffer: AudioBuffer) -> FileAnalysis:
    """Point d'entrée principal : calcule l'ensemble des features pour un fichier,
    y compris le contrôle d'intégrité complet (`integrity.check_integrity`) et le
    score de suspicion IA (`compute_ai_score`)."""
    loudness = measure(buffer)
    mono = buffer.samples.mean(axis=1) if buffer.samples.ndim > 1 else buffer.samples
    spectral = _spectral_profile(mono, buffer.sample_rate)
    dynamics = _dynamics_profile(buffer, loudness)
    integrity = check_integrity(buffer)
    ai_score = compute_ai_score(spectral, dynamics, integrity)
    return FileAnalysis(
        loudness=loudness,
        spectral=spectral,
        dynamics=dynamics,
        integrity=integrity,
        ai_score=ai_score,
    )
