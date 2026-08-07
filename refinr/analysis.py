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

Ces features servent ensuite de clés d'entrée à `preset_mapping.py` pour
choisir, parmi les presets AU fournis par l'utilisateur, ceux qui
correspondent le mieux à CE fichier précis.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import welch

from .audio_io import AudioBuffer
from .loudness import measure, LoudnessMeasurement


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


@dataclasses.dataclass
class SpectralProfile:
    band_energy_db: dict[str, float]  # énergie relative par bande, en dB (0 = référence globale)
    spectral_centroid_hz: float
    tilt_db_per_octave: float  # pente spectrale globale (positif = brillant, négatif = sombre)


@dataclasses.dataclass
class DynamicsProfile:
    crest_factor_db: float          # peak - RMS, indicateur de compression déjà présente
    clipping_ratio: float           # fraction d'échantillons proches de 0 dBFS (proxy d'écrêtage source)
    loudness_range_lu: float | None
    stereo_correlation: float       # -1..1, 1 = mono parfait, <0 = phase problématique


@dataclasses.dataclass
class FileAnalysis:
    loudness: LoudnessMeasurement
    spectral: SpectralProfile
    dynamics: DynamicsProfile

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

        return tags


def _band_energy(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(psd[mask]))


def _spectral_profile(mono: np.ndarray, sample_rate: int) -> SpectralProfile:
    freqs, psd = welch(mono, fs=sample_rate, nperseg=min(8192, len(mono)))
    total_energy = float(np.sum(psd)) + 1e-20

    band_energy_db: dict[str, float] = {}
    for name, (lo, hi) in BANDS_HZ.items():
        e = _band_energy(freqs, psd, lo, hi)
        ratio = (e + 1e-20) / total_energy
        band_energy_db[name] = 10.0 * np.log10(ratio) if ratio > 0 else -120.0

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

    return SpectralProfile(band_energy_db=band_energy_db, spectral_centroid_hz=centroid, tilt_db_per_octave=tilt)


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
    if np.std(left) > 1e-9 and np.std(right) > 1e-9:
        corr = float(np.corrcoef(left, right)[0, 1])
    else:
        corr = 1.0

    return DynamicsProfile(
        crest_factor_db=crest_factor_db,
        clipping_ratio=clipping_ratio,
        loudness_range_lu=loudness.loudness_range_lu,
        stereo_correlation=corr,
    )


def analyze(buffer: AudioBuffer) -> FileAnalysis:
    """Point d'entrée principal : calcule l'ensemble des features pour un fichier."""
    loudness = measure(buffer)
    mono = buffer.samples.mean(axis=1) if buffer.samples.ndim > 1 else buffer.samples
    spectral = _spectral_profile(mono, buffer.sample_rate)
    dynamics = _dynamics_profile(buffer, loudness)
    return FileAnalysis(loudness=loudness, spectral=spectral, dynamics=dynamics)
