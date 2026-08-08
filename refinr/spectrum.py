"""
Données de visualisation temps réel pour la GUI : courbe spectrale (pour
l'overlay "brut vs raffiné" façon FabFilter Pro-Q4), points de goniomètre
(Lissajous stéréo) et métadonnées fichier (sample rate / bit depth / durée).

Séparé de `analysis.py` volontairement : ce module ne calcule QUE ce qui
sert à dessiner (courbes prêtes à tracer), pas des features de décision. Pas
de dépendance à PyQt6 ici — reste testable/importable hors macOS.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from .audio_io import AudioBuffer

# Bornes audibles standard pour l'axe fréquentiel (log) façon EQ pro.
SPECTRUM_MIN_HZ = 20.0
SPECTRUM_MAX_HZ = 20000.0
SPECTRUM_DB_FLOOR = -80.0  # plancher d'affichage, pas une mesure réelle


@dataclasses.dataclass
class AudioFileInfo:
    sample_rate: int
    channels: int
    duration_seconds: float
    subtype: str  # ex "PCM_16", "PCM_24", "FLOAT" — bit depth lisible via SUBTYPE_BITS
    frames: int

    @property
    def bit_depth_label(self) -> str:
        return _SUBTYPE_BIT_LABELS.get(self.subtype, self.subtype)


_SUBTYPE_BIT_LABELS = {
    "PCM_16": "16-bit",
    "PCM_24": "24-bit",
    "PCM_32": "32-bit int",
    "FLOAT": "32-bit float",
    "DOUBLE": "64-bit float",
}


def file_info(path: str | Path) -> AudioFileInfo:
    """Lit les métadonnées d'un WAV SANS charger tout le signal en mémoire
    (soundfile.info est quasi instantané, même sur un gros fichier) — utilisé
    pour l'affichage immédiat rate/bits/durée dans la GUI."""
    info = sf.info(str(path))
    return AudioFileInfo(
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
        subtype=info.subtype,
        frames=info.frames,
    )


def compute_spectrum_db(
    buffer: AudioBuffer,
    n_points: int = 300,
    min_hz: float = SPECTRUM_MIN_HZ,
    max_hz: float = SPECTRUM_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Courbe spectrale lissée pour affichage EQ (freqs en Hz, magnitude en dB,
    0dB = référence RMS globale du signal) — `n_points` échantillons
    log-espacés entre `min_hz` et `max_hz`, façon analyseur Pro-Q4.
    Pas la même chose que `analysis.SpectralProfile` (features de décision
    par bande) : ici c'est une courbe continue destinée au dessin.
    """
    mono = buffer.samples.mean(axis=1) if buffer.samples.ndim > 1 else buffer.samples
    nyquist = buffer.sample_rate / 2.0
    max_hz = min(max_hz, nyquist * 0.999)

    freqs, psd = welch(mono, fs=buffer.sample_rate, nperseg=min(8192, max(256, len(mono))))
    psd = np.maximum(psd, 1e-20)
    rms_ref = float(np.sqrt(np.mean(mono**2))) + 1e-12
    ref_power = rms_ref**2

    log_targets = np.logspace(np.log10(min_hz), np.log10(max_hz), n_points)
    interp_db = np.interp(log_targets, freqs, 10.0 * np.log10(psd / ref_power), left=SPECTRUM_DB_FLOOR)
    interp_db = np.clip(interp_db, SPECTRUM_DB_FLOOR, None)
    return log_targets, interp_db


def compute_correlation(buffer: AudioBuffer) -> float:
    """
    Corrélation stéréo Pearson (-1..1) d'un buffer (typiquement une FENÊTRE
    courte extraite en temps réel pendant la lecture, voir
    `gui/analyzer_widgets.ABPlayerWidget`) — même formule que
    `analysis.DynamicsProfile.stereo_correlation`, dupliquée ici pour que ce
    module reste sans dépendance à `analysis.py` (voir docstring du module).
    1.0 si un canal est silencieux (évite une division par ~0 bruitée).
    """
    stereo = buffer.as_stereo()
    if stereo.shape[0] == 0:
        return 1.0
    left, right = stereo[:, 0], stereo[:, 1]
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def extract_window(buffer: AudioBuffer, center_sample: int, window_samples: int) -> AudioBuffer:
    """
    Extrait une fenêtre de `window_samples` échantillons centrée sur
    `center_sample` (bornée aux limites du buffer) — utilisé pour l'analyse
    "live" pendant la lecture (spectre/goniomètre/corrélation glissants,
    voir `ABPlayerWidget`), par opposition aux fonctions ci-dessus qui
    analysent le fichier entier.
    """
    n = buffer.n_frames
    half = window_samples // 2
    start = max(0, min(center_sample - half, max(0, n - window_samples)))
    end = min(n, start + window_samples)
    return AudioBuffer(samples=buffer.samples[start:end], sample_rate=buffer.sample_rate)


def compute_correlation_timeline(buffer: AudioBuffer, n_windows: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrélation stéréo PAR FENÊTRE le long du morceau entier (pas un chiffre
    unique moyen, pas une fenêtre "live" instantanée) — sert au graphique
    "placement du morceau sur la durée" (largeur stéréo dans le temps,
    voir `gui/analyzer_widgets.StereoWidthTimelineWidget`), pour repérer OÙ
    dans le morceau un problème de phase ou un rétrécissement stéréo se
    produit, plutôt qu'un score global qui peut masquer un souci localisé.

    Retourne (temps_sec, correlations) : `temps_sec[i]` = centre de la
    fenêtre i, `correlations[i]` = corrélation Pearson de cette fenêtre
    (voir `compute_correlation`). `n_windows` fenêtres réparties sur toute
    la durée (fenêtres non chevauchantes, dernière tronquée si besoin).
    """
    n = buffer.n_frames
    if n == 0 or n_windows <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    win_size = max(1, n // n_windows)
    times = []
    correlations = []
    for start in range(0, n, win_size):
        end = min(n, start + win_size)
        window = AudioBuffer(samples=buffer.samples[start:end], sample_rate=buffer.sample_rate)
        center_sec = (start + end) / 2.0 / buffer.sample_rate
        times.append(center_sec)
        correlations.append(compute_correlation(window))

    return np.array(times, dtype=np.float64), np.array(correlations, dtype=np.float64)


def compute_goniometer_points(buffer: AudioBuffer, max_points: int = 4000) -> np.ndarray:
    """
    Points (x, y) pour un goniomètre standard : x = (L-R)/sqrt(2) (Side),
    y = (L+R)/sqrt(2) (Mid) — mono parfait = ligne verticale, phase inversée
    = ligne horizontale, stéréo large = motif ouvert. Sous-échantillonne à
    `max_points` pour rester fluide à dessiner même sur un fichier long.
    """
    stereo = buffer.as_stereo()
    n = stereo.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        stereo = stereo[idx]

    left, right = stereo[:, 0], stereo[:, 1]
    x = (left - right) / np.sqrt(2.0)
    y = (left + right) / np.sqrt(2.0)
    return np.stack([x, y], axis=1).astype(np.float32)
