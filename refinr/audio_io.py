"""
Lecture / écriture WAV et petites structures de données audio partagées.

Suno exporte généralement du 44.1kHz/16 ou 24-bit stéréo à des niveaux
très élevés (souvent proches de 0 dBFS voire écrêtés). On travaille en
interne en float32 pour garder de la marge, et on écrit en 24-bit ou
32-bit float en sortie selon le profil de destination.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclasses.dataclass
class AudioBuffer:
    """Buffer audio en mémoire, canaux en colonnes (frames, channels)."""

    samples: np.ndarray  # float32, shape (n_frames, n_channels)
    sample_rate: int
    source_path: Path | None = None

    @property
    def n_channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    @property
    def n_frames(self) -> int:
        return self.samples.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / float(self.sample_rate)

    def as_stereo(self) -> np.ndarray:
        """Retourne toujours un tableau (n_frames, 2), duplique le mono."""
        if self.samples.ndim == 1:
            return np.stack([self.samples, self.samples], axis=1)
        if self.samples.shape[1] == 1:
            return np.repeat(self.samples, 2, axis=1)
        return self.samples

    def peak_dbfs(self) -> float:
        peak = float(np.max(np.abs(self.samples))) if self.samples.size else 0.0
        return 20.0 * np.log10(peak) if peak > 0 else -np.inf


def load_wav(path: str | Path) -> AudioBuffer:
    """Charge un WAV en float32, quel que soit le format source (16/24/32 bit)."""
    path = Path(path)
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return AudioBuffer(samples=samples, sample_rate=sample_rate, source_path=path)


def save_wav(
    buffer: AudioBuffer,
    path: str | Path,
    subtype: str = "PCM_24",
) -> Path:
    """
    Écrit un WAV.

    subtype: "PCM_16", "PCM_24", "PCM_32" ou "FLOAT" (32-bit float).
    Les profils de mastering "distribution" utilisent PCM_24 par défaut ;
    on garde FLOAT disponible pour des exports intermédiaires sans perte.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), buffer.samples, buffer.sample_rate, subtype=subtype)
    return path


def resample_if_needed(buffer: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Resample simple par interpolation polyphase (scipy) si nécessaire."""
    if buffer.sample_rate == target_rate:
        return buffer
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(buffer.sample_rate, target_rate)
    up = target_rate // g
    down = buffer.sample_rate // g
    resampled = resample_poly(buffer.samples, up, down, axis=0).astype(np.float32)
    return AudioBuffer(samples=resampled, sample_rate=target_rate, source_path=buffer.source_path)
