"""
Lecture / écriture WAV et petites structures de données audio partagées.

Les sources traitées ici sont souvent en 44.1kHz/16 ou 24-bit stéréo, à des
niveaux très élevés (souvent proches de 0 dBFS voire écrêtés). On travaille en
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


# Profondeur binaire (bits) associée à chaque subtype PCM entier. Les
# subtypes flottants (FLOAT) n'ont pas de palier de quantification pertinent
# à cette échelle et ne sont jamais dithérés.
_PCM_BIT_DEPTH = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}

# En dessous de cette profondeur, la troncature float32 -> PCM sans dither
# introduit une distorsion de quantification (bruit corrélé au signal, la
# plus audible sur les fondus/queues de reverb à bas niveau) au lieu d'un
# plancher de bruit propre et indépendant du signal — pratique standard de
# mastering pro. Au-dessus (24/32-bit), l'erreur de troncature est déjà ~100+
# dB sous le niveau du signal, largement sous tout plancher de bruit
# analogique réel : le dither n'y apporte rien d'audible, on ne le fait donc
# QUE sur la réduction vers 16-bit.
DITHER_MAX_BIT_DEPTH = 16

_INT16_FULL_SCALE = 32768.0
_INT16_MIN = -32768
_INT16_MAX = 32767


def _noise_shape_channel(mono: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Quantification manuelle vers PCM 16-bit AVEC dither TPDF + noise shaping
    du second ordre par feedback d'erreur — le principe des dithers de
    référence (iZotope MBIT+, POW-r) : au lieu de laisser un bruit de
    quantification plat (TPDF simple), l'erreur du sample précédent est
    réinjectée en amont pour pousser l'essentiel de l'énergie de bruit vers
    les hautes fréquences (>10kHz), où l'oreille est nettement moins
    sensible. Ceci exige de faire la quantification NOUS-MÊMES (pas
    libsndfile) : le feedback a besoin de l'erreur de quantification réelle
    à chaque échantillon, inaccessible une fois l'écriture déléguée.

    Traité canal par canal (chaque canal a sa PROPRE récursion de feedback —
    les mélanger corromprait le shaping).
    """
    n = mono.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int16)

    # Dither TPDF pré-calculé en un seul appel vectorisé (rapide) — seule la
    # récursion de noise shaping, dépendante de l'erreur précédente, doit
    # rester séquentielle.
    dither = (rng.uniform(-0.5, 0.5, size=n) + rng.uniform(-0.5, 0.5, size=n)) / _INT16_FULL_SCALE

    # Conversion en listes Python : nettement plus rapide que l'indexation
    # scalaire numpy dans une boucle Python pure (pas de temps réel ici,
    # mais autant ne pas gaspiller de CPU pour rien).
    flat = mono.astype(np.float64).tolist()
    dither_list = dither.tolist()
    out = [0] * n

    # Coefficients de feedback du 2nd ordre (forme "noise shaper" classique,
    # stable) : e1 = erreur du sample précédent, e2 = celle d'avant.
    e1 = 0.0
    e2 = 0.0
    for i in range(n):
        shaped_input = flat[i] + 1.5 * e1 - 0.5 * e2
        ideal = shaped_input + dither_list[i]
        scaled = ideal * _INT16_FULL_SCALE
        quantized = int(scaled + 0.5) if scaled >= 0 else int(scaled - 0.5)
        if quantized > _INT16_MAX:
            quantized = _INT16_MAX
        elif quantized < _INT16_MIN:
            quantized = _INT16_MIN
        error = ideal - quantized / _INT16_FULL_SCALE
        e2 = e1
        e1 = error
        out[i] = quantized

    return np.array(out, dtype=np.int16)


def _noise_shaped_dither_to_int16(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applique `_noise_shape_channel` à chaque canal indépendamment (voir
    sa docstring) et reconstruit un tableau (n_frames,) ou (n_frames, n_channels)."""
    if samples.ndim == 1:
        return _noise_shape_channel(samples, rng)
    channels = [_noise_shape_channel(samples[:, ch], rng) for ch in range(samples.shape[1])]
    return np.stack(channels, axis=1)


def save_wav(
    buffer: AudioBuffer,
    path: str | Path,
    subtype: str = "PCM_24",
    dither: bool = True,
    _dither_rng: np.random.Generator | None = None,
) -> Path:
    """
    Écrit un WAV.

    subtype: "PCM_16", "PCM_24", "PCM_32" ou "FLOAT" (32-bit float).
    Les profils de mastering "distribution" utilisent PCM_24 par défaut ;
    on garde FLOAT disponible pour des exports intermédiaires sans perte.

    `dither` (True par défaut) : applique un dither TPDF + noise shaping du
    second ordre (voir `_noise_shape_channel`) UNIQUEMENT si `subtype` réduit
    la profondeur binaire à `DITHER_MAX_BIT_DEPTH` bits ou moins (aujourd'hui :
    PCM_16) — quantification faite manuellement dans ce cas (pas déléguée à
    libsndfile), le résultat est écrit tel quel comme échantillons entiers.
    Sans effet sur PCM_24/PCM_32/FLOAT. `_dither_rng` permet d'injecter un
    générateur seedé pour des tests déterministes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bit_depth = _PCM_BIT_DEPTH.get(subtype)
    if dither and bit_depth is not None and bit_depth <= DITHER_MAX_BIT_DEPTH:
        rng = _dither_rng if _dither_rng is not None else np.random.default_rng()
        quantized = _noise_shaped_dither_to_int16(buffer.samples, rng)
        sf.write(str(path), quantized, buffer.sample_rate, subtype=subtype)
    else:
        sf.write(str(path), buffer.samples, buffer.sample_rate, subtype=subtype)
    return path


# Beta du filtre anti-repliement Kaiser utilisé pour le rééchantillonnage
# (voir resample_if_needed). Le défaut de scipy.signal.resample_poly (5.0,
# ~-53dB d'atténuation de bande coupée) est correct pour un usage générique,
# mais nettement en dessous de ce qu'utilisent les outils de conversion de
# fréquence "mastering-grade" (r8brain, SoX -VHQ, iZotope MBIT+ visent
# ~-140dB+, filtres FIR beaucoup plus longs et dédiés). Sans aller jusque-là,
# on relève significativement l'atténuation (beta 14 -> ~-100dB) : moins de
# repliement/ringing résiduel au changement de fréquence d'échantillonnage,
# pour un coût CPU qui reste négligeable en traitement offline par batch.
RESAMPLE_KAISER_BETA = 14.0


def resample_if_needed(buffer: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Resample par interpolation polyphase (scipy), filtre anti-repliement
    Kaiser haute atténuation (voir RESAMPLE_KAISER_BETA) — qualité maximale,
    pas de compromis CPU en traitement offline."""
    if buffer.sample_rate == target_rate:
        return buffer
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(buffer.sample_rate, target_rate)
    up = target_rate // g
    down = buffer.sample_rate // g
    resampled = resample_poly(buffer.samples, up, down, axis=0, window=("kaiser", RESAMPLE_KAISER_BETA)).astype(
        np.float32
    )
    return AudioBuffer(samples=resampled, sample_rate=target_rate, source_path=buffer.source_path)
