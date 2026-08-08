"""
Mesure de référence via `libebur128` (bindings `pyebur128`) — la même
bibliothèque C utilisée par ffmpeg (`loudnorm`/`ebur128`), VLC, et la
plupart des loudness meters "pro" du marché (Youlean, etc.), donc une
implémentation de référence de l'ITU-R BS.1770/EBU R128, indépendante de
notre propre mesure (`loudness.py`, basée sur `pyloudnorm` + un
suréchantillonnage maison pour le true peak).

Utilisée UNIQUEMENT comme contre-vérification finale avant de considérer un
export terminé (voir `chain.py::_validate_output`) : deux implémentations
indépendantes qui s'accordent donnent une vraie garantie, une seule ne
donne qu'une confiance dans son propre code. Si `pyebur128` n'est pas
installé, la contre-vérification est simplement indisponible (dégradation
gracieuse, avertissement explicite dans le rapport) — le reste du pipeline
n'en dépend pas.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .audio_io import AudioBuffer

try:
    import pyebur128 as _ebur128

    REFERENCE_LOUDNESS_AVAILABLE = True
except ImportError:  # pragma: no cover - dégradation gracieuse si non installé
    _ebur128 = None
    REFERENCE_LOUDNESS_AVAILABLE = False


@dataclasses.dataclass
class ReferenceMeasurement:
    integrated_lufs: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    loudness_range_lu: float | None
    source: str = "libebur128"


def measure_reference(buffer: AudioBuffer) -> ReferenceMeasurement | None:
    """Mesure LUFS/true-peak/LRA via libebur128. Retourne None si `pyebur128`
    n'est pas installé — à vérifier via `REFERENCE_LOUDNESS_AVAILABLE`
    avant d'afficher une contre-vérification à l'utilisateur."""
    if not REFERENCE_LOUDNESS_AVAILABLE:
        return None

    stereo = buffer.as_stereo()
    n_channels = stereo.shape[1]
    n_frames = stereo.shape[0]
    if n_frames == 0:
        return None

    interleaved = np.ascontiguousarray(stereo, dtype=np.float64).reshape(-1)

    mode = (
        _ebur128.MeasurementMode.MODE_I
        | _ebur128.MeasurementMode.MODE_LRA
        | _ebur128.MeasurementMode.MODE_TRUE_PEAK
        | _ebur128.MeasurementMode.MODE_SAMPLE_PEAK
    )
    state = _ebur128.R128State(n_channels, buffer.sample_rate, mode)
    state.add_frames(interleaved, n_frames)

    integrated_lufs = _ebur128.get_loudness_global(state)

    true_peaks = [_ebur128.get_true_peak(state, ch) for ch in range(n_channels)]
    true_peak_linear = max(true_peaks) if true_peaks else 0.0
    true_peak_dbtp = 20.0 * np.log10(true_peak_linear) if true_peak_linear > 0 else -np.inf

    sample_peaks = [_ebur128.get_sample_peak(state, ch) for ch in range(n_channels)]
    sample_peak_linear = max(sample_peaks) if sample_peaks else 0.0
    sample_peak_dbfs = 20.0 * np.log10(sample_peak_linear) if sample_peak_linear > 0 else -np.inf

    try:
        lra = _ebur128.get_loudness_range(state)
    except ValueError:  # signal trop court pour un LRA fiable
        lra = None

    return ReferenceMeasurement(
        integrated_lufs=float(integrated_lufs),
        true_peak_dbtp=float(true_peak_dbtp),
        sample_peak_dbfs=float(sample_peak_dbfs),
        loudness_range_lu=float(lra) if lra is not None and np.isfinite(lra) else None,
    )
