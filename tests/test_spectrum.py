"""Tests du module spectrum.py (données de visualisation GUI) — tournent
sur n'importe quel OS, aucune dépendance PyQt6."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.audio_io import AudioBuffer
from refinr.spectrum import (
    SPECTRUM_DB_FLOOR,
    compute_correlation,
    compute_correlation_timeline,
    compute_goniometer_points,
    compute_spectrum_db,
    extract_window,
    file_info,
)

SR = 44100


def _tone_stereo(freq: float, seconds: float = 1.0, amplitude: float = 0.4) -> AudioBuffer:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


def test_spectrum_curve_has_requested_length_and_finite_values():
    buffer = _tone_stereo(1000)
    freqs, db = compute_spectrum_db(buffer, n_points=150)
    assert len(freqs) == 150
    assert len(db) == 150
    assert np.all(np.isfinite(freqs))
    assert np.all(np.isfinite(db))
    assert np.all(db >= SPECTRUM_DB_FLOOR)


def test_spectrum_curve_peaks_near_tone_frequency():
    buffer = _tone_stereo(1000)
    freqs, db = compute_spectrum_db(buffer, n_points=400)
    peak_freq = freqs[np.argmax(db)]
    assert 800 <= peak_freq <= 1200, peak_freq


def test_spectrum_curve_frequencies_are_ascending_and_in_audible_range():
    buffer = _tone_stereo(500)
    freqs, _db = compute_spectrum_db(buffer)
    assert np.all(np.diff(freqs) > 0)
    assert freqs[0] >= 20.0
    assert freqs[-1] <= 22050.0


def test_goniometer_mono_signal_forms_vertical_line():
    """L == R -> x = (L-R)/sqrt2 == 0 partout (ligne verticale, mono parfait)."""
    buffer = _tone_stereo(300)
    points = compute_goniometer_points(buffer, max_points=500)
    assert points.shape[1] == 2
    assert np.allclose(points[:, 0], 0.0, atol=1e-5)
    assert np.any(np.abs(points[:, 1]) > 0.01)


def test_goniometer_inverted_phase_forms_horizontal_line():
    """L == -R -> y = (L+R)/sqrt2 == 0 partout (ligne horizontale)."""
    mono = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, SR))).astype(np.float32)
    buffer = AudioBuffer(samples=np.stack([mono, -mono], axis=1), sample_rate=SR)
    points = compute_goniometer_points(buffer, max_points=500)
    assert np.allclose(points[:, 1], 0.0, atol=1e-5)


def test_goniometer_respects_max_points_downsampling():
    buffer = _tone_stereo(300, seconds=2.0)
    points = compute_goniometer_points(buffer, max_points=100)
    assert points.shape[0] == 100


def test_goniometer_empty_buffer_does_not_crash():
    buffer = AudioBuffer(samples=np.zeros((0, 2), dtype=np.float32), sample_rate=SR)
    points = compute_goniometer_points(buffer)
    assert points.shape == (0, 2)


def test_file_info_reads_metadata_without_full_decode(tmp_path):
    mono = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(SR * 1.5)))).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "probe.wav"
    sf.write(str(path), stereo, SR, subtype="PCM_24")

    info = file_info(path)
    assert info.sample_rate == SR
    assert info.channels == 2
    assert abs(info.duration_seconds - 1.5) < 0.01
    assert info.subtype == "PCM_24"
    assert info.bit_depth_label == "24-bit"


def test_file_info_unknown_subtype_falls_back_to_raw_label(tmp_path):
    mono = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(SR * 0.5)))).astype(np.float32)
    path = tmp_path / "probe16.wav"
    sf.write(str(path), mono, SR, subtype="PCM_16")
    info = file_info(path)
    assert info.bit_depth_label == "16-bit"


def test_correlation_mono_signal_is_one():
    buffer = _tone_stereo(300)
    assert compute_correlation(buffer) > 0.99


def test_correlation_inverted_phase_is_minus_one():
    mono = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, SR))).astype(np.float32)
    buffer = AudioBuffer(samples=np.stack([mono, -mono], axis=1), sample_rate=SR)
    assert compute_correlation(buffer) < -0.99


def test_correlation_empty_buffer_defaults_to_one():
    buffer = AudioBuffer(samples=np.zeros((0, 2), dtype=np.float32), sample_rate=SR)
    assert compute_correlation(buffer) == 1.0


def test_extract_window_has_requested_length_centered():
    buffer = _tone_stereo(300, seconds=2.0)
    window = extract_window(buffer, center_sample=SR, window_samples=4096)
    assert window.n_frames == 4096
    assert window.sample_rate == SR


def test_extract_window_clamps_at_start():
    buffer = _tone_stereo(300, seconds=2.0)
    window = extract_window(buffer, center_sample=0, window_samples=4096)
    assert window.n_frames == 4096  # borné au début, pas de négatif -> toujours la même taille

    window_end = extract_window(buffer, center_sample=buffer.n_frames, window_samples=4096)
    assert window_end.n_frames == 4096  # borné à la fin


def test_extract_window_larger_than_buffer_returns_whole_buffer():
    mono = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.02, int(SR * 0.02)))).astype(np.float32)
    buffer = AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)
    window = extract_window(buffer, center_sample=buffer.n_frames // 2, window_samples=8192)
    assert window.n_frames == buffer.n_frames


def test_correlation_timeline_covers_full_duration():
    buffer = _tone_stereo(300, seconds=2.0)
    times, correlations = compute_correlation_timeline(buffer, n_windows=50)
    assert len(times) == len(correlations)
    assert len(times) > 0
    assert times[0] >= 0.0
    assert times[-1] <= 2.0
    assert np.all(np.diff(times) > 0)  # strictement croissant


def test_correlation_timeline_detects_localized_phase_flip():
    """La moitié du morceau reste corrélée, l'autre bascule en phase
    inversée -> la timeline doit refléter ce changement AU BON ENDROIT,
    pas juste un score moyen qui noierait le problème."""
    half = SR  # 1 seconde à 44.1kHz
    mono = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 2 * SR))).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1).copy()
    stereo[half:, 1] *= -1  # deuxième moitié : canal droit inversé
    buffer = AudioBuffer(samples=stereo, sample_rate=SR)

    times, correlations = compute_correlation_timeline(buffer, n_windows=20)
    first_half = correlations[times < 1.0]
    second_half = correlations[times >= 1.0]
    assert np.mean(first_half) > 0.9
    assert np.mean(second_half) < -0.9


def test_correlation_timeline_empty_buffer_returns_empty_arrays():
    buffer = AudioBuffer(samples=np.zeros((0, 2), dtype=np.float32), sample_rate=SR)
    times, correlations = compute_correlation_timeline(buffer)
    assert len(times) == 0
    assert len(correlations) == 0


if __name__ == "__main__":
    import tempfile

    test_spectrum_curve_has_requested_length_and_finite_values()
    test_spectrum_curve_peaks_near_tone_frequency()
    test_spectrum_curve_frequencies_are_ascending_and_in_audible_range()
    test_goniometer_mono_signal_forms_vertical_line()
    test_goniometer_inverted_phase_forms_horizontal_line()
    test_goniometer_respects_max_points_downsampling()
    test_goniometer_empty_buffer_does_not_crash()
    test_correlation_mono_signal_is_one()
    test_correlation_inverted_phase_is_minus_one()
    test_correlation_empty_buffer_defaults_to_one()
    test_extract_window_has_requested_length_centered()
    test_extract_window_clamps_at_start()
    test_extract_window_larger_than_buffer_returns_whole_buffer()
    test_correlation_timeline_covers_full_duration()
    test_correlation_timeline_detects_localized_phase_flip()
    test_correlation_timeline_empty_buffer_returns_empty_arrays()
    with tempfile.TemporaryDirectory() as tmp:
        test_file_info_reads_metadata_without_full_decode(Path(tmp))
        test_file_info_unknown_subtype_falls_back_to_raw_label(Path(tmp))
    print("Tous les tests spectrum sont passés.")
