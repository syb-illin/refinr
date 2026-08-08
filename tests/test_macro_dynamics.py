"""Tests du rider de gain macro-dynamique piloté par la LRA (macro_dynamics.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.macro_dynamics import (
    LRA_JARRING_THRESHOLD_LU,
    apply_macro_compression,
    compute_short_term_loudness_curve,
    decide_macro_compression,
)
from tests.generate_test_audio import SR


def _jarring_sections_buffer(seconds_per_section: float = 6.0) -> AudioBuffer:
    """Alterne une section très calme et une section forte, plusieurs fois —
    imite des sauts de volume section à section (couplet/refrain extrême)."""
    t = np.linspace(0, seconds_per_section, int(SR * seconds_per_section), endpoint=False)
    quiet = (0.03 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    loud = (0.85 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    mono = np.concatenate([quiet, loud, quiet, loud])
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


def _steady_buffer(seconds: float = 12.0) -> AudioBuffer:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


def test_compute_short_term_loudness_curve_tracks_level_changes():
    buffer = _jarring_sections_buffer()
    times, values = compute_short_term_loudness_curve(buffer)
    assert times.size > 4
    assert np.max(values) - np.min(values) > 10.0  # écart net entre section calme et forte


def test_decide_macro_compression_triggers_on_high_lra():
    buffer = _jarring_sections_buffer()
    a = analyze(buffer)
    assert a.dynamics.loudness_range_lu is not None
    assert a.dynamics.loudness_range_lu > LRA_JARRING_THRESHOLD_LU  # précondition du test
    decision = decide_macro_compression(a)
    assert decision.enabled is True
    assert decision.ratio > 0.0


def test_decide_macro_compression_stays_off_on_steady_material():
    buffer = _steady_buffer()
    a = analyze(buffer)
    decision = decide_macro_compression(a)
    if a.dynamics.loudness_range_lu is None or a.dynamics.loudness_range_lu <= LRA_JARRING_THRESHOLD_LU:
        assert decision.enabled is False


def test_apply_macro_compression_reduces_measured_lra():
    buffer = _jarring_sections_buffer()
    lra_before = analyze(buffer).dynamics.loudness_range_lu
    compressed = apply_macro_compression(buffer, ratio=0.8)
    lra_after = analyze(compressed).dynamics.loudness_range_lu
    assert lra_before is not None and lra_after is not None
    assert lra_after < lra_before


def test_apply_macro_compression_never_fully_flattens_dynamics():
    """ratio < 1.0 doit réduire la LRA sans l'aplatir à (quasi) zéro —
    la musicalité doit être préservée, pas juste comprimée à mort."""
    buffer = _jarring_sections_buffer()
    compressed = apply_macro_compression(buffer, ratio=0.5)
    lra_after = analyze(compressed).dynamics.loudness_range_lu
    assert lra_after is not None
    assert lra_after > 1.0


def test_apply_macro_compression_is_noop_at_zero_ratio():
    buffer = _jarring_sections_buffer()
    result = apply_macro_compression(buffer, ratio=0.0)
    assert result is buffer


if __name__ == "__main__":
    test_compute_short_term_loudness_curve_tracks_level_changes()
    test_decide_macro_compression_triggers_on_high_lra()
    test_decide_macro_compression_stays_off_on_steady_material()
    test_apply_macro_compression_reduces_measured_lra()
    test_apply_macro_compression_never_fully_flattens_dynamics()
    test_apply_macro_compression_is_noop_at_zero_ratio()
    print("Tous les tests macro_dynamics sont passés.")
