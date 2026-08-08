"""Tests du contrôle actif de largeur stéréo (stereo_width.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.stereo_width import (
    NARROW_CORRELATION_THRESHOLD,
    WIDEN_CORRELATION_THRESHOLD,
    apply_stereo_width,
    decide_width_factor,
)
from tests.generate_test_audio import SR


def _very_wide_buffer(seconds: float = 3.0) -> AudioBuffer:
    """Canaux quasi indépendants au-dessus du crossover mono-safe -> corrélation basse."""
    rng = np.random.default_rng(7)
    n = int(SR * seconds)
    left = 0.3 * rng.standard_normal(n).astype(np.float32)
    right = 0.3 * rng.standard_normal(n).astype(np.float32)
    return AudioBuffer(samples=np.stack([left, right], axis=1), sample_rate=SR)


def _mono_buffer(seconds: float = 3.0) -> AudioBuffer:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


def test_decide_width_factor_narrows_on_wide_correlation():
    a = analyze(_very_wide_buffer())
    assert a.dynamics.stereo_correlation < NARROW_CORRELATION_THRESHOLD  # précondition du test
    decision = decide_width_factor(a)
    assert decision.width_factor < 1.0


def test_decide_width_factor_widens_on_near_mono_correlation():
    a = analyze(_mono_buffer())
    assert a.dynamics.stereo_correlation > WIDEN_CORRELATION_THRESHOLD  # précondition du test
    decision = decide_width_factor(a)
    assert decision.width_factor > 1.0


def test_apply_stereo_width_narrowing_increases_correlation():
    buffer = _very_wide_buffer()
    before = analyze(buffer).dynamics.stereo_correlation
    narrowed = apply_stereo_width(buffer, width_factor=0.3)
    after = analyze(narrowed).dynamics.stereo_correlation
    assert after > before


def test_apply_stereo_width_is_noop_at_factor_one():
    buffer = _very_wide_buffer()
    result = apply_stereo_width(buffer, width_factor=1.0)
    assert result is buffer


def test_apply_stereo_width_never_clips():
    buffer = _mono_buffer()
    widened = apply_stereo_width(buffer, width_factor=1.3)
    assert np.max(np.abs(widened.samples)) <= 1.0 + 1e-6


def test_apply_stereo_width_leaves_low_band_untouched():
    """Le grave (<MONO_SAFE_CROSSOVER_HZ) ne doit jamais être affecté,
    même par un facteur de largeur agressif — vérifié en comparant le
    contenu basse fréquence du canal Mid avant/après (le Mid ne dépend
    QUE du contenu commun aux deux canaux, donc invariant si le grave du
    Side n'a pas été touché)."""
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    bass = 0.3 * np.sin(2 * np.pi * 60 * t)
    left = (bass + 0.1 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    right = (bass - 0.1 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    buffer = AudioBuffer(samples=np.stack([left, right], axis=1), sample_rate=SR)

    narrowed = apply_stereo_width(buffer, width_factor=0.1)
    mid_before = (buffer.samples[:, 0] + buffer.samples[:, 1]) / 2.0
    mid_after = (narrowed.samples[:, 0] + narrowed.samples[:, 1]) / 2.0
    assert np.allclose(mid_before, mid_after, atol=1e-3)


if __name__ == "__main__":
    test_decide_width_factor_narrows_on_wide_correlation()
    test_decide_width_factor_widens_on_near_mono_correlation()
    test_apply_stereo_width_narrowing_increases_correlation()
    test_apply_stereo_width_is_noop_at_factor_one()
    test_apply_stereo_width_never_clips()
    test_apply_stereo_width_leaves_low_band_untouched()
    print("Tous les tests stereo_width sont passés.")
