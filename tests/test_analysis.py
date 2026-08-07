"""Tests du module analysis — tournent sur n'importe quel OS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from tests.generate_test_audio import make_bright, make_dark, make_suno_hot_clipped, SR


def _buffer_from(samples):
    return AudioBuffer(samples=samples, sample_rate=SR)


def test_bright_signal_tagged_bright():
    a = analyze(_buffer_from(make_bright()))
    assert "bright" in a.summary_tags(), a.summary_tags()
    assert a.spectral.tilt_db_per_octave > 0


def test_dark_signal_tagged_dark():
    a = analyze(_buffer_from(make_dark()))
    assert "dark" in a.summary_tags(), a.summary_tags()
    assert a.spectral.tilt_db_per_octave < 0


def test_hot_signal_has_measurable_dynamics():
    a = analyze(_buffer_from(make_suno_hot_clipped()))
    assert a.dynamics.crest_factor_db > 0
    assert 0.0 <= a.dynamics.clipping_ratio <= 1.0


def test_different_files_get_different_tags():
    """Garde-fou anti-régression du côté 'jamais générique': deux fichiers
    différents doivent produire des analyses différentes."""
    a_bright = analyze(_buffer_from(make_bright()))
    a_dark = analyze(_buffer_from(make_dark()))
    assert set(a_bright.summary_tags()) != set(a_dark.summary_tags())


if __name__ == "__main__":
    test_bright_signal_tagged_bright()
    test_dark_signal_tagged_dark()
    test_hot_signal_has_measurable_dynamics()
    test_different_files_get_different_tags()
    print("Tous les tests analysis sont passés.")
