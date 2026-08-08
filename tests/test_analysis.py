"""Tests du module analysis — tournent sur n'importe quel OS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import (
    AI_SCORE_MAX,
    DynamicsProfile,
    SpectralProfile,
    analyze,
    compute_ai_score,
)
from refinr.audio_io import AudioBuffer
from refinr.integrity import IntegrityReport
from tests.generate_test_audio import SR, make_bright, make_dark, make_hot_clipped_source


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
    a = analyze(_buffer_from(make_hot_clipped_source()))
    assert a.dynamics.crest_factor_db > 0
    assert 0.0 <= a.dynamics.clipping_ratio <= 1.0


def test_different_files_get_different_tags():
    """Garde-fou anti-régression du côté 'jamais générique': deux fichiers
    différents doivent produire des analyses différentes."""
    a_bright = analyze(_buffer_from(make_bright()))
    a_dark = analyze(_buffer_from(make_dark()))
    assert set(a_bright.summary_tags()) != set(a_dark.summary_tags())


def test_ai_score_is_present_and_in_range():
    a = analyze(_buffer_from(make_bright()))
    assert 0.0 <= a.ai_score <= AI_SCORE_MAX
    assert round(a.ai_score, 1) == a.ai_score  # une seule décimale


def _clean_spectral() -> SpectralProfile:
    return SpectralProfile(
        band_energy_db={},
        spectral_centroid_hz=1000.0,
        tilt_db_per_octave=0.0,
        kb_band_density_db={"mud_300hz": 0.0, "metallic_4k": 0.0, "hf_fizz_14k": 0.0},
    )


def _clean_dynamics(crest_factor_db: float = 14.0) -> DynamicsProfile:
    return DynamicsProfile(
        crest_factor_db=crest_factor_db,
        clipping_ratio=0.0,
        loudness_range_lu=6.0,
        stereo_correlation=1.0,
    )


def _clean_integrity(localized_phase_issue_bands=None) -> IntegrityReport:
    return IntegrityReport(
        has_nan=False,
        has_inf=False,
        dc_offset_dbfs={"mono": -120.0},
        dc_offset_issue=False,
        leading_silence_sec=0.0,
        trailing_silence_sec=0.0,
        channel_balance_db=None,
        channel_balance_issue=False,
        clip_event_count=0,
        longest_clip_run_ms=0.0,
        noise_floor_dbfs=-60.0,
        spectral_rolloff_hz=18000.0,
        lossy_source_suspected=False,
        band_stereo_correlation={},
        localized_phase_issue_bands=localized_phase_issue_bands or [],
    )


def test_ai_score_zero_on_perfectly_clean_analysis():
    score = compute_ai_score(_clean_spectral(), _clean_dynamics(), _clean_integrity())
    assert score == 0.0


def test_ai_score_rises_with_elevated_kb_band():
    spectral = _clean_spectral()
    spectral.kb_band_density_db["metallic_4k"] = 6.0  # 3dB au-dessus du seuil -> plein poids de la bande
    score = compute_ai_score(spectral, _clean_dynamics(), _clean_integrity())
    assert score > 0.0


def test_ai_score_rises_with_collapsed_dynamics():
    score = compute_ai_score(_clean_spectral(), _clean_dynamics(crest_factor_db=4.0), _clean_integrity())
    assert score > 0.0


def test_ai_score_rises_with_localized_phase_issue():
    score = compute_ai_score(_clean_spectral(), _clean_dynamics(), _clean_integrity(["high_mid"]))
    assert score > 0.0


def test_ai_score_never_exceeds_max():
    spectral = _clean_spectral()
    for name in spectral.kb_band_density_db:
        spectral.kb_band_density_db[name] = 50.0
    score = compute_ai_score(spectral, _clean_dynamics(crest_factor_db=0.5), _clean_integrity(["mid", "high"]))
    assert score <= AI_SCORE_MAX


if __name__ == "__main__":
    test_bright_signal_tagged_bright()
    test_dark_signal_tagged_dark()
    test_hot_signal_has_measurable_dynamics()
    test_different_files_get_different_tags()
    test_ai_score_is_present_and_in_range()
    test_ai_score_zero_on_perfectly_clean_analysis()
    test_ai_score_rises_with_elevated_kb_band()
    test_ai_score_rises_with_collapsed_dynamics()
    test_ai_score_rises_with_localized_phase_issue()
    test_ai_score_never_exceeds_max()
    print("Tous les tests analysis sont passés.")
