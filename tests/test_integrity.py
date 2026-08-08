"""Tests du module integrity.py — contrôle qualité "pré-mastering".

Tournent sur n'importe quel OS (numpy/scipy uniquement)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.audio_io import AudioBuffer
from refinr.integrity import MONO_FOLD_LOSS_ISSUE_DB, check_integrity

SR = 44100


def _tone(freq: float, seconds: float, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


def _broadband_clean_signal(seconds: float = 2.0, amplitude: float = 0.12) -> np.ndarray:
    """Bruit blanc filtré passe-haut à 20Hz — spectre plein (contrairement à
    un ton pur), représentatif d'un signal musical "propre" pour les tests
    de non-régression (aucun des checks d'intégrité ne doit se déclencher)."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(7)
    noise = rng.normal(0, amplitude, int(seconds * SR)).astype(np.float32)
    sos = butter(4, 20, btype="high", fs=SR, output="sos")
    return sosfiltfilt(sos, noise).astype(np.float32)


def test_clean_signal_has_no_issues():
    mono = _broadband_clean_signal()
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert not report.has_nan
    assert not report.has_inf
    assert not report.dc_offset_issue
    assert report.clip_event_count == 0
    assert not report.mono_fold_issue
    assert report.issue_tags() == []


def test_nan_detected():
    mono = _tone(1000, 1.0)
    mono[100] = np.nan
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.has_nan
    assert "corrupted_samples" in report.issue_tags()


def test_inf_detected():
    mono = _tone(1000, 1.0)
    mono[100] = np.inf
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.has_inf
    assert "corrupted_samples" in report.issue_tags()


def test_dc_offset_detected():
    mono = _tone(1000, 2.0) + 0.05  # offset DC nettement au-dessus du seuil
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.dc_offset_issue
    assert "dc_offset_detected" in report.issue_tags()


def test_no_dc_offset_on_centered_signal():
    mono = _tone(1000, 2.0)
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert not report.dc_offset_issue


def test_leading_and_trailing_silence_detected():
    tone = _tone(1000, 1.0)
    silence = np.zeros(int(0.8 * SR), dtype=np.float32)
    mono = np.concatenate([silence, tone, silence])
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.leading_silence_sec > 0.5
    assert report.trailing_silence_sec > 0.5
    tags = report.issue_tags()
    assert "leading_silence" in tags
    assert "trailing_silence" in tags


def test_no_edge_silence_on_continuous_tone():
    mono = _tone(1000, 2.0)
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.leading_silence_sec < 0.1
    assert report.trailing_silence_sec < 0.1


def test_channel_balance_issue_detected():
    left = _tone(1000, 2.0, amplitude=0.5)
    right = _tone(1000, 2.0, amplitude=0.1)  # bien plus faible -> déséquilibre net
    stereo = np.stack([left, right], axis=1)
    report = check_integrity(AudioBuffer(samples=stereo, sample_rate=SR))
    assert report.channel_balance_issue
    assert report.channel_balance_db is not None
    assert report.channel_balance_db < -1.0  # right plus faible que left
    assert "channel_balance_off" in report.issue_tags()


def test_balanced_channels_no_issue():
    mono = _tone(1000, 2.0)
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert not report.channel_balance_issue


def test_hard_clipping_run_detected():
    mono = _tone(1000, 1.0, amplitude=0.9)
    mono[1000:1010] = 1.0  # run de 10 échantillons collés au plafond
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.clip_event_count >= 1
    assert report.longest_clip_run_ms > 0
    assert "hard_clipping_detected" in report.issue_tags()


def test_isolated_near_peak_sample_not_counted_as_hard_clip():
    """Un seul échantillon proche de 0dBFS est un pic musical normal, pas
    un run de clipping — ne doit PAS déclencher l'issue."""
    mono = _tone(1000, 1.0, amplitude=0.9)
    mono[1000] = 0.999  # isolé, pas un run
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.clip_event_count == 0


def test_lossy_source_suspected_on_bandlimited_signal():
    """Un signal artificiellement limité à 8kHz dans un fichier 44.1kHz
    imite la signature d'une source transcodée en lossy (mp3 128-192kbps)."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, int(2.0 * SR)).astype(np.float32)
    sos = butter(8, 8000, btype="low", fs=SR, output="sos")
    bandlimited = sosfiltfilt(sos, noise).astype(np.float32)

    report = check_integrity(AudioBuffer(samples=_stereo(bandlimited), sample_rate=SR))
    assert report.lossy_source_suspected
    assert report.spectral_rolloff_hz < 16000.0


def test_fullband_noise_not_flagged_as_lossy():
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, int(2.0 * SR)).astype(np.float32)
    report = check_integrity(AudioBuffer(samples=_stereo(noise), sample_rate=SR))
    assert not report.lossy_source_suspected


def test_band_stereo_correlation_high_on_mono_source():
    mono = _broadband_clean_signal()
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert all(corr > 0.9 for corr in report.band_stereo_correlation.values())
    assert report.localized_phase_issue_bands == []


def test_band_stereo_correlation_low_on_inverted_channel():
    mono = _broadband_clean_signal()
    stereo = np.stack([mono, -mono], axis=1)  # phase inversée -> anti-corrélé
    report = check_integrity(AudioBuffer(samples=stereo, sample_rate=SR))
    # La bande "sub" (20-60Hz) porte trop peu d'énergie après le filtre
    # passe-haut à 20Hz pour donner un constat fiable (voir seuil relatif
    # dans integrity._band_stereo_correlation) — toutes les autres bandes,
    # qui portent une énergie significative, doivent ressortir anti-corrélées.
    significant_bands = {k: v for k, v in report.band_stereo_correlation.items() if k != "sub"}
    assert all(corr < -0.5 for corr in significant_bands.values()), significant_bands
    assert set(significant_bands) <= set(report.localized_phase_issue_bands)


def test_uncorrelated_stereo_not_flagged_as_mono_fold_issue():
    """Deux canaux indépendants (bruit large-bande, seeds différents) perdent
    naturellement ~3dB au repli mono — c'est la ligne de base normale d'un
    signal stéréo non corrélé, PAS une annulation de phase à signaler."""
    left = _broadband_clean_signal()
    right = np.random.default_rng(99).normal(0, 0.12, left.shape).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 20, btype="high", fs=SR, output="sos")
    right = sosfiltfilt(sos, right).astype(np.float32)

    stereo = np.stack([left, right], axis=1)
    report = check_integrity(AudioBuffer(samples=stereo, sample_rate=SR))
    assert report.mono_fold_loss_db < 4.0, report.mono_fold_loss_db
    assert not report.mono_fold_issue
    assert "mono_fold_down_issue" not in report.issue_tags()


def test_out_of_phase_stereo_flagged_as_mono_fold_issue():
    """Canal droit inversé (hors-phase parfait) -> quasi-annulation totale au
    repli mono, un vrai problème de compatibilité mono à signaler."""
    mono = _broadband_clean_signal()
    stereo = np.stack([mono, -mono], axis=1)
    report = check_integrity(AudioBuffer(samples=stereo, sample_rate=SR))
    assert report.mono_fold_loss_db > MONO_FOLD_LOSS_ISSUE_DB
    assert report.mono_fold_issue
    assert "mono_fold_down_issue" in report.issue_tags()


def test_identical_channels_no_mono_fold_loss():
    """Un vrai mono dupliqué en stéréo (corrélation parfaite) ne doit montrer
    aucune perte au repli mono."""
    mono = _tone(1000, 1.0)
    report = check_integrity(AudioBuffer(samples=_stereo(mono), sample_rate=SR))
    assert report.mono_fold_loss_db < 0.5
    assert not report.mono_fold_issue


def test_mono_buffer_does_not_crash():
    mono = _tone(1000, 1.0)
    report = check_integrity(AudioBuffer(samples=mono, sample_rate=SR))
    assert "mono" in report.dc_offset_dbfs
    assert report.channel_balance_db is None
    assert not report.channel_balance_issue


if __name__ == "__main__":
    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for fn in test_functions:
        fn()
        print(f"OK: {fn.__name__}")
    print(f"Tous les tests integrity sont passés ({len(test_functions)}).")
