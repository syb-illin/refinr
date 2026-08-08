"""Tests du module audio_io.py — lecture/écriture WAV, y compris le dither
TPDF appliqué en sortie 16-bit. Tournent sur n'importe quel OS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.audio_io import AudioBuffer, load_wav, save_wav

SR = 44100


def _quiet_tone(seconds: float = 1.0, amplitude: float = 0.02, sr: int = SR) -> np.ndarray:
    """Signal calme, amplitude assez basse pour que la distorsion de
    quantification (sans dither) soit mesurable au 16-bit."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _buffer(samples: np.ndarray) -> AudioBuffer:
    return AudioBuffer(samples=samples, sample_rate=SR)


def test_dither_changes_output_samples_on_pcm16(tmp_path):
    """Le dither doit perturber le signal écrit (bruit ajouté avant
    quantification) — comparé à une écriture sans dither, les échantillons
    relus ne doivent PAS être bit-exact."""
    samples = _quiet_tone()
    path_no_dither = tmp_path / "no_dither.wav"
    path_dither = tmp_path / "dither.wav"

    save_wav(_buffer(samples), path_no_dither, subtype="PCM_16", dither=False)
    save_wav(_buffer(samples), path_dither, subtype="PCM_16", dither=True, _dither_rng=np.random.default_rng(1))

    a = load_wav(path_no_dither).samples
    b = load_wav(path_dither).samples
    assert not np.array_equal(a, b)
    # Le dither reste une perturbation de l'ordre du LSB, pas une distorsion audible.
    assert float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) < 0.001


def test_dither_is_deterministic_with_seeded_rng(tmp_path):
    samples = _quiet_tone()
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"

    save_wav(_buffer(samples), path_a, subtype="PCM_16", dither=True, _dither_rng=np.random.default_rng(42))
    save_wav(_buffer(samples), path_b, subtype="PCM_16", dither=True, _dither_rng=np.random.default_rng(42))

    assert np.array_equal(load_wav(path_a).samples, load_wav(path_b).samples)


def test_dither_not_applied_above_16bit(tmp_path):
    """PCM_24/PCM_32/FLOAT : la troncature est déjà largement sous tout
    plancher de bruit pertinent, dither=True ne doit rien changer."""
    samples = _quiet_tone()
    for subtype in ("PCM_24", "PCM_32", "FLOAT"):
        path_no_dither = tmp_path / f"{subtype}_no_dither.wav"
        path_dither = tmp_path / f"{subtype}_dither.wav"
        save_wav(_buffer(samples), path_no_dither, subtype=subtype, dither=False)
        save_wav(_buffer(samples), path_dither, subtype=subtype, dither=True)
        assert sf.info(str(path_no_dither)).subtype == sf.info(str(path_dither)).subtype
        a = load_wav(path_no_dither).samples
        b = load_wav(path_dither).samples
        assert np.array_equal(a, b), f"dither ne devrait pas s'appliquer sur {subtype}"


def test_dither_disabled_flag_is_bit_exact_across_runs(tmp_path):
    samples = _quiet_tone()
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    save_wav(_buffer(samples), path_a, subtype="PCM_16", dither=False)
    save_wav(_buffer(samples), path_b, subtype="PCM_16", dither=False)
    assert np.array_equal(load_wav(path_a).samples, load_wav(path_b).samples)


def test_save_wav_pcm16_dither_does_not_clip_full_scale(tmp_path):
    """Un signal déjà à pleine échelle ne doit pas être poussé hors [-1, 1]
    par le dither (voir le clip de sécurité dans save_wav)."""
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    mono = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)  # crête exactement à 1.0
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    path = tmp_path / "full_scale.wav"
    save_wav(_buffer(stereo), path, subtype="PCM_16", dither=True)
    out = load_wav(path).samples
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-6


if __name__ == "__main__":
    import tempfile

    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for fn in test_functions:
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
        print(f"OK: {fn.__name__}")
    print(f"Tous les tests audio_io sont passés ({len(test_functions)}).")
