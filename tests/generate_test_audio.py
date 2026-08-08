"""
Génère des WAV synthétiques pour tester le pipeline sans dépendre de vrais
fichiers sources : un signal très chaud/écrêté (typique d'une source non
masterisée), un signal calme et dynamique, et un signal sombre vs. brillant
pour vérifier l'analyse spectrale.

Usage: python3 tests/generate_test_audio.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _tone_mix(duration_s: float, freqs_amp: list[tuple[float, float]]) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sig = np.zeros_like(t)
    for freq, amp in freqs_amp:
        sig += amp * np.sin(2 * np.pi * freq * t)
    return sig.astype(np.float32)


def make_hot_clipped_source(duration_s: float = 8.0) -> np.ndarray:
    """Simule une source non masterisée typique : chaude, quasi écrêtée, spectre dense."""
    rng = np.random.default_rng(42)
    mono = _tone_mix(duration_s, [(110, 0.5), (440, 0.4), (2200, 0.25), (6000, 0.15)])
    mono += 0.05 * rng.standard_normal(mono.shape).astype(np.float32)
    mono = mono / np.max(np.abs(mono)) * 0.98  # proche de 0 dBFS
    mono = np.clip(mono, -0.97, 0.97)  # petit écrêtage volontaire
    stereo = np.stack([mono, mono * 0.98], axis=1)
    return stereo.astype(np.float32)


def make_calm_dynamic(duration_s: float = 8.0) -> np.ndarray:
    """Signal calme et dynamique (crest factor élevé, loin du plafond)."""
    mono = _tone_mix(duration_s, [(220, 0.3), (880, 0.15)])
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * np.linspace(0, duration_s, mono.size))
    mono = (mono * envelope * 0.3).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    return stereo.astype(np.float32)


def make_bright(duration_s: float = 6.0) -> np.ndarray:
    mono = _tone_mix(duration_s, [(4000, 0.3), (8000, 0.3), (12000, 0.2)])
    mono = (mono / np.max(np.abs(mono)) * 0.7).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def make_suno_artifact_like(duration_s: float = 6.0) -> np.ndarray:
    """
    Signal synthétique avec énergie concentrée dans les deux bandes KB
    documentées comme artefacts Suno (metallic_4k: 3.5-5kHz, hf_fizz_14k:
    14-20kHz), en plus d'un fondamental grave pour rester un signal "normal"
    par ailleurs. Amplitudes (220Hz/0.2, 4200Hz/0.4, 16500Hz/0.8) calibrées
    empiriquement pour placer les DEUX bandes nettement au-dessus de
    `analysis.KB_BAND_ELEVATED_THRESHOLD_DB` (3.0dB) — voir
    `refinr.analysis._band_density_relative_db`. Sert à tester le
    déclenchement MESURÉ (pas a priori) de `proq4_control.decide_bands` en
    `suno_mode=True`, par opposition à `make_bright()` (une seule bande
    élevée) et `make_calm_dynamic()` (aucune bande élevée).
    """
    mono = _tone_mix(duration_s, [(220, 0.2), (4200, 0.4), (16500, 0.8)])
    mono = (mono / np.max(np.abs(mono)) * 0.7).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def make_dark(duration_s: float = 6.0) -> np.ndarray:
    mono = _tone_mix(duration_s, [(80, 0.4), (160, 0.3), (300, 0.2)])
    mono = (mono / np.max(np.abs(mono)) * 0.7).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "hot_clipped_source.wav": make_hot_clipped_source(),
        "calm_dynamic.wav": make_calm_dynamic(),
        "bright.wav": make_bright(),
        "dark.wav": make_dark(),
    }
    for name, data in files.items():
        path = out_dir / name
        sf.write(str(path), data, SR, subtype="PCM_24")
        print(f"écrit: {path}")


if __name__ == "__main__":
    main()
