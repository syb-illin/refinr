#!/usr/bin/env python3
"""
Test de bout en bout du pilotage DYNAMIQUE de Pro-Q4 — À LANCER SUR MACOS.

Contrairement à test_proq4_real.py (qui charge un .aupreset existant), ce
script :
  1. Analyse le WAV source (refinr/analysis.py) — spectre, dynamique, stéréo.
  2. Décide des réglages EQ concrets À PARTIR de cette analyse
     (refinr/proq4_control.decide_bands) — pas de preset figé.
  3. Construit l'état binaire Pro-Q4 correspondant et l'applique réellement
     via le hosting AU (refinr/au_host.py).

Usage:
    python3 tools/test_dynamic_eq.py mon_fichier.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze  # noqa: E402
from refinr.au_host import process_chain_offline  # noqa: E402
from refinr.audio_io import load_wav, save_wav  # noqa: E402
from refinr.proq4_control import decide_bands, make_dynamic_preset  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} chemin/vers/un.wav")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = str(Path(input_path).with_suffix(".dynamic_eq_out.wav"))

    print(f"[1/4] Chargement et analyse de : {input_path}")
    buffer = load_wav(input_path)
    file_analysis = analyze(buffer)
    tags = file_analysis.summary_tags()
    print(f"  tags détectés : {tags}")
    print(f"  tilt spectral : {file_analysis.spectral.tilt_db_per_octave:.2f} dB/oct")
    print(f"  crest factor  : {file_analysis.dynamics.crest_factor_db:.2f} dB")
    print(f"  clipping ratio: {file_analysis.dynamics.clipping_ratio:.4f}")
    print(f"  corrélation stéréo: {file_analysis.dynamics.stereo_correlation:.3f}")

    print("[2/4] Décision des réglages EQ à partir de l'analyse...")
    bands = decide_bands(file_analysis)
    for b in bands:
        extra = f", dyn={b.dynamic_range_db}dB (auto={b.dynamic_auto_threshold})" if b.dynamic_range_db else ""
        slope = f", pente={b.slope_db_per_oct}dB/oct" if b.slope_db_per_oct else ""
        print(
            f"  - {b.shape:12s} {b.freq_hz:7.0f}Hz  gain={b.gain_db:+.1f}dB  Q={b.q:.2f}  "
            f"placement={b.stereo}{slope}{extra}"
        )

    print("[3/4] Construction de l'état Pro-Q4 et rendu offline réel...")
    preset = make_dynamic_preset("refinr auto EQ", bands)
    out_buffer, result = process_chain_offline(buffer, [preset])
    print(f"  -> {result.frames_rendered} frames rendues")

    save_wav(out_buffer, output_path)
    print(f"[4/4] OK — écrit : {output_path}")
    print()
    print(
        "Écoute la sortie et compare à l'original : les réglages ci-dessus "
        "doivent correspondre à ce que tu entends (tilt corrigé, dynamique "
        "sur les zones dures si clipping détecté, etc.)"
    )


if __name__ == "__main__":
    main()
