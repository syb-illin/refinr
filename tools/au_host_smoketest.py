#!/usr/bin/env python3
"""
Smoke test du hosting AU (refinr/au_host.py) — À LANCER SUR MACOS.

Vérifie la mécanique de bout en bout (recherche du composant, instanciation,
initialisation, callback de rendu, AudioUnitRender) avec un Audio Unit
SYSTÈME (Apple AULowpass, toujours présent sur macOS), donc sans dépendre de
tes plugins commerciaux ni d'un .aupreset réel. Objectif : isoler les
éventuels problèmes de bas niveau (ctypes, layout de structures, constantes)
de ceux liés à un plugin tiers en particulier.

Usage:
    python3 tools/au_host_smoketest.py chemin/vers/un.wav

Si ça passe, tu peux enchaîner avec un vrai .aupreset FabFilter/Saturn2/J37 :
    python3 -c "
from refinr.audio_io import load_wav, save_wav
from refinr.preset_types import load_aupreset
from refinr.au_host import process_chain_offline
buf = load_wav('in.wav')
preset = load_aupreset('config/presets/eq/mon_preset.aupreset')
out_buf, result = process_chain_offline(buf, [preset])
save_wav(out_buf, 'out.wav')
print(result)
"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.audio_io import load_wav, save_wav  # noqa: E402
from refinr.au_host import process_chain_offline  # noqa: E402
from refinr.preset_types import PluginPreset  # noqa: E402


def make_system_lowpass_preset() -> PluginPreset:
    """AU Apple 'AULowpass' (type=aufx, subtype=lpas, manufacturer=appl), sans état particulier."""
    return PluginPreset(
        name="System AULowpass (smoketest)",
        source_path=Path("<system>"),
        component_type="aufx",
        component_subtype="lpas",
        component_manufacturer="appl",
        full_state={},  # pas d'état à restaurer, on teste juste le hosting
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} chemin/vers/un.wav")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = str(Path(input_path).with_suffix(".smoketest_out.wav"))

    print("[1/3] Chargement du WAV et de l'AU système AULowpass...")
    buffer = load_wav(input_path)
    preset = make_system_lowpass_preset()

    print(f"  {buffer.n_frames} frames @ {buffer.sample_rate}Hz, {buffer.n_channels} canal(aux)")

    print("[2/3] Rendu offline via AudioUnitRender (ctypes)...")
    out_buffer, result = process_chain_offline(buffer, [preset])
    print(f"  -> {result.frames_rendered} frames rendues (inclut le flush de queue)")

    save_wav(out_buffer, output_path)
    print(f"  écrit: {output_path}")

    print("[3/3] OK — la mécanique de base fonctionne. Tu peux passer à un vrai plugin/.aupreset.")


if __name__ == "__main__":
    main()
