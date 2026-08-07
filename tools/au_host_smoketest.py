#!/usr/bin/env python3
"""
Smoke test du hosting AU (refinr/au_host.py) — À LANCER SUR MACOS.

Vérifie la mécanique de bout en bout (instanciation AU async, application
d'un état, rendu offline via AVAudioEngine manual rendering mode) avec un
Audio Unit SYSTÈME (Apple AULowpass, toujours présent sur macOS), donc sans
dépendre de tes plugins commerciaux ni d'un .aupreset réel. Objectif :
isoler les éventuels problèmes de sélecteurs PyObjC de ceux liés à un
plugin tiers en particulier.

Usage:
    python3 tools/au_host_smoketest.py chemin/vers/un.wav

Si ça passe, tu peux enchaîner avec un vrai .aupreset FabFilter/Saturn2/J37 :
    python3 -c "
from refinr.preset_types import load_aupreset
from refinr.au_host import process_chain_offline
preset = load_aupreset('config/presets/eq/mon_preset.aupreset')
process_chain_offline('in.wav', 'out.wav', [preset])
"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.au_host import instantiate_au, process_chain_offline, four_char_code  # noqa: E402
from refinr.preset_types import PluginPreset  # noqa: E402


def make_system_lowpass_preset() -> PluginPreset:
    """AU Apple 'AULowpass' (type=aufx, subtype=lpas, manufacturer=appl)."""
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

    print("[1/3] Instanciation de l'AU système AULowpass...")
    preset = make_system_lowpass_preset()
    # fullState vide -> on skippe setFullState_ pour ce smoketest minimal
    from AVFoundation import AVAudioUnit
    from AudioToolbox import AudioComponentDescription

    desc = AudioComponentDescription(
        componentType=four_char_code("aufx"),
        componentSubType=four_char_code("lpas"),
        componentManufacturer=four_char_code("appl"),
        componentFlags=0,
        componentFlagsMask=0,
    )
    print(f"  AudioComponentDescription construit: {desc}")

    print("[2/3] Rendu offline via AVAudioEngine (manual rendering mode)...")
    result = process_chain_offline(input_path, output_path, [preset])
    print(f"  -> {result.frames_rendered} frames rendues, écrit dans {result.output_path}")

    print("[3/3] OK — la mécanique de base fonctionne. Tu peux passer à un vrai plugin/.aupreset.")


if __name__ == "__main__":
    main()
