#!/usr/bin/env python3
"""
Premier test de bout en bout avec un VRAI plugin Pro-Q4 — À LANCER SUR MACOS.

Contrairement à au_host_smoketest.py (qui utilise l'AU système AULowpass),
ce script charge un vrai .aupreset FabFilter Pro-Q4 et l'applique à un WAV
réel via le hosting AU ctypes (refinr/au_host.py). C'est le premier test qui
valide que la chaîne complète (recherche du composant Pro-Q4 installé,
restauration d'état via ClassInfo/NSDictionary, rendu offline) fonctionne
vraiment avec un plugin commercial, pas seulement un AU système.

Usage:
    python3 tools/test_proq4_real.py mon_fichier.wav "config/presets/eq/mon_preset.aupreset"

Si aucun chemin de preset n'est donné, utilise par défaut
config/presets/eq/stereo.aupreset (preset neutre, juste pour vérifier que le
plugin est bien trouvé et que le rendu tourne sans erreur).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.audio_io import load_wav, save_wav  # noqa: E402
from refinr.au_host import process_chain_offline  # noqa: E402
from refinr.preset_types import load_aupreset  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} chemin/vers/un.wav [chemin/vers/preset.aupreset]")
        sys.exit(1)

    input_path = sys.argv[1]
    preset_path = sys.argv[2] if len(sys.argv) > 2 else "config/presets/eq/stereo.aupreset"
    output_path = str(Path(input_path).with_suffix(".proq4_test_out.wav"))

    print(f"[1/4] Chargement du WAV source : {input_path}")
    buffer = load_wav(input_path)
    print(f"  {buffer.n_frames} frames @ {buffer.sample_rate}Hz, {buffer.n_channels} canal(aux)")

    print(f"[2/4] Chargement du preset Pro-Q4 : {preset_path}")
    preset = load_aupreset(preset_path)
    print(f"  nom='{preset.name}' type={preset.component_type} subtype={preset.component_subtype} "
          f"manufacturer={preset.component_manufacturer}")

    print("[3/4] Rendu offline via AudioUnitRender (recherche + instanciation + ClassInfo + rendu)...")
    out_buffer, result = process_chain_offline(buffer, [preset])
    print(f"  -> {result.frames_rendered} frames rendues (inclut le flush de queue de fin)")

    save_wav(out_buffer, output_path)
    print(f"[4/4] OK — écrit : {output_path}")
    print()
    print("Écoute la sortie pour vérifier que le traitement EQ a bien été appliqué "
          "(comparé au fichier source).")


if __name__ == "__main__":
    main()
