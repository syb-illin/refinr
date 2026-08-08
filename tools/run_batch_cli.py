#!/usr/bin/env python3
"""
CLI de batch, sans GUI — pratique pour scripter/automatiser, ou pour tester
le pipeline complet avant de lancer l'app graphique.

Usage:
    python3 tools/run_batch_cli.py --profile spotify fichier1.wav fichier2.wav ...
    python3 tools/run_batch_cli.py --profile youtube --workers 8 --out ~/RefinrOut *.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.batch import run_batch  # noqa: E402
from refinr.profiles import DEFAULT_PROFILES_PATH, load_profiles  # noqa: E402
from refinr.report import write_reports  # noqa: E402

DEFAULT_PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Fichiers WAV à traiter")
    parser.add_argument(
        "--profile", required=True, help="Clé du profil de destination (voir config/destination_profiles.yaml)"
    )
    parser.add_argument("--out", default=str(Path.home() / "Refinr_Output"), help="Dossier de sortie")
    parser.add_argument(
        "--presets-root", default=str(DEFAULT_PRESETS_ROOT), help="Racine de la bibliothèque de presets"
    )
    parser.add_argument("--workers", type=int, default=4, help="Nombre de process parallèles")
    parser.add_argument(
        "--subtype",
        default=None,
        choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
        help="Override manuel du bit depth de sortie — par défaut (non fourni), utilise "
        "profile.output_bit_depth (voir config/destination_profiles.yaml). Le sample rate "
        "de sortie suit toujours profile.output_sample_rate, pas d'option CLI pour ça.",
    )
    args = parser.parse_args()

    catalog = load_profiles(DEFAULT_PROFILES_PATH)
    try:
        catalog.get(args.profile)
    except KeyError as exc:
        parser.error(str(exc))

    def _on_progress(outcome):
        status = "OK" if outcome.success else "ÉCHEC"
        print(f"[{status}] {outcome.input_path}")

    result = run_batch(
        input_paths=args.files,
        output_dir=args.out,
        presets_root=args.presets_root,
        profile_key=args.profile,
        profiles_path=str(DEFAULT_PROFILES_PATH),
        max_workers=args.workers,
        output_subtype=args.subtype,
        on_progress=_on_progress,
    )

    report_paths = write_reports(result, args.out, args.profile)

    print(f"\n{len(result.succeeded)} réussi(s), {len(result.failed)} échoué(s).")
    print(f"Rapport HTML: {report_paths['html_summary']}")
    if result.failed:
        print("\nDétail des échecs :")
        for outcome in result.failed:
            print(f"--- {outcome.input_path} ---")
            print(outcome.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
