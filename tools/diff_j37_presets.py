#!/usr/bin/env python3
"""
Outil de diff pour reverse-engineer le mapping paramètre <-> index dans le
tableau `<Parameters Type="RealWorld">` des .aupreset Waves J37 (tableau plat
non documenté — 195 tokens mesurés sur les 3 presets réels actuellement en
main, voir config/presets/tape/j37_parameter_reference.md ; le nombre exact
peut varier légèrement selon la version du plugin/preset).

Ce script ne DEVINE rien : il se contente de comparer les valeurs numériques
du tableau RealWorld entre deux (ou plus) fichiers .aupreset et de lister les
indices qui diffèrent. Il appartient à l'utilisateur d'exporter, dans Logic
Pro, des presets ne différant QUE par un seul contrôle GUI à la fois (ex:
"Speed" seul, en partant d'un preset de référence commun) — c'est la seule
façon fiable d'isoler quel(s) index correspond(ent) à quel contrôle nommé
(voir la section "Notes" de j37_parameter_reference.md : ratio 58 contrôles
GUI / 215 valeurs brutes, donc pas un mapping 1:1 trivial).

Usage:
    python3 tools/diff_j37_presets.py reference.aupreset variante_speed.aupreset
    python3 tools/diff_j37_presets.py --all config/presets/tape/*.aupreset

Avec --all : diff chaque paire du groupe (utile pour un premier passage
exploratoire sur les 3 presets déjà en main, même s'ils diffèrent par PLUSIEURS
contrôles à la fois — dans ce cas, les indices reportés sont juste "candidats
possibles", pas une identification certaine ; c'est explicitement indiqué
dans la sortie).
"""

from __future__ import annotations

import argparse
import itertools
import plistlib
import re
import sys
from pathlib import Path

_PARAMS_RE = re.compile(
    r'<Parameters\s+Type="RealWorld"[^>]*>(.*?)</Parameters>',
    re.DOTALL,
)
_FLOAT_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_xml_tree(aupreset_path: Path) -> str:
    with open(aupreset_path, "rb") as fh:
        plist = plistlib.load(fh)
    blob = plist.get("Waves_XPst")
    if blob is None:
        raise ValueError(f"{aupreset_path}: pas de clé 'Waves_XPst' (pas un preset Waves J37 ?)")
    text = bytes(blob).decode("utf-8", errors="replace")
    xml_start = text.find("<PresetChunkXMLTree")
    if xml_start == -1:
        raise ValueError(f"{aupreset_path}: <PresetChunkXMLTree> introuvable dans Waves_XPst.")
    return text[xml_start:]


def extract_realworld_values(aupreset_path: Path) -> list[float]:
    """Lit le tableau plat de floats <Parameters Type="RealWorld">...</Parameters>."""
    xml_tree = _extract_xml_tree(aupreset_path)
    match = _PARAMS_RE.search(xml_tree)
    if not match:
        raise ValueError(f'{aupreset_path}: bloc <Parameters Type="RealWorld"> introuvable.')
    tokens = _FLOAT_TOKEN_RE.findall(match.group(1))
    return [float(t) for t in tokens]


def diff_values(a: list[float], b: list[float], tolerance: float = 1e-6) -> list[tuple[int, float, float]]:
    """Retourne [(index, valeur_a, valeur_b), ...] pour chaque index qui diffère.
    Compare jusqu'à min(len(a), len(b)) ; signale une taille différente séparément."""
    diffs = []
    for i, (va, vb) in enumerate(zip(a, b, strict=False)):
        if abs(va - vb) > tolerance:
            diffs.append((i, va, vb))
    return diffs


def _report_pair(path_a: Path, path_b: Path) -> None:
    values_a = extract_realworld_values(path_a)
    values_b = extract_realworld_values(path_b)

    print(f"\n=== {path_a.name}  vs  {path_b.name} ===")
    print(f"  tokens: {len(values_a)} vs {len(values_b)}")
    if len(values_a) != len(values_b):
        print("  [!] Tailles différentes — probablement des versions de plugin différentes, diff non fiable.")
        return

    diffs = diff_values(values_a, values_b)
    if not diffs:
        print("  Aucune différence détectée (presets identiques au niveau RealWorld).")
        return

    print(f"  {len(diffs)} index diffèrent :")
    for index, va, vb in diffs:
        print(f"    [{index:3d}]  {va:>12.6f}  ->  {vb:>12.6f}   (Δ={vb - va:+.6f})")

    if len(diffs) > 1:
        print(
            "  [i] Plus d'un index a changé : ces presets diffèrent probablement par plusieurs "
            "contrôles GUI simultanément. Ces indices sont des CANDIDATS, pas une identification "
            "certaine — pour confirmer un mapping exploitable en confiance, exporter deux presets "
            "ne différant QUE par un seul contrôle (voir docstring de ce script)."
        )
    else:
        index, va, vb = diffs[0]
        print(
            f"  [i] UN SEUL index a changé ([{index}]) : si tu confirmes que ces deux presets ne "
            f"diffèrent que par un seul contrôle GUI dans Logic Pro, [{index}] est identifié avec "
            f"une forte confiance comme correspondant à ce contrôle. Documente-le dans "
            f"config/presets/tape/j37_parameter_reference.md avant de t'en servir dans un futur "
            f"j37_control.py (jamais fabriquer un mapping sans cette confirmation)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--all", action="store_true", help="Diff chaque paire du groupe fourni, pas seulement les 2 premiers fichiers."
    )
    args = parser.parse_args()

    for f in args.files:
        if not f.exists():
            print(f"[!] Introuvable: {f}", file=sys.stderr)
            sys.exit(1)

    if args.all:
        for path_a, path_b in itertools.combinations(args.files, 2):
            _report_pair(path_a, path_b)
    else:
        if len(args.files) != 2:
            print("[!] Sans --all, fournir exactement 2 fichiers à comparer.", file=sys.stderr)
            sys.exit(1)
        _report_pair(args.files[0], args.files[1])


if __name__ == "__main__":
    main()
