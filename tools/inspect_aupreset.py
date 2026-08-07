#!/usr/bin/env python3
"""
Inspecteur de fichiers .aupreset — à lancer sur le Mac, sur tes vrais exports
Pro-Q4 / Saturn 2 / HG2 / J37, AVANT d'écrire le mapping de paramètres.

Un .aupreset est un plist (XML ou binaire) contenant en général :
  - des métadonnées lisibles : "name", "manufacturer", "subtype", "type",
    "version" (des OSType encodés en entiers 4 caractères)
  - une clé "data" : le state binaire propriétaire du plugin (le format
    exact dépend du vendor — souvent illisible tel quel, parfois avec des
    fragments de texte lisibles à l'intérieur, comme chez FabFilter)

Ce script ne dépend PAS de PyObjC : plistlib (stdlib) suffit à lire le
fichier. Il tourne donc aussi bien en dev qu'en prod.

Usage:
    python3 tools/inspect_aupreset.py chemin/vers/preset.aupreset [...]
    python3 tools/inspect_aupreset.py --dump-blob chemin/vers/preset.aupreset

Ce que ça produit :
  - un résumé des métadonnées (nom, OSType manufacturer/type/subtype décodés
    en 4 caractères, taille du blob "data")
  - une extraction des chaînes ASCII imprimables trouvées dans le blob
    (utile pour repérer des noms de paramètres lisibles, ex: FabFilter
    embarque parfois du texte structuré dans son state)
  - avec --dump-blob : écrit le blob binaire brut à côté, pour diff manuel
    entre deux presets du même plugin avec un seul paramètre changé
    (méthode la plus fiable pour localiser un paramètre dans un blob
    opaque : exporter deux presets identiques sauf un seul réglage, puis
    diffser les deux .bin).
"""

from __future__ import annotations

import argparse
import plistlib
import re
import struct
import sys
from pathlib import Path


def ostype_to_str(value) -> str:
    """Décode un OSType (entier 4 octets) en chaîne 4 caractères lisible."""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, int):
        try:
            raw = struct.pack(">I", value & 0xFFFFFFFF)
        except struct.error:
            return str(value)
    else:
        return str(value)
    try:
        s = raw.decode("mac_roman")
    except UnicodeDecodeError:
        return raw.hex()
    return s if s.isprintable() else raw.hex()


def extract_printable_strings(data: bytes, min_len: int = 4) -> list[str]:
    pattern = re.compile(rb"[ -~]{%d,}" % min_len)
    found = [m.group().decode("ascii", errors="replace") for m in pattern.finditer(data)]
    # dé-duplique en gardant l'ordre
    seen = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def inspect_file(path: Path, dump_blob: bool) -> None:
    print(f"\n=== {path} ===")
    with open(path, "rb") as fh:
        raw = fh.read()

    try:
        plist = plistlib.loads(raw)
    except Exception as exc:
        print(f"  [!] Impossible de parser en plist ({exc}). Fichier peut-être corrompu ou non-plist.")
        return

    if not isinstance(plist, dict):
        print(f"  [!] Racine du plist inattendue: {type(plist)}")
        return

    for key in ("name", "manufacturer", "type", "subtype", "version"):
        if key in plist:
            value = plist[key]
            if key in ("manufacturer", "type", "subtype"):
                print(f"  {key:14s}: {value!r}  ->  '{ostype_to_str(value)}'")
            else:
                print(f"  {key:14s}: {value!r}")

    other_keys = [k for k in plist.keys() if k not in ("name", "manufacturer", "type", "subtype", "version", "data")]
    if other_keys:
        print(f"  autres clés top-level: {other_keys}")
        for k in other_keys:
            v = plist[k]
            preview = repr(v)
            if len(preview) > 200:
                preview = preview[:200] + "...(tronqué)"
            print(f"    - {k}: {preview}")

    data = plist.get("data")
    if data is None:
        print("  [i] Pas de clé 'data' — ce plugin expose peut-être ses paramètres "
              "directement comme clés du plist (cas favorable pour un mapping fin).")
        return

    data = bytes(data)
    print(f"  data: blob binaire de {len(data)} octets (state propriétaire du plugin)")

    strings = extract_printable_strings(data)
    if strings:
        print(f"  [i] {len(strings)} chaîne(s) ASCII imprimable(s) détectée(s) dans le blob "
              f"(souvent des noms de paramètres ou de presets internes) :")
        for s in strings[:40]:
            print(f"      - {s!r}")
        if len(strings) > 40:
            print(f"      ... ({len(strings) - 40} de plus, tronqué)")
    else:
        print("  [i] Aucune chaîne lisible trouvée : state très probablement entièrement "
              "binaire/compressé (courant chez Waves). Le mapping par diff binaire "
              "(--dump-blob sur deux presets quasi identiques) sera la seule approche fiable.")

    if dump_blob:
        out_path = path.with_suffix(path.suffix + ".data.bin")
        out_path.write_bytes(data)
        print(f"  [+] Blob écrit dans: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--dump-blob", action="store_true", help="Écrit le blob 'data' brut en .bin pour diff manuel")
    args = parser.parse_args()

    for f in args.files:
        if not f.exists():
            print(f"[!] Introuvable: {f}", file=sys.stderr)
            continue
        inspect_file(f, args.dump_blob)


if __name__ == "__main__":
    main()
