#!/usr/bin/env python3
"""
Extrait la section de CHANGELOG.md correspondant à une version donnée, pour
remplir automatiquement le corps (`body_path`) de la GitHub Release créée
par .github/workflows/build-macos-app.yml — les notes de release reflètent
alors exactement le changelog généré par le hook .githooks/post-commit,
au lieu d'une liste de commits bruts.

Usage:
    python3 tools/extract_changelog_section.py 0.1.8
    python3 tools/extract_changelog_section.py v0.1.8   # préfixe "v" toléré
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def extract_section(content: str, version: str) -> str | None:
    pattern = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None
    next_match = re.search(r"^## \[", content[match.end() :], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(content)
    return content[match.start() : end].strip()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: extract_changelog_section.py <version>", file=sys.stderr)
        return 1

    version = sys.argv[1].lstrip("v")

    if not CHANGELOG_PATH.exists():
        print(f"## v{version}\n\n(CHANGELOG.md introuvable)")
        return 0

    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    section = extract_section(content, version)
    if section is None:
        print(f"## v{version}\n\n(pas d'entrée correspondante dans CHANGELOG.md)")
        return 0

    print(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
