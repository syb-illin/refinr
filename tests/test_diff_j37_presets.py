"""Tests de l'outil de diff J37 (tools/diff_j37_presets.py) — vérifie qu'il
lit correctement le tableau RealWorld des vrais presets et détecte les
index qui diffèrent, sans jamais fabriquer un mapping nommé (ce script ne
fait QUE remonter des index candidats, pas des noms de paramètres)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.diff_j37_presets import diff_values, extract_realworld_values

PRESETS_DIR = Path(__file__).resolve().parent.parent / "config" / "presets" / "tape"


def test_extract_realworld_values_reads_real_presets():
    for name in ("j37_bass_di.aupreset", "j37_bus_insert_general.aupreset", "j37_suno_artifact_tuned.aupreset"):
        values = extract_realworld_values(PRESETS_DIR / name)
        assert len(values) > 100  # tableau plat substantiel, pas un parsing vide/cassé
        assert all(isinstance(v, float) for v in values)


def test_diff_values_finds_no_diff_on_identical_lists():
    values = extract_realworld_values(PRESETS_DIR / "j37_bass_di.aupreset")
    assert diff_values(values, values) == []


def test_diff_values_detects_real_differences_between_presets():
    a = extract_realworld_values(PRESETS_DIR / "j37_bass_di.aupreset")
    b = extract_realworld_values(PRESETS_DIR / "j37_bus_insert_general.aupreset")
    diffs = diff_values(a, b)
    assert len(diffs) > 0  # ces deux presets réels sont bien différents
    for index, va, vb in diffs:
        assert 0 <= index < min(len(a), len(b))
        assert va != vb


if __name__ == "__main__":
    test_extract_realworld_values_reads_real_presets()
    test_diff_values_finds_no_diff_on_identical_lists()
    test_diff_values_detects_real_differences_between_presets()
    print("Tous les tests diff_j37_presets sont passés.")
