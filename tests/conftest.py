"""
Génère automatiquement les fixtures audio (tests/fixtures/*.wav) avant de
lancer les tests, si elles n'existent pas déjà. Les WAV de test ne sont pas
versionnés dans git (fichiers binaires générés), donc ceci rend `pytest`
utilisable directement après un clone, en local comme en CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf

from tests.generate_test_audio import (
    SR,
    make_bright,
    make_calm_dynamic,
    make_dark,
    make_hot_clipped_source,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Sur un Mac CI hébergé (ex: runner macos-14 de build-macos-app.yml), les
# plugins commerciaux réels (FabFilter Pro-Q4, Softube, Waves) ne sont PAS
# installés — seul un Mac personnel les a. On désactive donc le hosting AU
# réel pour toute la session de tests via une variable d'environnement
# (voir refinr/chain.py::AU_HOSTING_AVAILABLE pour le pourquoi — un
# monkeypatch pytest ne suffit pas ici car batch.py parallélise via
# ProcessPoolExecutor, dont les workers réimportent au_host à neuf et ne
# voient donc pas un monkeypatch fait dans le process principal).
#
# Positionnée en variable d'environnement (pas via un fixture avec
# monkeypatch.setenv) pour être active AVANT que le premier worker ne soit
# spawné, et le rester pour toute la session, peu importe l'ordre des
# tests. Toute la logique (analyse, décision EQ dynamique, gain staging,
# leveling, rapport) reste testée de bout en bout ; seul le rendu DSP réel
# est court-circuité. Il se valide manuellement sur un Mac équipé des
# plugins via tools/test_dynamic_eq.py et tools/au_host_smoketest.py.
os.environ.setdefault("REFINR_TEST_DISABLE_AU_HOSTING", "1")


def pytest_configure(config):
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "hot_clipped_source.wav": make_hot_clipped_source,
        "calm_dynamic.wav": make_calm_dynamic,
        "bright.wav": make_bright,
        "dark.wav": make_dark,
    }
    for name, generator in generators.items():
        path = FIXTURES_DIR / name
        if not path.exists():
            sf.write(str(path), generator(), SR, subtype="PCM_24")
