"""
Génère automatiquement les fixtures audio (tests/fixtures/*.wav) avant de
lancer les tests, si elles n'existent pas déjà. Les WAV de test ne sont pas
versionnés dans git (fichiers binaires générés), donc ceci rend `pytest`
utilisable directement après un clone, en local comme en CI.
"""

from __future__ import annotations

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
