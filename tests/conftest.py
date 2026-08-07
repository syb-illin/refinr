"""
Génère automatiquement les fixtures audio (tests/fixtures/*.wav) avant de
lancer les tests, si elles n'existent pas déjà. Les WAV de test ne sont pas
versionnés dans git (fichiers binaires générés), donc ceci rend `pytest`
utilisable directement après un clone, en local comme en CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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


@pytest.fixture(autouse=True)
def _stub_au_hosting_on_macos(monkeypatch):
    """
    refinr.au_host n'est même importable que sur macOS (voir sa garde
    `sys.platform != "darwin"` en tête de module). Sur un Mac CI hébergé
    (ex: le runner macos-14 de build-macos-app.yml), les plugins commerciaux
    réels (FabFilter Pro-Q4, Softube, Waves) ne sont PAS installés — seul
    ton Mac personnel les a. Pour que la suite de tests reste exécutable
    partout sans dépendre de ces plugins, on stubbe process_chain_offline
    en pass-through pur sur macOS : toute la logique (analyse, décision EQ
    dynamique, gain staging, leveling, rapport) reste testée de bout en
    bout, seul le rendu DSP réel est court-circuité.

    Le rendu réel, lui, se valide manuellement sur un Mac équipé des
    plugins via tools/test_dynamic_eq.py et tools/au_host_smoketest.py —
    voir leurs docstrings.
    """
    if sys.platform != "darwin":
        yield
        return

    from refinr import au_host

    def _passthrough(buffer, presets_in_order):
        return buffer, au_host.RenderResult(frames_rendered=buffer.samples.shape[0])

    monkeypatch.setattr(au_host, "process_chain_offline", _passthrough)
    yield
