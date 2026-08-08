"""Tests de la sélection de presets par rôle (preset_mapping.py), en
particulier le rôle TAPE (J37) une fois les 3 vrais presets utilisateur
intégrés dans config/presets/tape/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.preset_mapping import PresetLibrary, select_preset_for_role
from refinr.preset_types import PluginRole
from tests.generate_test_audio import SR, make_bright, make_calm_dynamic

PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"


def _buffer_from(samples):
    return AudioBuffer(samples=samples, sample_rate=SR)


def _make_low_end_dominant(seconds: float = 3.0) -> np.ndarray:
    """Signal synthétique dont quasi toute l'énergie est sous 200Hz — imite
    un bass DI/sub isolé (voir analysis._is_low_end_dominant)."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (0.4 * np.sin(2 * np.pi * 55 * t) + 0.15 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_tape_role_has_four_presets_loaded():
    """3 presets statiques (sélection par tags) + j37_baseline_reference,
    ce dernier servant de template au pilotage dynamique (voir
    chain._find_j37_template) plutôt qu'à la sélection par tags elle-même."""
    library = PresetLibrary.load(PRESETS_ROOT)
    entries = library.entries_by_role[PluginRole.TAPE]
    names = {e.preset.name for e in entries}
    assert names == {"for all instr bus insert", "for suno", "my bass", "j37_baseline_reference"}


def test_tape_role_defaults_to_general_bus_insert_without_kb_tags():
    library = PresetLibrary.load(PRESETS_ROOT)
    a = analyze(_buffer_from(make_calm_dynamic()))
    tags = set(a.summary_tags())
    assert not any(t.startswith("kb_") for t in tags), tags  # précondition du test

    result = select_preset_for_role(library, PluginRole.TAPE, a)
    assert result.preset is not None
    assert result.preset.name == "for all instr bus insert"


def test_tape_role_prefers_suno_tuned_preset_when_kb_band_elevated():
    library = PresetLibrary.load(PRESETS_ROOT)
    a = analyze(_buffer_from(make_bright()))
    tags = set(a.summary_tags())
    assert any(t.startswith("kb_") for t in tags), tags  # précondition du test

    result = select_preset_for_role(library, PluginRole.TAPE, a)
    assert result.preset is not None
    assert result.preset.name == "for suno"


def test_bass_di_preset_never_wins_on_full_mix_content():
    """Sur un contenu qui n'est PAS low_end_dominant (mix normal, même sombre),
    le preset bass DI ne doit jamais être choisi — voir
    config/presets/tape/j37_bass_di.meta.yaml."""
    library = PresetLibrary.load(PRESETS_ROOT)
    for maker in (make_bright, make_calm_dynamic):
        a = analyze(_buffer_from(maker()))
        assert "low_end_dominant" not in a.summary_tags()  # précondition du test
        result = select_preset_for_role(library, PluginRole.TAPE, a)
        assert result.preset is not None
        assert result.preset.name != "my bass"


def test_bass_di_preset_selected_for_low_end_dominant_content():
    library = PresetLibrary.load(PRESETS_ROOT)
    a = analyze(_buffer_from(_make_low_end_dominant()))
    assert "low_end_dominant" in a.summary_tags(), a.summary_tags()  # précondition du test

    result = select_preset_for_role(library, PluginRole.TAPE, a)
    assert result.preset is not None
    assert result.preset.name == "my bass"


if __name__ == "__main__":
    test_tape_role_has_four_presets_loaded()
    test_tape_role_defaults_to_general_bus_insert_without_kb_tags()
    test_tape_role_prefers_suno_tuned_preset_when_kb_band_elevated()
    test_bass_di_preset_never_wins_on_full_mix_content()
    test_bass_di_preset_selected_for_low_end_dominant_content()
    print("Tous les tests preset_mapping sont passés.")
