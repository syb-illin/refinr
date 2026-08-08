"""Tests du pilotage dynamique (partiel) du J37 (j37_control.py) : lecture
du tableau RealWorld réel, construction d'un preset dynamique par
substitution de spans, décisions de paramètres à partir de l'analyse."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.j37_control import (
    IDX_FLUTTER_DEPTH,
    IDX_FLUTTER_RATE,
    IDX_NOISE,
    IDX_SATURATION,
    IDX_SATURATION_MIRROR,
    IDX_WOW_DEPTH,
    IDX_WOW_RATE,
    NOISE_OFF_VALUE,
    NOISE_ON_VALUE,
    J37Params,
    J37Template,
    decide_params,
    make_dynamic_preset,
    parse_realworld_values,
)
from refinr.preset_types import load_aupreset
from tests.generate_test_audio import SR, make_bright, make_hot_clipped_source

J37_BASELINE_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "presets" / "tape" / "j37_baseline_reference.aupreset"
)


def _buffer_from(samples):
    return AudioBuffer(samples=samples, sample_rate=SR)


def test_parse_realworld_values_reads_real_preset():
    preset = load_aupreset(J37_BASELINE_PATH)
    values = parse_realworld_values(preset)
    assert len(values) == 195
    # Réglage par défaut du preset de référence (Formula 815 / Speed 15 /
    # Bias 0 / Modeled Tracks 2+3) : Saturation off. Le Noise Level brut du
    # template n'est PAS asserté ici (valeur incidente à l'export, non
    # contrôlée) — make_dynamic_preset l'écrase TOUJOURS explicitement, voir
    # test_make_dynamic_preset_round_trips_through_parse.
    assert values[IDX_SATURATION] == 0.0
    assert values[IDX_SATURATION_MIRROR] == 0.0


def test_make_dynamic_preset_round_trips_through_parse():
    preset = load_aupreset(J37_BASELINE_PATH)
    template = J37Template.from_preset(preset)

    params = J37Params(
        saturation=42.0, noise_on=True, wow_rate=12.0, wow_depth=-5.0, flutter_rate=3.0, flutter_depth=7.0
    )
    dynamic = make_dynamic_preset("test dynamic", params, template)
    values = parse_realworld_values(dynamic)

    assert values[IDX_SATURATION] == 42.0
    assert values[IDX_SATURATION_MIRROR] == 42.0
    assert values[IDX_NOISE] == NOISE_ON_VALUE
    assert values[IDX_WOW_RATE] == 12.0
    assert values[IDX_WOW_DEPTH] == -5.0
    assert values[IDX_FLUTTER_RATE] == 3.0
    assert values[IDX_FLUTTER_DEPTH] == 7.0
    # Le reste du tableau (indices non pilotés) doit rester identique au template.
    original_values = parse_realworld_values(preset)
    touched = {
        IDX_SATURATION,
        IDX_SATURATION_MIRROR,
        IDX_NOISE,
        IDX_WOW_RATE,
        IDX_WOW_DEPTH,
        IDX_FLUTTER_RATE,
        IDX_FLUTTER_DEPTH,
    }
    for i, (orig, new) in enumerate(zip(original_values, values, strict=True)):
        if i not in touched:
            assert orig == new, f"index {i} modifié alors qu'il ne devrait pas l'être ({orig} -> {new})"


def test_make_dynamic_preset_noise_off_by_default():
    preset = load_aupreset(J37_BASELINE_PATH)
    template = J37Template.from_preset(preset)
    dynamic = make_dynamic_preset("test noise off", J37Params(), template)
    values = parse_realworld_values(dynamic)
    assert values[IDX_NOISE] == NOISE_OFF_VALUE


def test_dynamic_preset_preserves_plist_envelope():
    """Le reste de l'enveloppe plist (name/type/subtype/manufacturer/element-name/data)
    doit rester exploitable par au_host — seule Waves_XPst et 'name' changent."""
    preset = load_aupreset(J37_BASELINE_PATH)
    template = J37Template.from_preset(preset)
    dynamic = make_dynamic_preset("mon nom de test", J37Params(saturation=10.0), template)

    assert dynamic.name == "mon nom de test"
    assert dynamic.full_state["name"] == "mon nom de test"
    # Les autres clés de l'enveloppe (élément-name, data, manufacturer, etc.)
    # doivent être copiées telles quelles depuis le template.
    for key in ("element-name", "data", "manufacturer", "subtype", "type", "version"):
        assert dynamic.full_state.get(key) == preset.full_state.get(key)


def test_decide_params_default_light_saturation():
    a = analyze(_buffer_from(make_bright()))
    params = decide_params(a)
    assert 0.0 < params.saturation <= 8.0
    assert params.wow_rate == 0.0
    assert params.wow_depth == 0.0
    assert params.flutter_rate == 0.0
    assert params.flutter_depth == 0.0


def test_decide_params_zero_saturation_on_clipped_source():
    a = analyze(_buffer_from(make_hot_clipped_source()))
    params = decide_params(a)
    if a.dynamics.clipping_ratio > 0.001:
        assert params.saturation == 0.0


if __name__ == "__main__":
    test_parse_realworld_values_reads_real_preset()
    test_make_dynamic_preset_round_trips_through_parse()
    test_make_dynamic_preset_noise_off_by_default()
    test_dynamic_preset_preserves_plist_envelope()
    test_decide_params_default_light_saturation()
    test_decide_params_zero_saturation_on_clipped_source()
    print("Tous les tests j37_control sont passés.")
