"""Tests du pilotage dynamique TAIP (taip_control.py) : lecture du blob
jucePluginState réel, construction d'un preset dynamique, décisions de
paramètres à partir de l'analyse."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.preset_types import load_aupreset
from refinr.taip_control import TaipTemplate, decide_params, make_dynamic_preset, parse_taip_params
from tests.generate_test_audio import SR, make_bright, make_calm_dynamic
from tests.test_preset_mapping import _make_low_end_dominant

TAIP_PRESET_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "presets" / "saturation" / "taip_suno_tuned.aupreset"
)


def _buffer_from(samples):
    return AudioBuffer(samples=samples, sample_rate=SR)


def test_parse_taip_params_reads_real_preset():
    preset = load_aupreset(TAIP_PRESET_PATH)
    params = parse_taip_params(preset)
    assert params["drive"] == 4.0
    assert params["glue"] == 15.0
    assert params["wear"] == 22.0
    assert len(params) == 15


def test_make_dynamic_preset_round_trips_through_parse():
    preset = load_aupreset(TAIP_PRESET_PATH)
    template = TaipTemplate.from_preset(preset)

    dynamic = make_dynamic_preset("test dynamic", {"drive": 3.0, "glue": 7.5, "wear": 1.0}, template)
    round_tripped = parse_taip_params(dynamic)

    assert round_tripped["drive"] == 3.0
    assert round_tripped["glue"] == 7.5
    assert round_tripped["wear"] == 1.0
    # Les champs non fournis retombent sur les valeurs par défaut, pas sur du bruit.
    assert round_tripped["mix"] == 100.0
    assert round_tripped["power"] == 1.0


def test_decide_params_never_adds_drive_on_clipped_source():
    a = analyze(_buffer_from(make_bright()))
    params = decide_params(a)
    if a.dynamics.clipping_ratio > 0.001:
        assert params["drive"] == 0.0


def test_decide_params_increases_glue_on_low_end_dominant_content():
    a = analyze(_buffer_from(_make_low_end_dominant()))
    assert "low_end_dominant" in a.summary_tags()  # précondition du test
    params = decide_params(a)
    assert params["glue"] > 0.0
    assert params["lo_shape"] > 0.0


def test_decide_params_stays_within_declared_bounds():
    for maker in (make_bright, make_calm_dynamic, _make_low_end_dominant):
        a = analyze(_buffer_from(maker()))
        params = decide_params(a)
        assert 0.0 <= params["drive"] <= 10.0
        assert 0.0 <= params["glue"] <= 20.0
        assert 0.0 <= params["wear"] <= 30.0
        assert 0.0 <= params["presence"] <= 20.0


if __name__ == "__main__":
    test_parse_taip_params_reads_real_preset()
    test_make_dynamic_preset_round_trips_through_parse()
    test_decide_params_never_adds_drive_on_clipped_source()
    test_decide_params_increases_glue_on_low_end_dominant_content()
    test_decide_params_stays_within_declared_bounds()
    print("Tous les tests taip_control sont passés.")
