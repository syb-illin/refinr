"""Tests du pilotage dynamique Pro-Q4 : décision de bandes + export .aupreset.

Tournent sur n'importe quel OS — proq4_control.py ne fait que construire
un dict Python + des bytes (pas de hosting AU réel, voir au_host.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.preset_types import load_aupreset, write_aupreset
from refinr.proq4_control import decide_bands, make_dynamic_preset
from tests.generate_test_audio import (
    SR,
    make_bright,
    make_calm_dynamic,
    make_hot_clipped_source,
    make_suno_artifact_like,
)


def _analysis_from(samples):
    return analyze(AudioBuffer(samples=samples, sample_rate=SR))


def test_suno_mode_adds_two_extra_bands_when_both_kb_bands_elevated():
    """make_suno_artifact_like() est calibré pour dépasser le seuil KB dans
    les DEUX bandes (metallic_4k et hf_fizz_14k) — voir generate_test_audio.py."""
    a = _analysis_from(make_suno_artifact_like())
    bands_normal = decide_bands(a, suno_mode=False)
    bands_suno = decide_bands(a, suno_mode=True)
    assert len(bands_suno) == len(bands_normal) + 2

    extra_freqs = {b.freq_hz for b in bands_suno} - {b.freq_hz for b in bands_normal}
    assert extra_freqs == {14000.0, 4000.0}


def test_suno_mode_bands_have_cited_reason():
    a = _analysis_from(make_suno_artifact_like())
    bands_suno = decide_bands(a, suno_mode=True)
    suno_bands = [b for b in bands_suno if b.freq_hz in (14000.0, 4000.0)]
    assert len(suno_bands) == 2
    for band in suno_bands:
        assert "suno_artifacts_kb.md" in band.reason


def test_suno_mode_gated_by_measurement_not_blanket():
    """Coeur du fix : suno_mode=True n'ajoute PLUS les deux bandes à
    l'aveugle. make_bright() n'a qu'une seule bande KB élevée (metallic_4k,
    via son fort contenu à 4000Hz) — seule celle-là doit être ajoutée,
    hf_fizz_14k n'étant pas élevée sur ce signal (pas de contenu au-delà de
    12kHz)."""
    a = _analysis_from(make_bright())
    bands_normal = decide_bands(a, suno_mode=False)
    bands_suno = decide_bands(a, suno_mode=True)
    extra_freqs = {b.freq_hz for b in bands_suno} - {b.freq_hz for b in bands_normal}
    assert extra_freqs == {4000.0}, "seule la bande réellement mesurée comme élevée doit être ajoutée"


def test_suno_mode_adds_nothing_when_no_kb_band_elevated():
    """make_calm_dynamic() n'a d'énergie dans aucune des bandes KB — même
    avec suno_mode=True, aucune correction a priori ne doit être ajoutée."""
    a = _analysis_from(make_calm_dynamic())
    bands_normal = decide_bands(a, suno_mode=False)
    bands_suno = decide_bands(a, suno_mode=True)
    assert bands_suno == bands_normal
    assert 14000.0 not in {b.freq_hz for b in bands_suno}
    assert 4000.0 not in {b.freq_hz for b in bands_suno}


def test_suno_mode_never_added_by_default():
    a = _analysis_from(make_hot_clipped_source())
    bands = decide_bands(a)  # pas de suno_mode explicite
    assert 14000.0 not in {b.freq_hz for b in bands}
    assert 4000.0 not in {b.freq_hz for b in bands}


def test_write_aupreset_round_trip(tmp_path):
    """Le preset dynamique écrit en .aupreset doit se relire à l'identique
    via load_aupreset (même mécanisme que pour un vrai preset FabFilter)."""
    a = _analysis_from(make_hot_clipped_source())
    bands = decide_bands(a)
    preset = make_dynamic_preset("test export", bands)

    out_path = tmp_path / "test_export.aupreset"
    written_path = write_aupreset(preset, out_path)
    assert written_path == out_path
    assert out_path.exists()

    reloaded = load_aupreset(out_path)
    assert reloaded.name == preset.name
    assert reloaded.component_subtype == preset.component_subtype
    assert reloaded.component_manufacturer == preset.component_manufacturer
    assert reloaded.full_state["FabFilterPluginState"] == preset.full_state["FabFilterPluginState"]


def test_write_aupreset_creates_parent_dirs(tmp_path):
    a = _analysis_from(make_bright())
    preset = make_dynamic_preset("test nested", decide_bands(a))
    nested_path = tmp_path / "nested" / "dir" / "preset.aupreset"

    write_aupreset(preset, nested_path)
    assert nested_path.exists()


if __name__ == "__main__":
    import tempfile

    test_suno_mode_adds_two_extra_bands_when_both_kb_bands_elevated()
    test_suno_mode_bands_have_cited_reason()
    test_suno_mode_gated_by_measurement_not_blanket()
    test_suno_mode_adds_nothing_when_no_kb_band_elevated()
    test_suno_mode_never_added_by_default()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_aupreset_round_trip(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_write_aupreset_creates_parent_dirs(Path(tmp))
    print("Tous les tests proq4_control sont passés.")
