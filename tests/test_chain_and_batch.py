"""
Tests d'intégration bout-en-bout du pipeline (hors hosting AU réel, qui
nécessite macOS — voir tools/au_host_smoketest.py pour cette partie-là).

Vérifie : analyse -> gain staging -> sélection presets -> leveling profil
-> export -> batch parallèle -> rapport JSON/HTML, sans planter et avec des
mesures cohérentes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.batch import run_batch
from refinr.chain import OutputValidationError, _validate_output, process_file
from refinr.preset_mapping import PresetLibrary
from refinr.profiles import DestinationProfile, load_profiles
from refinr.report import write_reports

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"
PROFILES_PATH = Path(__file__).resolve().parent.parent / "config" / "destination_profiles.yaml"


def test_process_file_end_to_end(tmp_path):
    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")

    out_path = tmp_path / "out.wav"
    report = process_file(FIXTURES / "hot_clipped_source.wav", out_path, library, profile, catalog)

    assert out_path.exists()
    assert report.final_measurement["true_peak_dbtp"] <= profile.true_peak_ceiling_dbtp + 0.05
    assert abs(report.final_measurement["integrated_lufs"] - profile.target_lufs) < 1.0
    assert report.gain_staging_db != 0.0  # source chaude -> gain staging doit bouger le niveau


def test_process_file_resamples_output_to_profile_delivery_rate(tmp_path):
    """AVANT profiles.DestinationProfile.output_sample_rate : la sortie
    gardait le sample rate de la SOURCE quel qu'il soit. Un WAV source à
    48000Hz doit maintenant ressortir à 44100Hz pour un profil qui vise
    44100Hz (voir chain.process_file, resample_if_needed)."""
    import soundfile as sf

    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")
    assert profile.output_sample_rate == 44100

    sr_source = 48000
    t = np.linspace(0, 2.0, sr_source * 2, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    source_path = tmp_path / "source_48k.wav"
    sf.write(str(source_path), np.stack([mono, mono], axis=1), sr_source, subtype="PCM_24")

    out_path = tmp_path / "out.wav"
    process_file(source_path, out_path, library, profile, catalog)

    info = sf.info(str(out_path))
    assert info.samplerate == 44100


def test_process_file_exports_aupreset_when_requested(tmp_path):
    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")
    export_dir = tmp_path / "presets_aupreset"

    out_path = tmp_path / "out.wav"
    report = process_file(
        FIXTURES / "hot_clipped_source.wav",
        out_path,
        library,
        profile,
        catalog,
        export_eq_preset_dir=export_dir,
    )

    assert len(report.exported_presets) == 1
    exported_path = Path(report.exported_presets[0])
    assert exported_path.exists()
    assert exported_path.parent == export_dir


def test_process_file_suno_mode_flag_propagates_to_report(tmp_path):
    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")

    out_path = tmp_path / "out.wav"
    report = process_file(
        FIXTURES / "bright.wav",
        out_path,
        library,
        profile,
        catalog,
        suno_mode=True,
    )
    assert report.suno_mode is True


def test_process_file_populates_qc_and_comparison_table(tmp_path):
    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")

    out_path = tmp_path / "out.wav"
    report = process_file(FIXTURES / "hot_clipped_source.wav", out_path, library, profile, catalog)

    assert report.qc_passed is True
    assert report.output_diagnostic  # non vide
    assert len(report.comparison_table) > 10  # tableau raw/refined complet
    assert all({"metric", "raw", "refined", "delta"} <= set(row) for row in report.comparison_table)
    # true peak et LUFS doivent apparaître dans le tableau
    metrics = {row["metric"] for row in report.comparison_table}
    assert "loudness.true_peak_dbtp" in metrics
    assert "loudness.integrated_lufs" in metrics


def _clean_analysis_and_duration():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    buffer = AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=sr)
    return analyze(buffer), buffer.duration_seconds


def _test_profile(true_peak_ceiling_dbtp: float = -1.0, target_lufs: float = -14.0) -> DestinationProfile:
    return DestinationProfile(
        key="test",
        label="Test",
        target_lufs=target_lufs,
        true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
        boosts_quiet=None,
    )


def test_validate_output_fails_on_true_peak_violation():
    analysis, dur = _clean_analysis_and_duration()
    profile = _test_profile(true_peak_ceiling_dbtp=-1.0)
    final_measurement = {
        "integrated_lufs": -14.0,
        "true_peak_dbtp": 0.5,
        "loudness_range_lu": None,
        "sample_peak_dbfs": 0.4,
    }

    errors, _warns = _validate_output(analysis, final_measurement, None, profile, dur, dur)
    assert any("True peak" in e for e in errors)


def test_validate_output_fails_on_lufs_drift():
    analysis, dur = _clean_analysis_and_duration()
    profile = _test_profile(target_lufs=-14.0)
    final_measurement = {
        "integrated_lufs": -20.0,
        "true_peak_dbtp": -3.0,
        "loudness_range_lu": None,
        "sample_peak_dbfs": -3.0,
    }

    errors, _warns = _validate_output(analysis, final_measurement, None, profile, dur, dur)
    assert any("Loudness" in e for e in errors)


def test_validate_output_fails_on_truncated_duration():
    analysis, dur = _clean_analysis_and_duration()
    profile = _test_profile()
    final_measurement = {
        "integrated_lufs": -14.0,
        "true_peak_dbtp": -3.0,
        "loudness_range_lu": None,
        "sample_peak_dbfs": -3.0,
    }

    errors, _warns = _validate_output(analysis, final_measurement, None, profile, dur, dur - 1.0)
    assert any("courte" in e for e in errors)


def test_validate_output_passes_on_compliant_output():
    analysis, dur = _clean_analysis_and_duration()
    profile = _test_profile()
    final_measurement = {
        "integrated_lufs": -14.0,
        "true_peak_dbtp": -2.0,
        "loudness_range_lu": None,
        "sample_peak_dbfs": -3.0,
    }

    errors, _warns = _validate_output(analysis, final_measurement, None, profile, dur, dur)
    assert errors == []


def test_output_validation_error_deletes_bad_file_and_raises(tmp_path, monkeypatch):
    """Preuve de bout en bout que le gate, quand il échoue, supprime le
    fichier de sortie ET remonte une erreur explicite — pas de fichier
    non conforme laissé silencieusement sur disque."""
    import refinr.chain as chain_module

    def _always_fails(**_kwargs):
        return ["erreur forcée pour le test"], []

    monkeypatch.setattr(chain_module, "_validate_output", _always_fails)

    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")
    out_path = tmp_path / "out.wav"

    with pytest.raises(OutputValidationError):
        process_file(FIXTURES / "hot_clipped_source.wav", out_path, library, profile, catalog)

    assert not out_path.exists()


def test_process_file_new_stages_appear_in_report_steps(tmp_path):
    """Transient shaping et contrôle de largeur stéréo (nouveaux stages, pas
    de plugin AU) doivent apparaître dans report.steps comme n'importe quel
    autre stage de la chaîne."""
    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")

    out_path = tmp_path / "out.wav"
    report = process_file(FIXTURES / "hot_clipped_source.wav", out_path, library, profile, catalog)

    roles = [s.role for s in report.steps]
    assert "transient_shaping" in roles
    assert "stereo_width" in roles
    assert "macro_dynamics" in roles
    assert report.qc_correction_passes == 0  # cas nominal, pas d'échec à corriger


def test_process_file_iterative_qc_correction_recovers_from_forced_lufs_drift(tmp_path, monkeypatch):
    """Preuve de bout en bout que la boucle de repasses correctives (voir
    chain.MAX_QC_CORRECTION_PASSES) fonctionne réellement : on force un biais
    de +2.5 LU UNIQUEMENT sur la première tentative de leveling (le gate QC
    doit donc échouer une fois), et on vérifie que le fichier finit par être
    produit avec une loudness conforme, sans jamais avoir eu besoin de le
    supprimer/relancer manuellement."""
    import refinr.chain as chain_module

    catalog = load_profiles(PROFILES_PATH)
    library = PresetLibrary.load(PRESETS_ROOT)
    profile = catalog.get("spotify")
    assert profile.target_lufs == -14.0  # précondition du test, sinon le biais ci-dessous ne cible pas la bonne étape

    real_gain_to_target = chain_module.loudness.gain_to_target_lufs
    state = {"biased": False}

    def _biased_gain(buffer, target_lufs, ceiling_dbtp):
        if not state["biased"] and abs(target_lufs - profile.target_lufs) < 0.01:
            state["biased"] = True
            return real_gain_to_target(buffer, target_lufs=target_lufs + 2.5, ceiling_dbtp=ceiling_dbtp)
        return real_gain_to_target(buffer, target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp)

    monkeypatch.setattr(chain_module.loudness, "gain_to_target_lufs", _biased_gain)

    out_path = tmp_path / "out.wav"
    report = process_file(FIXTURES / "hot_clipped_source.wav", out_path, library, profile, catalog)

    assert state["biased"] is True  # le biais a bien été déclenché une fois
    assert report.qc_correction_passes >= 1  # au moins une repasse corrective a été nécessaire
    assert out_path.exists()
    assert abs(report.final_measurement["integrated_lufs"] - profile.target_lufs) < 1.0
    assert any("Repasse corrective QC" in w for w in report.warnings)


def test_batch_processes_all_files_in_parallel(tmp_path):
    files = sorted(str(p) for p in FIXTURES.glob("*.wav") if "smoketest" not in p.name)
    result = run_batch(
        input_paths=files,
        output_dir=tmp_path,
        presets_root=PRESETS_ROOT,
        profile_key="youtube",
        profiles_path=PROFILES_PATH,
        max_workers=2,
    )
    assert len(result.outcomes) == len(files)
    assert len(result.failed) == 0, [(o.input_path, o.error) for o in result.failed]

    report_paths = write_reports(result, tmp_path, "youtube")
    assert Path(report_paths["html_summary"]).exists()
    assert len(report_paths["per_file_json"]) == len(files)


def test_batch_custom_profile_output_sample_rate_and_bit_depth(tmp_path):
    """Sliders kHz/bit-depth de la GUI (Cibles personnalisées) : le fichier
    de sortie doit vraiment être livré au sample rate/bit depth choisis,
    pas seulement au LUFS/true peak."""
    import soundfile as sf

    custom = {
        "target_lufs": -14.0,
        "true_peak_ceiling_dbtp": -1.0,
        "output_sample_rate": 48000,
        "output_bit_depth": "PCM_16",
    }
    result = run_batch(
        input_paths=[str(FIXTURES / "hot_clipped_source.wav")],
        output_dir=tmp_path,
        presets_root=PRESETS_ROOT,
        profile_key="spotify",
        profiles_path=PROFILES_PATH,
        max_workers=1,
        custom_profile=custom,
    )
    assert len(result.failed) == 0, [(o.input_path, o.error) for o in result.failed]
    info = sf.info(result.succeeded[0].report.output_path)
    assert info.samplerate == 48000
    assert info.subtype == "PCM_16"


def test_batch_custom_profile_overrides_yaml_targets(tmp_path):
    """Sliders 'Cibles personnalisées' de la GUI : run_batch(custom_profile=...)
    doit ignorer les cibles du profil YAML résolu par profile_key et viser
    à la place les valeurs fournies manuellement."""
    custom = {"target_lufs": -9.0, "true_peak_ceiling_dbtp": -2.0}
    result = run_batch(
        input_paths=[str(FIXTURES / "hot_clipped_source.wav")],
        output_dir=tmp_path,
        presets_root=PRESETS_ROOT,
        profile_key="spotify",  # -14 LUFS / -1dBTP -> doit être ignoré au profit de `custom`
        profiles_path=PROFILES_PATH,
        max_workers=1,
        custom_profile=custom,
    )
    assert len(result.failed) == 0, [(o.input_path, o.error) for o in result.failed]
    report = result.succeeded[0].report
    assert abs(report.final_measurement["integrated_lufs"] - custom["target_lufs"]) < 1.0
    assert report.final_measurement["true_peak_dbtp"] <= custom["true_peak_ceiling_dbtp"] + 0.05


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_process_file_end_to_end(Path(tmp) / "a")
    with tempfile.TemporaryDirectory() as tmp:
        test_batch_processes_all_files_in_parallel(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_batch_custom_profile_overrides_yaml_targets(Path(tmp))
    print("Tous les tests chain/batch sont passés.")
