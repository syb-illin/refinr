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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.batch import run_batch
from refinr.chain import process_file
from refinr.preset_mapping import PresetLibrary
from refinr.profiles import load_profiles
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


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_process_file_end_to_end(Path(tmp) / "a")
    with tempfile.TemporaryDirectory() as tmp:
        test_batch_processes_all_files_in_parallel(Path(tmp))
    print("Tous les tests chain/batch sont passés.")
