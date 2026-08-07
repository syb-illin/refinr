"""
Orchestrateur de la chaîne de traitement complète, PAR FICHIER :

  1. Analyse du WAV source (analysis.py)
  2. Gain staging vers -18 LUFS (loudness.py) — pour ne pas exploser les
     plugins de retraitement avec les niveaux très chauds typiques de Suno
  3. Sélection des presets AU spécifiques à CE fichier (preset_mapping.py)
     pour EQ (Pro-Q4) -> Saturation (Saturn2/HG2) -> Tape (J37)
  4. Traitement réel via les AU (au_host.py, macOS uniquement)
  5. Leveling final vers le profil de destination choisi (loudness.py :
     gain vers target_lufs + limiteur true-peak de sécurité)
  6. Export + construction du rapport détaillé (ChainStepReport par étape)

Sur un OS non-macOS (dev/test), l'étape 4 est court-circuitée (le signal
gain-staged passe tel quel) pour que tout le reste du pipeline (analyse,
sélection, leveling, reporting, batch) reste testable de bout en bout.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

from . import loudness
from .analysis import analyze, FileAnalysis
from .audio_io import AudioBuffer, load_wav, save_wav
from .preset_mapping import PresetLibrary, select_chain
from .preset_types import ChainStepReport, PluginRole
from .profiles import DestinationProfile, ProfileCatalog

AU_HOSTING_AVAILABLE = sys.platform == "darwin"


@dataclasses.dataclass
class FileProcessingReport:
    input_path: str
    output_path: str
    destination_profile: str
    analysis_tags: list[str]
    steps: list[ChainStepReport]
    input_measurement: dict
    gain_staging_db: float
    post_gain_staging_measurement: dict
    final_measurement: dict
    warnings: list[str]
    duration_seconds: float
    au_hosting_used: bool


def _measurement_to_dict(m: loudness.LoudnessMeasurement) -> dict:
    return {
        "integrated_lufs": round(m.integrated_lufs, 3) if m.integrated_lufs == m.integrated_lufs else None,
        "true_peak_dbtp": round(m.true_peak_dbtp, 3),
        "loudness_range_lu": round(m.loudness_range_lu, 3) if m.loudness_range_lu is not None else None,
        "sample_peak_dbfs": round(m.sample_peak_dbfs, 3),
    }


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    library: PresetLibrary,
    profile: DestinationProfile,
    catalog: ProfileCatalog,
    output_subtype: str = "PCM_24",
) -> FileProcessingReport:
    t0 = time.monotonic()
    warnings: list[str] = []

    input_path = Path(input_path)
    output_path = Path(output_path)

    buffer = load_wav(input_path)
    input_measurement = loudness.measure(buffer)

    file_analysis: FileAnalysis = analyze(buffer)
    tags = file_analysis.summary_tags()

    gained_buffer, gain_db = loudness.gain_to_target_lufs(
        buffer,
        target_lufs=catalog.gain_staging.target_lufs,
        ceiling_dbtp=catalog.gain_staging.true_peak_ceiling_dbtp,
    )
    post_gain_measurement = loudness.measure(gained_buffer)

    selection = select_chain(library, file_analysis, roles=(PluginRole.EQ, PluginRole.SATURATION, PluginRole.TAPE))

    steps: list[ChainStepReport] = []
    processed_buffer = gained_buffer

    ordered_roles = [PluginRole.EQ, PluginRole.SATURATION, PluginRole.TAPE]
    chosen_presets = []
    for role in ordered_roles:
        result = selection[role]
        if result.preset is None:
            warnings.append(f"Rôle '{role.value}' ignoré: {result.reason}")
            continue
        chosen_presets.append((role, result))

    if AU_HOSTING_AVAILABLE and chosen_presets:
        from . import au_host  # import tardif : indisponible hors macOS

        pre_measure = loudness.measure(processed_buffer)
        tmp_staged_path = output_path.with_suffix(".staged_tmp.wav")
        save_wav(processed_buffer, tmp_staged_path, subtype="FLOAT")

        tmp_processed_path = output_path.with_suffix(".chain_tmp.wav")
        presets_in_order = [result.preset for _role, result in chosen_presets]
        au_host.process_chain_offline(str(tmp_staged_path), str(tmp_processed_path), presets_in_order)

        processed_buffer = load_wav(tmp_processed_path)
        post_measure = loudness.measure(processed_buffer)

        for role, result in chosen_presets:
            steps.append(
                ChainStepReport(
                    role=role.value,
                    plugin_name=result.preset.component_subtype,
                    preset_name=result.preset.name,
                    preset_path=str(result.preset.source_path),
                    reason=result.reason,
                    pre_measurement=_measurement_to_dict(pre_measure),
                    post_measurement=_measurement_to_dict(post_measure),
                )
            )

        tmp_staged_path.unlink(missing_ok=True)
        tmp_processed_path.unlink(missing_ok=True)
    else:
        if not AU_HOSTING_AVAILABLE:
            warnings.append(
                "Hosting AU indisponible sur cet OS (macOS requis) — la chaîne EQ/saturation/tape "
                "n'a PAS été appliquée, seul le gain staging + leveling final ont été effectués."
            )
        for role, result in chosen_presets:
            steps.append(
                ChainStepReport(
                    role=role.value,
                    plugin_name=result.preset.component_subtype,
                    preset_name=result.preset.name,
                    preset_path=str(result.preset.source_path),
                    reason=result.reason + " [SIMULÉ — AU non appliqué hors macOS]",
                    pre_measurement={},
                    post_measurement={},
                )
            )

    pre_level_measure = loudness.measure(processed_buffer)
    leveled_buffer, level_gain_db = loudness.gain_to_target_lufs(
        processed_buffer,
        target_lufs=profile.target_lufs,
        ceiling_dbtp=profile.true_peak_ceiling_dbtp,
    )
    limited_buffer = loudness.limit_true_peak(leveled_buffer, ceiling_dbtp=profile.true_peak_ceiling_dbtp)
    final_measurement = loudness.measure(limited_buffer)

    steps.append(
        ChainStepReport(
            role=PluginRole.LEVELING.value,
            plugin_name="refinr.loudness (gain + limiteur true-peak)",
            preset_name=profile.label,
            preset_path="",
            reason=(
                f"Leveling final vers profil '{profile.key}': cible {profile.target_lufs} LUFS / "
                f"{profile.true_peak_ceiling_dbtp} dBTP."
            ),
            pre_measurement=_measurement_to_dict(pre_level_measure),
            post_measurement=_measurement_to_dict(final_measurement),
            extra={"leveling_gain_db": round(level_gain_db, 3)},
        )
    )

    save_wav(limited_buffer, output_path, subtype=output_subtype)

    duration = time.monotonic() - t0
    return FileProcessingReport(
        input_path=str(input_path),
        output_path=str(output_path),
        destination_profile=profile.key,
        analysis_tags=tags,
        steps=steps,
        input_measurement=_measurement_to_dict(input_measurement),
        gain_staging_db=round(gain_db, 3),
        post_gain_staging_measurement=_measurement_to_dict(post_gain_measurement),
        final_measurement=_measurement_to_dict(final_measurement),
        warnings=warnings,
        duration_seconds=round(duration, 3),
        au_hosting_used=AU_HOSTING_AVAILABLE and bool(chosen_presets),
    )
