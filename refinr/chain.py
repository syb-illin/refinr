"""
Orchestrateur de la chaîne de traitement complète, PAR FICHIER :

  1. Analyse du WAV source (analysis.py)
  2. Gain staging vers -18 LUFS (loudness.py) — pour ne pas exploser les
     plugins de retraitement avec des sources qui sortent à des niveaux très chauds
  3. EQ Pro-Q4 : PILOTAGE AUTOMATIQUE (proq4_control.decide_bands), pas de
     sélection parmi des presets figés — les réglages (fréquence/gain/Q/
     forme/pente/stéréo/dynamique) sont décidés directement à partir de
     l'analyse de CE fichier. Saturation (Saturn2/HG2) et Tape (J37) restent
     sur sélection de presets par tags (preset_mapping.py) — ces plugins
     n'ont pas encore été reverse-engineered pour un pilotage fin.
  4. Traitement réel via les AU (au_host.py, macOS uniquement)
  5. Leveling final vers le profil de destination choisi (loudness.py :
     gain vers target_lufs + limiteur true-peak de sécurité)
  6. Export + construction du rapport détaillé (ChainStepReport par étape)

Sur un OS non-macOS (dev/test), l'étape 4 est court-circuitée (le signal
gain-staged passe tel quel) pour que tout le reste du pipeline (analyse,
décision EQ, sélection saturation/tape, leveling, reporting, batch) reste
testable de bout en bout.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

from . import loudness
from .analysis import FileAnalysis, analyze
from .audio_io import load_wav, save_wav
from .preset_mapping import PresetLibrary, select_chain
from .preset_types import ChainStepReport, PluginRole
from .profiles import DestinationProfile, ProfileCatalog
from .proq4_control import Band, decide_bands, make_dynamic_preset

AU_HOSTING_AVAILABLE = sys.platform == "darwin"


def _describe_band(b: Band) -> str:
    parts = [f"{b.shape} {b.freq_hz:.0f}Hz"]
    if b.gain_db:
        parts.append(f"{b.gain_db:+.1f}dB")
    if b.q != 1.0:
        parts.append(f"Q{b.q:.2f}")
    if b.slope_db_per_oct:
        parts.append(f"{b.slope_db_per_oct:.0f}dB/oct")
    if b.stereo != "stereo":
        parts.append(b.stereo)
    if b.dynamic_range_db:
        auto = " auto" if b.dynamic_auto_threshold else ""
        parts.append(f"dyn{auto} {b.dynamic_range_db:+.1f}dB")
    return " ".join(parts)


def _build_diagnostic(analysis: FileAnalysis) -> dict:
    """Diagnostic COMPLET de l'analyse, pas juste les tags résumés — toutes
    les mesures brutes qui ont servi (ou pas) à décider des corrections."""
    return {
        "tags": analysis.summary_tags(),
        "spectral": {
            "band_energy_db": {k: round(v, 2) for k, v in analysis.spectral.band_energy_db.items()},
            "spectral_centroid_hz": round(analysis.spectral.spectral_centroid_hz, 1),
            "tilt_db_per_octave": round(analysis.spectral.tilt_db_per_octave, 3),
        },
        "dynamics": {
            "crest_factor_db": round(analysis.dynamics.crest_factor_db, 2),
            "clipping_ratio_pct": round(analysis.dynamics.clipping_ratio * 100, 3),
            "loudness_range_lu": (
                round(analysis.dynamics.loudness_range_lu, 2)
                if analysis.dynamics.loudness_range_lu is not None
                else None
            ),
            "stereo_correlation": round(analysis.dynamics.stereo_correlation, 3),
        },
        "loudness": {
            "integrated_lufs": (
                round(analysis.loudness.integrated_lufs, 2)
                if analysis.loudness.integrated_lufs == analysis.loudness.integrated_lufs
                else None
            ),
            "sample_peak_dbfs": round(analysis.loudness.sample_peak_dbfs, 2),
            "true_peak_dbtp": round(analysis.loudness.true_peak_dbtp, 2),
        },
    }


@dataclasses.dataclass
class FileProcessingReport:
    input_path: str
    output_path: str
    destination_profile: str
    analysis_tags: list[str]
    diagnostic: dict
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
    diagnostic = _build_diagnostic(file_analysis)

    gained_buffer, gain_db = loudness.gain_to_target_lufs(
        buffer,
        target_lufs=catalog.gain_staging.target_lufs,
        ceiling_dbtp=catalog.gain_staging.true_peak_ceiling_dbtp,
    )
    post_gain_measurement = loudness.measure(gained_buffer)

    # EQ Pro-Q4 : TOUJOURS piloté dynamiquement (proq4_control), jamais de
    # sélection de preset. Saturation/Tape : sélection par tags, faute
    # d'avoir reverse-engineered Saturn2/HG2/J37 pour l'instant.
    eq_bands = decide_bands(file_analysis)
    if eq_bands:
        eq_reason = (
            f"{len(eq_bands)} bande(s) EQ décidée(s) à partir du diagnostic ci-dessus (voir raison "
            "détaillée par bande)."
        )
    else:
        eq_reason = "Aucune correction EQ jugée nécessaire — diagnostic dans les limites normales."
    eq_preset = make_dynamic_preset(f"refinr auto EQ — {input_path.name}", eq_bands)

    selection = select_chain(library, file_analysis, roles=(PluginRole.SATURATION, PluginRole.TAPE))

    steps: list[ChainStepReport] = []
    processed_buffer = gained_buffer

    chosen_extra_presets = []  # saturation/tape uniquement, EQ traité à part
    for role in (PluginRole.SATURATION, PluginRole.TAPE):
        result = selection[role]
        if result.preset is None:
            warnings.append(f"Rôle '{role.value}' ignoré: {result.reason}")
            continue
        chosen_extra_presets.append((role, result))

    if AU_HOSTING_AVAILABLE:
        from . import au_host  # import tardif : indisponible hors macOS

        pre_measure = loudness.measure(processed_buffer)
        presets_in_order = [eq_preset] + [result.preset for _role, result in chosen_extra_presets]
        processed_buffer, _render_result = au_host.process_chain_offline(processed_buffer, presets_in_order)
        post_measure = loudness.measure(processed_buffer)

        steps.append(
            ChainStepReport(
                role=PluginRole.EQ.value,
                plugin_name="FabFilter Pro-Q4 (pilotage dynamique)",
                preset_name="auto",
                preset_path="",
                reason=eq_reason,
                pre_measurement=_measurement_to_dict(pre_measure),
                post_measurement=_measurement_to_dict(post_measure),
                extra={"bands": [{"summary": _describe_band(b), "reason": b.reason} for b in eq_bands]},
            )
        )
        for role, result in chosen_extra_presets:
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
    else:
        warnings.append(
            "Hosting AU indisponible sur cet OS (macOS requis) — la chaîne EQ/saturation/tape "
            "n'a PAS été appliquée, seul le gain staging + leveling final ont été effectués."
        )
        steps.append(
            ChainStepReport(
                role=PluginRole.EQ.value,
                plugin_name="FabFilter Pro-Q4 (pilotage dynamique)",
                preset_name="auto",
                preset_path="",
                reason=eq_reason + " [SIMULÉ — AU non appliqué hors macOS]",
                pre_measurement={},
                post_measurement={},
                extra={"bands": [{"summary": _describe_band(b), "reason": b.reason} for b in eq_bands]},
            )
        )
        for role, result in chosen_extra_presets:
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
        diagnostic=diagnostic,
        steps=steps,
        input_measurement=_measurement_to_dict(input_measurement),
        gain_staging_db=round(gain_db, 3),
        post_gain_staging_measurement=_measurement_to_dict(post_gain_measurement),
        final_measurement=_measurement_to_dict(final_measurement),
        warnings=warnings,
        duration_seconds=round(duration, 3),
        au_hosting_used=AU_HOSTING_AVAILABLE,
    )
