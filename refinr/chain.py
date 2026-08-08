"""
Orchestrateur de la chaîne de traitement complète, PAR FICHIER :

  1. Analyse du WAV source (analysis.py)
  2. Gain staging vers -18 LUFS (loudness.py) — pour ne pas exploser les
     plugins de retraitement avec des sources qui sortent à des niveaux très chauds
  3. EQ Pro-Q4 : PILOTAGE AUTOMATIQUE (proq4_control.decide_bands), pas de
     sélection parmi des presets figés — les réglages (fréquence/gain/Q/
     forme/pente/stéréo/dynamique) sont décidés directement à partir de
     l'analyse de CE fichier. Saturation : PILOTAGE AUTOMATIQUE également
     (taip_control.decide_params, Baby Audio TAIP) dès qu'un preset TAIP de
     référence est présent dans la bibliothèque (voir `_find_taip_template`)
     — sinon repli sur la sélection de presets par tags (preset_mapping.py).
     Tape (J37) : PILOTAGE AUTOMATIQUE PARTIEL (j37_control.decide_params) —
     Saturation/Noise/Wow/Flutter sont pilotés dynamiquement dès que
     `config/presets/tape/j37_baseline_reference.aupreset` est présent (voir
     `_find_j37_template`), mais Speed/Bias/Formula/Modeled Tracks restent
     FIXES sur ce preset de référence : reverse-engineering confirmé que ces
     quatre-là déclenchent chacun un recalcul physique corrélé sur 30 à 69
     indices à la fois (pas de mapping scalaire possible), voir la docstring
     de `j37_control.py` pour le détail. Sinon (aucun preset de référence
     trouvé), repli sur la sélection de presets par tags, comme avant.
  4. Traitement réel via les AU (au_host.py, macOS uniquement)
  5. Transient shaping (transient_shaping.py) : corrige les attaques molles/
     lissées sur du matériel déjà compressé — décidé automatiquement à
     partir de l'analyse, pas de plugin AU, DSP maison.
  6. Contrôle actif de largeur stéréo (stereo_width.py) : resserre ou
     élargit le canal Side (mono-safe <150Hz) selon la corrélation stéréo
     mesurée — décidé automatiquement, pas de plugin AU, DSP maison.
  7. Correction macro-dynamique (macro_dynamics.py) : rider de gain à
     constante de temps lente (secondes), piloté par la Loudness Range
     (LRA) mesurée — réduit les sauts de volume gênants entre sections
     (couplet/refrain) sans aplatir la dynamique musicale, se déclenche
     seulement si la LRA dépasse un seuil. Pas de plugin AU, DSP maison.
  8. Leveling final vers le profil de destination choisi (loudness.py :
     gain vers target_lufs + limiteur true-peak de sécurité), avec repasses
     CORRECTIVES automatiques (jusqu'à MAX_QC_CORRECTION_PASSES) si le gate
     de validation échoue pour une raison corrigible par le gain (dérive
     LUFS, dépassement true peak) — voir la boucle dans `process_file`.
  9. Export + construction du rapport détaillé (ChainStepReport par étape)

Sur un OS non-macOS (dev/test), l'étape 4 est court-circuitée (le signal
gain-staged passe tel quel) pour que tout le reste du pipeline (analyse,
décision EQ, sélection saturation/tape, leveling, reporting, batch) reste
testable de bout en bout.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import loudness
from .analysis import FileAnalysis, analyze
from .audio_io import load_wav, resample_if_needed, save_wav
from .j37_control import J37Template
from .j37_control import decide_params as decide_j37_params
from .j37_control import make_dynamic_preset as make_dynamic_j37_preset
from .macro_dynamics import apply_macro_compression, decide_macro_compression
from .preset_mapping import PresetLibrary, SelectionResult, select_chain
from .preset_types import ChainStepReport, PluginRole, write_aupreset
from .profiles import DestinationProfile, ProfileCatalog
from .proq4_control import Band, decide_bands, make_dynamic_preset
from .reference_loudness import ReferenceMeasurement, measure_reference
from .stereo_width import apply_stereo_width, decide_width_factor
from .taip_control import TaipTemplate
from .taip_control import decide_params as decide_taip_params
from .taip_control import make_dynamic_preset as make_dynamic_taip_preset
from .transient_shaping import apply_transient_shaping, decide_attack_amount_db

# Tolérances du gate de validation post-traitement (voir _validate_output).
# Choisies volontairement plus larges que la précision réellement observée
# (les tests d'intégration visent <1.0 LU) pour ne détecter que de vrais
# bugs, pas du bruit de mesure — mais un dépassement du plafond true peak
# est TOUJOURS bloquant, sans tolérance au-delà de l'epsilon flottant.
QC_TRUE_PEAK_EPSILON_DB = 0.05
QC_LUFS_WARN_LU = 0.5
QC_LUFS_FAIL_LU = 1.5
QC_CROSS_VALIDATION_TOLERANCE_DB = 0.3
QC_DURATION_TRUNCATION_TOLERANCE_SEC = 0.1

# Nombre de repasses CORRECTIVES supplémentaires (donc MAX_QC_CORRECTION_PASSES+1
# tentatives au total) quand le gate de validation échoue pour une raison
# corrigible par le gain (dérive LUFS, dépassement true peak) — voir la
# boucle dans process_file. Les échecs non corrigibles par le gain (NaN/Inf,
# troncature, écrêtage dur...) ne consomment PAS de repasse : ils sortent
# immédiatement de la boucle, comme avant l'ajout de cette fonctionnalité.
MAX_QC_CORRECTION_PASSES = 2


class OutputValidationError(RuntimeError):
    """
    Levée quand le WAV final ne respecte pas les critères minimaux de
    conformité avant diffusion (true peak, LUFS, intégrité du signal —
    voir `_validate_output`). Le fichier de sortie est supprimé AVANT que
    l'exception ne remonte : mieux vaut ne produire aucun fichier qu'un
    fichier non conforme silencieusement laissé dans le dossier de sortie.
    """


# REFINR_TEST_DISABLE_AU_HOSTING=1 court-circuite le hosting AU réel même
# sur macOS. Nécessaire (pas juste pratique) pour les tests : batch.py
# parallélise via ProcessPoolExecutor, et un monkeypatch pytest ne
# traverse PAS la frontière process (chaque worker réimporte au_host à
# neuf) — une variable d'environnement, elle, est héritée par les process
# enfants. Positionnée par tests/conftest.py, jamais en usage normal.
AU_HOSTING_AVAILABLE = sys.platform == "darwin" and os.environ.get("REFINR_TEST_DISABLE_AU_HOSTING") != "1"


def _find_taip_template(library: PresetLibrary) -> TaipTemplate | None:
    """
    Cherche un preset TAIP réel dans la bibliothèque SATURATION pour servir
    de template (enveloppe plist + attributs `<TAIP_1 ...>`, voir
    `taip_control.TaipTemplate`) au pilotage dynamique — le format TAIP
    n'a pas besoin d'être reverse-engineré au-delà de ses `<PARAM>` (déjà
    fait, voir taip_control.py), seule l'enveloppe doit être copiée depuis
    un vrai fichier.

    Retourne None si aucun preset TAIP n'est présent (bibliothèque perso
    sans TAIP, ou role SATURATION vide) : `process_file` retombe alors sur
    la sélection statique par tags (preset_mapping), exactement comme avant
    l'introduction du pilotage dynamique de la saturation.
    """
    for entry in library.entries_by_role.get(PluginRole.SATURATION, []):
        blob = entry.preset.full_state.get("jucePluginState")
        if isinstance(blob, (bytes, bytearray)) and bytes(blob).startswith(b"VC2!"):
            try:
                return TaipTemplate.from_preset(entry.preset)
            except ValueError:
                continue
    return None


# Nom de fichier EXACT (sans extension) requis pour servir de template J37 —
# contrairement à TAIP (une seule instance réelle disponible, donc aucune
# ambiguïté), la bibliothèque TAPE contient PLUSIEURS vrais presets J37
# (bass_di, bus_insert_general, suno_artifact_tuned, en plus de la
# référence). Tous parseraient structurellement comme un J37 valide, mais
# chacun fige Speed/Bias/Formula/Modeled Tracks à des valeurs différentes et
# pas forcément neutres (voir j37_control.py : ces quatre-là ne sont PAS
# pilotables dynamiquement). Une correspondance par nom exact évite de piocher
# au hasard un preset accordé pour un usage créatif précis comme base du
# pilotage dynamique corrective.
J37_TEMPLATE_STEM = "j37_baseline_reference"


def _find_j37_template(library: PresetLibrary) -> J37Template | None:
    """
    Cherche PRÉCISÉMENT `config/presets/tape/j37_baseline_reference.aupreset`
    (voir J37_TEMPLATE_STEM) dans la bibliothèque TAPE pour servir de
    template au pilotage dynamique partiel (voir `j37_control.py` : seuls
    Saturation/Noise/Wow/Flutter sont pilotés, le reste vient de ce preset).

    Retourne None si absent (bibliothèque perso sans ce fichier précis) :
    `process_file` retombe alors sur la sélection statique par tags
    (preset_mapping), exactement comme avant l'introduction du pilotage
    dynamique du J37.
    """
    for entry in library.entries_by_role.get(PluginRole.TAPE, []):
        if entry.preset.source_path.stem != J37_TEMPLATE_STEM:
            continue
        try:
            return J37Template.from_preset(entry.preset)
        except ValueError:
            continue
    return None


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
        "ai_score": analysis.ai_score,
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


def _flatten_metrics(a: FileAnalysis) -> dict[str, float | int | None]:
    """Aplati TOUTES les métriques d'une FileAnalysis en un dict plat
    metric_path -> valeur, pour construire le tableau comparatif brut/raffiné
    (voir `_build_comparison_table`)."""
    flat: dict[str, float | int | None] = {
        "ai_score": a.ai_score,
        "loudness.integrated_lufs": (
            a.loudness.integrated_lufs if a.loudness.integrated_lufs == a.loudness.integrated_lufs else None
        ),
        "loudness.true_peak_dbtp": a.loudness.true_peak_dbtp,
        "loudness.loudness_range_lu": a.loudness.loudness_range_lu,
        "loudness.sample_peak_dbfs": a.loudness.sample_peak_dbfs,
        "spectral.spectral_centroid_hz": a.spectral.spectral_centroid_hz,
        "spectral.tilt_db_per_octave": a.spectral.tilt_db_per_octave,
        "dynamics.crest_factor_db": a.dynamics.crest_factor_db,
        "dynamics.clipping_ratio_pct": a.dynamics.clipping_ratio * 100.0,
        "dynamics.stereo_correlation": a.dynamics.stereo_correlation,
        "integrity.channel_balance_db": a.integrity.channel_balance_db,
        "integrity.clip_event_count": a.integrity.clip_event_count,
        "integrity.longest_clip_run_ms": a.integrity.longest_clip_run_ms,
        "integrity.noise_floor_dbfs": a.integrity.noise_floor_dbfs,
        "integrity.spectral_rolloff_hz": a.integrity.spectral_rolloff_hz,
        "integrity.leading_silence_sec": a.integrity.leading_silence_sec,
        "integrity.trailing_silence_sec": a.integrity.trailing_silence_sec,
    }
    for band, val in a.spectral.band_energy_db.items():
        flat[f"spectral.band_energy_db.{band}"] = val
    for band, val in a.spectral.kb_band_density_db.items():
        flat[f"spectral.kb_band_density_db.{band}"] = val
    for ch, val in a.integrity.dc_offset_dbfs.items():
        flat[f"integrity.dc_offset_dbfs.{ch}"] = val
    for band, val in a.integrity.band_stereo_correlation.items():
        flat[f"integrity.band_stereo_correlation.{band}"] = val
    return flat


def _build_comparison_table(raw: FileAnalysis, refined: FileAnalysis) -> list[dict]:
    """Tableau ligne par ligne : TOUTES les métriques mesurées, brut vs
    raffiné, avec delta — pour ne rien laisser passer entre les deux."""
    raw_flat = _flatten_metrics(raw)
    refined_flat = _flatten_metrics(refined)
    rows = []
    for key in sorted(set(raw_flat) | set(refined_flat)):
        rv, fv = raw_flat.get(key), refined_flat.get(key)
        delta = round(fv - rv, 3) if isinstance(rv, (int, float)) and isinstance(fv, (int, float)) else None
        rows.append(
            {
                "metric": key,
                "raw": round(rv, 3) if isinstance(rv, float) else rv,
                "refined": round(fv, 3) if isinstance(fv, float) else fv,
                "delta": delta,
            }
        )
    return rows


def _validate_output(
    output_analysis: FileAnalysis,
    reference: ReferenceMeasurement | None,
    profile: DestinationProfile,
    input_duration_sec: float,
    output_duration_sec: float,
) -> tuple[list[str], list[str]]:
    """
    Gate de conformité final, appliqué au WAV RÉELLEMENT écrit sur disque
    (relu depuis le fichier, pas le buffer en mémoire) — c'est ce fichier-là
    qui sera posté sur la plateforme, donc c'est lui qui doit être vérifié.

    Le true peak / LUFS vérifiés ici viennent de `output_analysis.loudness`
    — mesurés sur le fichier RÉELLEMENT écrit sur disque, PAS sur le buffer
    d'avant écriture. Point important : avant l'introduction du
    rééchantillonnage vers le format de livraison, ce gate utilisait la
    mesure d'avant écriture (buffer déjà limité mais pas encore rééchantillonné) ;
    or une conversion de fréquence d'échantillonnage peut créer un
    dépassement inter-échantillon qui n'existait pas avant elle (ringing/
    overshoot classique de tout resampling). `process_file` rééchantillonne
    maintenant AVANT le limiteur final précisément pour que ce cas ne puisse
    plus se produire, mais on vérifie ici la mesure du fichier réel quoi
    qu'il arrive plutôt que de dépendre uniquement de cet ordre d'opérations.

    Retourne (erreurs_bloquantes, avertissements). Toute erreur bloquante
    déclenche `OutputValidationError` dans `process_file` (fichier supprimé).
    """
    errors: list[str] = []
    warns: list[str] = []
    integ = output_analysis.integrity

    if integ.has_nan or integ.has_inf:
        errors.append("Échantillons invalides (NaN/Inf) détectés dans le fichier de sortie — export corrompu.")

    measured_tp = output_analysis.loudness.true_peak_dbtp
    measured_tp = measured_tp if np.isfinite(measured_tp) else None
    worst_tp = measured_tp
    if reference is not None and measured_tp is not None:
        worst_tp = max(measured_tp, reference.true_peak_dbtp)
        if abs(measured_tp - reference.true_peak_dbtp) > QC_CROSS_VALIDATION_TOLERANCE_DB:
            warns.append(
                f"Écart de mesure true peak entre refinr ({measured_tp:.2f}dBTP) et la contre-vérification "
                f"libebur128 ({reference.true_peak_dbtp:.2f}dBTP) > {QC_CROSS_VALIDATION_TOLERANCE_DB}dB — "
                "à vérifier avec un meter calibré avant diffusion."
            )
    if worst_tp is not None and worst_tp > profile.true_peak_ceiling_dbtp + QC_TRUE_PEAK_EPSILON_DB:
        errors.append(
            f"True peak final {worst_tp:.2f}dBTP dépasse le plafond du profil '{profile.key}' "
            f"({profile.true_peak_ceiling_dbtp}dBTP) — risque de distorsion à la transcodification "
            "(ex: Ogg Vorbis Spotify, AAC YouTube)."
        )

    measured_lufs = output_analysis.loudness.integrated_lufs
    measured_lufs = measured_lufs if np.isfinite(measured_lufs) else None
    if measured_lufs is None:
        errors.append("Loudness intégrée finale non mesurable (signal silencieux ou invalide).")
    else:
        if reference is not None and abs(measured_lufs - reference.integrated_lufs) > QC_CROSS_VALIDATION_TOLERANCE_DB:
            warns.append(
                f"Écart de mesure LUFS entre refinr ({measured_lufs:.2f}) et la contre-vérification "
                f"libebur128 ({reference.integrated_lufs:.2f}) > {QC_CROSS_VALIDATION_TOLERANCE_DB} LU."
            )
        lufs_diff = abs(measured_lufs - profile.target_lufs)
        if lufs_diff > QC_LUFS_FAIL_LU:
            errors.append(
                f"Loudness intégrée finale {measured_lufs:.2f} LUFS trop éloignée de la cible du profil "
                f"'{profile.key}' ({profile.target_lufs} LUFS, écart {lufs_diff:.2f} LU) — le leveling a échoué."
            )
        elif lufs_diff > QC_LUFS_WARN_LU:
            warns.append(
                f"Loudness finale {measured_lufs:.2f} LUFS à {lufs_diff:.2f} LU de la cible "
                f"({profile.target_lufs} LUFS) — dans la tolérance mais à surveiller."
            )

    if reference is None:
        warns.append(
            "Contre-vérification libebur128 indisponible (pyebur128 non installé) — seule la mesure "
            "refinr (pyloudnorm) a été utilisée pour ce fichier."
        )

    if output_duration_sec < input_duration_sec - QC_DURATION_TRUNCATION_TOLERANCE_SEC:
        errors.append(
            f"Durée de sortie ({output_duration_sec:.2f}s) plus courte que la source "
            f"({input_duration_sec:.2f}s) — troncature suspectée."
        )

    if integ.dc_offset_issue:
        warns.append(f"Offset DC détecté en sortie : {integ.dc_offset_dbfs}.")
    if integ.channel_balance_issue:
        warns.append(f"Déséquilibre de balance L/R en sortie : {integ.channel_balance_db:+.2f}dB.")
    if integ.clip_event_count > 0:
        errors.append(
            f"{integ.clip_event_count} run(s) d'écrêtage dur détecté(s) dans le fichier de sortie "
            f"(plus long : {integ.longest_clip_run_ms:.1f}ms) — le limiteur final n'a pas suffi."
        )
    if integ.localized_phase_issue_bands:
        warns.append(f"Problème de phase localisé en sortie sur : {', '.join(integ.localized_phase_issue_bands)}.")
    if integ.mono_fold_issue:
        warns.append(
            f"Perte de {integ.mono_fold_loss_db:.1f}dB au repli mono (L+R)/2 en sortie — au-delà de la perte "
            "~3dB normale d'un signal stéréo non corrélé, signe d'une annulation de phase réelle. À vérifier "
            "en écoute mono (téléphone, enceinte Bluetooth, club en mono partiel) avant diffusion."
        )

    return errors, warns


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
    suno_mode: bool = False
    exported_presets: list[str] = dataclasses.field(default_factory=list)
    qc_passed: bool = True
    qc_correction_passes: int = 0
    output_diagnostic: dict = dataclasses.field(default_factory=dict)
    comparison_table: list[dict] = dataclasses.field(default_factory=list)
    reference_measurement: dict | None = None


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
    output_subtype: str | None = None,
    suno_mode: bool = False,
    export_eq_preset_dir: str | Path | None = None,
) -> FileProcessingReport:
    """
    `suno_mode` : active les corrections a priori spécifiques aux artefacts
    connus des générateurs IA type Suno (voir `config/suno_artifacts_kb.md`
    et `proq4_control.decide_bands`) — opt-in, jamais activé par défaut.

    `export_eq_preset_dir` : si fourni, écrit le(s) preset(s) piloté(s)
    dynamiquement pour CE fichier en vrais `.aupreset` réutilisables dans
    Logic Pro (voir `preset_types.write_aupreset`) — le preset Pro-Q4
    (toujours), le preset TAIP (si `_find_taip_template` en a trouvé un) et
    le preset J37 (si `_find_j37_template` en a trouvé un) n'existent sinon
    qu'en mémoire, pilotés directement vers `au_host`.

    `output_subtype` : override manuel du bit depth de sortie (voir
    `audio_io.save_wav`). Si None (par défaut), utilise `profile.output_bit_depth`
    — le sample rate de sortie suit TOUJOURS `profile.output_sample_rate`
    (resample si la source diffère), pour ne jamais livrer un fichier à un
    sample rate/bit depth qui ne correspond pas à ce qu'attend la
    destination (voir `profiles.DestinationProfile`).
    """
    t0 = time.monotonic()
    warnings: list[str] = []
    exported_presets: list[str] = []

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
    eq_bands = decide_bands(file_analysis, suno_mode=suno_mode)
    if eq_bands:
        eq_reason = (
            f"{len(eq_bands)} bande(s) EQ décidée(s) à partir du diagnostic ci-dessus (voir raison "
            "détaillée par bande)."
        )
    else:
        eq_reason = "Aucune correction EQ jugée nécessaire — diagnostic dans les limites normales."
    eq_preset = make_dynamic_preset(f"refinr auto EQ — {input_path.name}", eq_bands)

    if export_eq_preset_dir is not None and eq_bands:
        export_path = Path(export_eq_preset_dir) / f"{input_path.stem}_ProQ4.aupreset"
        try:
            write_aupreset(eq_preset, export_path)
            exported_presets.append(str(export_path))
        except OSError as exc:
            warnings.append(f"Export du preset .aupreset échoué pour {input_path.name}: {exc}")

    # Saturation : pilotage DYNAMIQUE (taip_control) dès qu'un preset TAIP de
    # référence est disponible dans la bibliothèque (voir _find_taip_template)
    # — même logique que l'EQ Pro-Q4 ci-dessus, pas de sélection parmi des
    # presets figés. Tape (J37) : pilotage DYNAMIQUE PARTIEL (j37_control) dès
    # que le preset de référence exact est disponible (voir _find_j37_template)
    # — Saturation/Noise/Wow/Flutter pilotés, Speed/Bias/Formula/Modeled Tracks
    # figés sur ce preset (voir docstring de j37_control.py). Sans preset de
    # référence pour l'un ou l'autre rôle, repli sur la sélection statique par
    # tags (preset_mapping), comme avant l'introduction du pilotage dynamique.
    taip_template = _find_taip_template(library)
    j37_template = _find_j37_template(library)
    static_roles = tuple(
        role
        for role, template in ((PluginRole.SATURATION, taip_template), (PluginRole.TAPE, j37_template))
        if template is None
    )
    selection = select_chain(library, file_analysis, roles=static_roles)

    steps: list[ChainStepReport] = []
    processed_buffer = gained_buffer

    # Résultats dynamiques indexés par rôle — assemblés dans l'ORDRE FIXE
    # (SATURATION puis TAPE, voir chosen_extra_presets ci-dessous) quel que
    # soit l'ordre dans lequel dynamique/statique sont décidés ici, pour que
    # la chaîne AU garde toujours le flux de signal documenté EQ -> saturation
    # -> tape -> leveling.
    dynamic_results: dict[PluginRole, SelectionResult] = {}

    if taip_template is not None:
        taip_params = decide_taip_params(file_analysis)
        taip_preset = make_dynamic_taip_preset(
            f"refinr auto saturation (TAIP) — {input_path.name}", taip_params, taip_template
        )
        taip_reason = (
            "Saturation pilotée DYNAMIQUEMENT (Baby Audio TAIP, même principe que l'EQ Pro-Q4) à partir du "
            f"diagnostic de ce fichier (voir taip_control.decide_params) — paramètres : {taip_params}."
        )
        dynamic_results[PluginRole.SATURATION] = SelectionResult(preset=taip_preset, reason=taip_reason)

        if export_eq_preset_dir is not None:
            taip_export_path = Path(export_eq_preset_dir) / f"{input_path.stem}_TAIP.aupreset"
            try:
                write_aupreset(taip_preset, taip_export_path)
                exported_presets.append(str(taip_export_path))
            except OSError as exc:
                warnings.append(f"Export du preset TAIP .aupreset échoué pour {input_path.name}: {exc}")

    if j37_template is not None:
        j37_params = decide_j37_params(file_analysis)
        j37_preset = make_dynamic_j37_preset(f"refinr auto tape (J37) — {input_path.name}", j37_params, j37_template)
        j37_reason = (
            "Tape pilotée DYNAMIQUEMENT EN PARTIE (Waves Abbey Road J37 : Saturation/Noise/Wow/Flutter, voir "
            f"j37_control.decide_params) à partir du diagnostic de ce fichier — paramètres : {j37_params}. "
            "Speed/Bias/Formula/Modeled Tracks restent figés sur le preset de référence (pas de mapping "
            "scalaire possible pour ces quatre-là, voir j37_control.py)."
        )
        dynamic_results[PluginRole.TAPE] = SelectionResult(preset=j37_preset, reason=j37_reason)

        if export_eq_preset_dir is not None:
            j37_export_path = Path(export_eq_preset_dir) / f"{input_path.stem}_J37.aupreset"
            try:
                write_aupreset(j37_preset, j37_export_path)
                exported_presets.append(str(j37_export_path))
            except OSError as exc:
                warnings.append(f"Export du preset J37 .aupreset échoué pour {input_path.name}: {exc}")

    chosen_extra_presets: list[tuple[PluginRole, SelectionResult]] = []  # saturation/tape, EQ traité à part
    for role in (PluginRole.SATURATION, PluginRole.TAPE):
        result = dynamic_results.get(role) or selection.get(role)
        if result is None:
            continue
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

    # --- Transient shaping (attaque/punch) -----------------------------------
    # Nouveau stage, absent de la chaîne EQ -> saturation -> tape -> leveling
    # historique : corrige les attaques molles/lissées (artefact IA connu)
    # sur du matériel déjà compressé, voir transient_shaping.decide_attack_amount_db.
    pre_transient_measure = loudness.measure(processed_buffer)
    transient_decision = decide_attack_amount_db(processed_buffer, file_analysis)
    processed_buffer = apply_transient_shaping(processed_buffer, transient_decision.attack_amount_db)
    post_transient_measure = loudness.measure(processed_buffer)
    steps.append(
        ChainStepReport(
            role="transient_shaping",
            plugin_name="refinr.transient_shaping (transient designer maison)",
            preset_name="auto",
            preset_path="",
            reason=transient_decision.reason,
            pre_measurement=_measurement_to_dict(pre_transient_measure),
            post_measurement=_measurement_to_dict(post_transient_measure),
            extra={"attack_amount_db": round(transient_decision.attack_amount_db, 2)},
        )
    )

    # --- Contrôle actif de largeur stéréo ------------------------------------
    # Jusqu'ici analysis.py se contentait de FLAGGER une corrélation stéréo
    # anormale (tags wide_stereo/mono_leaning) ; on corrige maintenant
    # réellement le canal Side (mono-safe sous 150Hz), voir stereo_width.py.
    pre_width_measure = loudness.measure(processed_buffer)
    width_decision = decide_width_factor(file_analysis)
    processed_buffer = apply_stereo_width(processed_buffer, width_decision.width_factor)
    post_width_measure = loudness.measure(processed_buffer)
    steps.append(
        ChainStepReport(
            role="stereo_width",
            plugin_name="refinr.stereo_width (M/S, mono-safe <150Hz)",
            preset_name="auto",
            preset_path="",
            reason=width_decision.reason,
            pre_measurement=_measurement_to_dict(pre_width_measure),
            post_measurement=_measurement_to_dict(post_width_measure),
            extra={"width_factor": width_decision.width_factor},
        )
    )

    # --- Correction macro-dynamique (LRA) ------------------------------------
    # Le limiteur final (loudness.limit_true_peak) ne réagit qu'aux pics
    # individuels (lookahead de quelques ms) ; rien jusqu'ici ne traitait les
    # écarts de niveau ENTRE SECTIONS d'un même morceau. Se déclenche
    # seulement si la LRA mesurée sur la source dépasse le seuil "gênant"
    # (voir macro_dynamics.decide_macro_compression) — corrective, pas
    # systématique, et ratio < 1.0 : réduit l'écart, n'aplatit jamais la
    # dynamique musicale.
    pre_macro_measure = loudness.measure(processed_buffer)
    macro_decision = decide_macro_compression(file_analysis)
    processed_buffer = apply_macro_compression(processed_buffer, macro_decision.ratio)
    post_macro_measure = loudness.measure(processed_buffer)
    steps.append(
        ChainStepReport(
            role="macro_dynamics",
            plugin_name="refinr.macro_dynamics (rider de gain macro, piloté LRA)",
            preset_name="auto",
            preset_path="",
            reason=macro_decision.reason,
            pre_measurement=_measurement_to_dict(pre_macro_measure),
            post_measurement=_measurement_to_dict(post_macro_measure),
            extra={"ratio": macro_decision.ratio, "enabled": macro_decision.enabled},
        )
    )

    # --- Leveling final + gate QC avec repasses correctives ------------------
    # `_validate_output` peut échouer pour des raisons purement liées au gain
    # (dérive LUFS, dépassement true peak) — dans ce cas précis, plutôt que
    # d'abandonner immédiatement (supprimer le fichier, lever une erreur),
    # on retente jusqu'à MAX_QC_CORRECTION_PASSES fois avec une correction de
    # gain calculée à partir de l'écart RÉELLEMENT mesuré à la tentative
    # précédente. Les échecs non corrigibles par le gain (NaN/Inf, troncature,
    # écrêtage dur, etc.) ne déclenchent PAS de repasse : `step_correction`
    # reste à 0.0 et on sort de la boucle immédiatement, comme avant.
    correction_db = 0.0
    correction_log: list[str] = []
    qc_errors: list[str] = []
    qc_warnings: list[str] = []
    final_measurement: dict = {}
    output_analysis: FileAnalysis | None = None
    output_diagnostic: dict = {}
    comparison_table: list[dict] = []
    reference_dict: dict | None = None
    level_gain_db = 0.0
    output_buffer = None

    for attempt in range(MAX_QC_CORRECTION_PASSES + 1):
        pre_level_measure = loudness.measure(processed_buffer)
        leveled_buffer, level_gain_db = loudness.gain_to_target_lufs(
            processed_buffer,
            target_lufs=profile.target_lufs + correction_db,
            ceiling_dbtp=profile.true_peak_ceiling_dbtp,
        )
        # Conforme la sortie au format de LIVRAISON du profil, pas au format
        # de la source : sans ça, un WAV source à 96kHz/32-bit resterait
        # 96kHz/32-bit en sortie même pour une plateforme qui attend du
        # 24-bit/44.1kHz.
        #
        # IMPORTANT : le rééchantillonnage se fait AVANT le limiteur, pas
        # après. La conversion de fréquence d'échantillonnage (resample_poly)
        # peut créer un dépassement inter-échantillon qui n'existait pas dans
        # le signal source (ringing/overshoot classique de toute conversion
        # de fréquence) — si on limitait AVANT de rééchantillonner, ce
        # dépassement se retrouverait dans le fichier livré SANS avoir été vu
        # par le limiteur. En rééchantillonnant d'abord, le détecteur de true
        # peak suréchantillonné du limiteur (limit_true_peak) est la DERNIÈRE
        # étape à toucher le signal à la fréquence réellement livrée : le
        # respect du plafond est garanti par construction, pas seulement
        # détecté a posteriori par le gate QC ci-dessous.
        delivery_pre_limit = resample_if_needed(leveled_buffer, profile.output_sample_rate)
        if attempt == 0 and delivery_pre_limit.sample_rate != leveled_buffer.sample_rate:
            warnings.append(
                f"Rééchantillonné {leveled_buffer.sample_rate}Hz -> {delivery_pre_limit.sample_rate}Hz "
                f"pour correspondre au format de livraison du profil '{profile.key}' (AVANT le limiteur final, "
                "pour que le plafond de true peak soit garanti au format réellement livré)."
            )

        limited_buffer = loudness.limit_true_peak(delivery_pre_limit, ceiling_dbtp=profile.true_peak_ceiling_dbtp)
        final_measurement = _measurement_to_dict(loudness.measure(limited_buffer))

        actual_output_subtype = output_subtype or profile.output_bit_depth
        save_wav(limited_buffer, output_path, subtype=actual_output_subtype)

        # --- Gate de validation post-traitement ------------------------------
        # On ne fait PAS confiance au buffer en mémoire : on relit le fichier
        # RÉELLEMENT écrit sur disque (c'est lui qui sera posté sur la
        # plateforme) et on le ré-analyse intégralement — mêmes vérifications
        # que sur la source (spectral, dynamique, intégrité complète), plus une
        # contre-vérification indépendante de la loudness/true-peak via
        # libebur128 (voir reference_loudness.py).
        output_buffer = load_wav(output_path)
        output_analysis = analyze(output_buffer)
        output_diagnostic = _build_diagnostic(output_analysis)
        comparison_table = _build_comparison_table(file_analysis, output_analysis)

        reference = measure_reference(output_buffer)
        reference_dict = (
            {
                "integrated_lufs": round(reference.integrated_lufs, 3),
                "true_peak_dbtp": round(reference.true_peak_dbtp, 3),
                "sample_peak_dbfs": round(reference.sample_peak_dbfs, 3),
                "loudness_range_lu": round(reference.loudness_range_lu, 3) if reference.loudness_range_lu else None,
                "source": reference.source,
            }
            if reference is not None
            else None
        )

        qc_errors, qc_warnings = _validate_output(
            output_analysis=output_analysis,
            reference=reference,
            profile=profile,
            input_duration_sec=buffer.duration_seconds,
            output_duration_sec=output_buffer.duration_seconds,
        )

        if not qc_errors or attempt == MAX_QC_CORRECTION_PASSES:
            break

        step_correction = 0.0
        measured_tp = final_measurement.get("true_peak_dbtp")
        measured_lufs = final_measurement.get("integrated_lufs")
        if any("True peak" in e for e in qc_errors) and measured_tp is not None:
            # Priorité au true peak : le dépassement est TOUJOURS bloquant
            # (voir _validate_output), donc on baisse le gain net d'au moins
            # l'écart mesuré, avec une petite marge de sécurité.
            overshoot = measured_tp - profile.true_peak_ceiling_dbtp
            step_correction -= overshoot + 0.2
        elif any("Loudness" in e for e in qc_errors) and measured_lufs is not None:
            # Corrige la cible visée à la prochaine tentative de l'écart
            # RÉELLEMENT mesuré (pas juste "on retente pareil").
            step_correction += profile.target_lufs - measured_lufs

        if step_correction == 0.0:
            # Échec non corrigible par un simple ajustement de gain (NaN/Inf,
            # troncature, écrêtage dur, DC offset...) — pas la peine de
            # reboucler, on sort avec les erreurs actuelles.
            break

        correction_db += step_correction
        correction_log.append(
            f"Repasse corrective QC #{attempt + 1}/{MAX_QC_CORRECTION_PASSES} : {' | '.join(qc_errors)} -> "
            f"correction de gain cumulée ajustée à {correction_db:+.2f}dB pour la tentative suivante."
        )

    warnings.extend(correction_log)
    warnings.extend(qc_warnings)

    steps.append(
        ChainStepReport(
            role=PluginRole.LEVELING.value,
            plugin_name="refinr.loudness (gain + limiteur true-peak)",
            preset_name=profile.label,
            preset_path="",
            reason=(
                f"Leveling final vers profil '{profile.key}': cible {profile.target_lufs} LUFS / "
                f"{profile.true_peak_ceiling_dbtp} dBTP."
                + (f" ({len(correction_log)} repasse(s) corrective(s) QC appliquée(s).)" if correction_log else "")
            ),
            pre_measurement=_measurement_to_dict(pre_level_measure),
            post_measurement=final_measurement,
            extra={"leveling_gain_db": round(level_gain_db, 3), "qc_correction_passes": len(correction_log)},
        )
    )

    if qc_errors:
        output_path.unlink(missing_ok=True)
        raise OutputValidationError(
            f"{input_path.name} : validation de sortie échouée après {len(correction_log)} repasse(s) "
            f"corrective(s), fichier NON produit — " + " | ".join(qc_errors)
        )

    assert output_analysis is not None and output_buffer is not None  # garanti par la boucle ci-dessus

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
        final_measurement=final_measurement,
        warnings=warnings,
        duration_seconds=round(duration, 3),
        au_hosting_used=AU_HOSTING_AVAILABLE,
        suno_mode=suno_mode,
        exported_presets=exported_presets,
        output_diagnostic=output_diagnostic,
        comparison_table=comparison_table,
        reference_measurement=reference_dict,
        qc_passed=True,
        qc_correction_passes=len(correction_log),
    )
