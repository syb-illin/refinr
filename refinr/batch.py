"""
Traitement batch en parallèle de plusieurs WAV, avec rapport détaillé.

Chaque fichier est traité indépendamment via `chain.process_file` (donc
avec SA propre sélection de presets, jamais un traitement générique
partagé). Le parallélisme utilise des process séparés (pas des threads) car
le hosting AU / DSP est CPU-bound et parfois pas thread-safe.
"""

from __future__ import annotations

import dataclasses
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .chain import FileProcessingReport, process_file
from .preset_mapping import PresetLibrary
from .profiles import DestinationProfile, ProfileCatalog


@dataclasses.dataclass
class BatchFileOutcome:
    input_path: str
    success: bool
    report: FileProcessingReport | None
    error: str | None


@dataclasses.dataclass
class BatchResult:
    outcomes: list[BatchFileOutcome]

    @property
    def succeeded(self) -> list[BatchFileOutcome]:
        return [o for o in self.outcomes if o.success]

    @property
    def failed(self) -> list[BatchFileOutcome]:
        return [o for o in self.outcomes if not o.success]


def _process_one(
    input_path: str,
    output_dir: str,
    presets_root: str,
    profile_key: str,
    profiles_path: str,
    output_subtype: str | None = None,
    suno_mode: bool = False,
    export_eq_preset_dir: str | None = None,
    custom_profile: dict | None = None,
) -> BatchFileOutcome:
    """
    Fonction exécutée dans un process worker séparé : recharge la bibliothèque
    de presets et le catalogue de profils localement (plus simple/robuste à
    pickler que de repasser les objets complexes d'un process à l'autre),
    puis traite un seul fichier.

    `custom_profile`, si fourni (`{"target_lufs": float, "true_peak_ceiling_dbtp":
    float, "output_sample_rate": int (optionnel), "output_bit_depth": str (optionnel)}`),
    REMPLACE le profil chargé depuis `profiles_path` par des cibles définies
    manuellement dans la GUI (sliders "Mastering Targets" / "Custom Targets",
    y compris sample rate/bit depth de livraison) — pratique pour une cible
    non couverte par les profils YAML prédéfinis. `profile_key` est alors
    ignoré pour la résolution du profil, mais reste utilisé pour nommer le
    fichier de sortie. `output_sample_rate`/`output_bit_depth` absents ->
    défauts `DestinationProfile` (44100Hz / PCM_24).
    """
    try:
        library = PresetLibrary.load(presets_root)
        catalog = ProfileCatalog_load(profiles_path)
        if custom_profile is not None:
            profile_kwargs = {
                "key": "custom",
                "label": "Cibles personnalisées",
                "target_lufs": float(custom_profile["target_lufs"]),
                "true_peak_ceiling_dbtp": float(custom_profile["true_peak_ceiling_dbtp"]),
                "boosts_quiet": None,
                "notes": "Défini manuellement via les sliders GUI, pas un profil YAML prédéfini.",
            }
            if custom_profile.get("output_sample_rate") is not None:
                profile_kwargs["output_sample_rate"] = int(custom_profile["output_sample_rate"])
            if custom_profile.get("output_bit_depth") is not None:
                profile_kwargs["output_bit_depth"] = str(custom_profile["output_bit_depth"])
            profile = DestinationProfile(**profile_kwargs)
        else:
            profile = catalog.get(profile_key)

        in_path = Path(input_path)
        out_path = Path(output_dir) / f"{in_path.stem}_{profile_key}.wav"
        report = process_file(
            in_path,
            out_path,
            library,
            profile,
            catalog,
            output_subtype=output_subtype,
            suno_mode=suno_mode,
            export_eq_preset_dir=export_eq_preset_dir,
        )
        return BatchFileOutcome(input_path=input_path, success=True, report=report, error=None)
    except Exception:  # noqa: BLE001 - on veut capturer toute erreur pour ne pas planter tout le batch
        return BatchFileOutcome(input_path=input_path, success=False, report=None, error=traceback.format_exc())


def ProfileCatalog_load(path: str) -> ProfileCatalog:
    from .profiles import load_profiles

    return load_profiles(path)


def run_batch(
    input_paths: list[str | Path],
    output_dir: str | Path,
    presets_root: str | Path,
    profile_key: str,
    profiles_path: str | Path,
    max_workers: int = 4,
    output_subtype: str | None = None,
    on_progress=None,
    suno_mode: bool = False,
    export_eq_presets: bool = False,
    custom_profile: dict | None = None,
) -> BatchResult:
    """
    `on_progress`, si fourni, est appelé avec chaque `BatchFileOutcome` au
    fur et à mesure qu'il se termine (utile pour une barre de progression
    GUI) — voir gui/worker.py.

    `suno_mode` : voir `chain.process_file`. `export_eq_presets` : si True,
    écrit chaque preset Pro-Q4 dynamique en `.aupreset` dans
    `output_dir/presets_aupreset/` (voir `preset_types.write_aupreset`).
    `custom_profile` : voir `_process_one` — cibles LUFS/true peak définies
    manuellement (sliders GUI), remplace le profil résolu via `profile_key`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = str(output_dir / "presets_aupreset") if export_eq_presets else None

    outcomes: list[BatchFileOutcome] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                str(p),
                str(output_dir),
                str(presets_root),
                profile_key,
                str(profiles_path),
                output_subtype,
                suno_mode,
                export_dir,
                custom_profile,
            ): str(p)
            for p in input_paths
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            if on_progress is not None:
                on_progress(outcome)

    # ré-ordonne selon l'ordre d'entrée pour un rapport lisible
    order = {str(p): i for i, p in enumerate(input_paths)}
    outcomes.sort(key=lambda o: order.get(o.input_path, 1 << 30))
    return BatchResult(outcomes=outcomes)
