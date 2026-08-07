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
    output_subtype: str,
) -> BatchFileOutcome:
    """
    Fonction exécutée dans un process worker séparé : recharge la bibliothèque
    de presets et le catalogue de profils localement (plus simple/robuste à
    pickler que de repasser les objets complexes d'un process à l'autre),
    puis traite un seul fichier.
    """
    try:
        library = PresetLibrary.load(presets_root)
        catalog = ProfileCatalog_load(profiles_path)
        profile = catalog.get(profile_key)

        in_path = Path(input_path)
        out_path = Path(output_dir) / f"{in_path.stem}_{profile_key}.wav"
        report = process_file(in_path, out_path, library, profile, catalog, output_subtype=output_subtype)
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
    output_subtype: str = "PCM_24",
    on_progress=None,
) -> BatchResult:
    """
    `on_progress`, si fourni, est appelé avec chaque `BatchFileOutcome` au
    fur et à mesure qu'il se termine (utile pour une barre de progression
    GUI) — voir gui/worker.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
