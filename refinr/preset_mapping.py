"""
Bibliothèque de presets AU + logique de sélection PAR FICHIER.

Convention de rangement (modifiable via `PresetLibrary.load`) :

    config/presets/
      eq/
        dark_compensate_bright.aupreset
        dark_compensate_bright.meta.yaml   (optionnel)
        vocal_deess_light.aupreset
      saturation/
        saturn_light_warmth.aupreset
        hg2_heavy_drive.aupreset
      tape/
        j37_15ips_subtle.aupreset

Chaque .aupreset peut avoir un fichier compagnon `<meme_nom>.meta.yaml` qui
décrit QUAND ce preset doit être choisi. Exemple :

    tags: [warmth, gentle]
    intensity: light
    suited_for:
      tags_any: [dark, already_compressed]
      tags_none: [very_dynamic]
    priority: 1

Si un preset n'a pas de .meta.yaml, il est traité comme un candidat
"universel" pour son rôle (toujours éligible, priorité 0) — pratique au
début, le temps d'annoter progressivement la bibliothèque.

La sélection n'est donc jamais générique "un preset pour tous les WAV" :
elle dépend des tags produits par `analysis.FileAnalysis.summary_tags()`
pour CE fichier précis.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from .analysis import FileAnalysis
from .preset_types import PluginPreset, PluginRole, load_aupreset

DEFAULT_PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"

_ROLE_DIRS = {
    PluginRole.EQ: "eq",
    PluginRole.SATURATION: "saturation",
    PluginRole.TAPE: "tape",
}


@dataclasses.dataclass
class SuitedFor:
    tags_any: tuple[str, ...] = ()
    tags_none: tuple[str, ...] = ()


@dataclasses.dataclass
class PresetEntry:
    preset: PluginPreset
    suited_for: SuitedFor
    priority: int = 0

    def score(self, file_tags: set[str]) -> float | None:
        """None = inéligible. Sinon score, plus haut = meilleur candidat."""
        if self.suited_for.tags_none and file_tags.intersection(self.suited_for.tags_none):
            return None
        if not self.suited_for.tags_any:
            return float(self.priority)  # candidat universel
        overlap = file_tags.intersection(self.suited_for.tags_any)
        if not overlap:
            return None
        return float(self.priority) + len(overlap)


@dataclasses.dataclass
class PresetLibrary:
    entries_by_role: dict[PluginRole, list[PresetEntry]]

    @classmethod
    def load(cls, root: str | Path = DEFAULT_PRESETS_ROOT) -> "PresetLibrary":
        root = Path(root)
        entries_by_role: dict[PluginRole, list[PresetEntry]] = {role: [] for role in _ROLE_DIRS}

        for role, dirname in _ROLE_DIRS.items():
            role_dir = root / dirname
            if not role_dir.is_dir():
                continue
            for preset_path in sorted(role_dir.glob("*.aupreset")):
                preset = load_aupreset(preset_path)
                preset.role = role
                meta_path = preset_path.with_suffix(".meta.yaml")
                suited_for = SuitedFor()
                priority = 0
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = yaml.safe_load(fh) or {}
                    preset.tags = tuple(meta.get("tags", []))
                    preset.intensity = meta.get("intensity")
                    priority = int(meta.get("priority", 0))
                    sf = meta.get("suited_for", {}) or {}
                    suited_for = SuitedFor(
                        tags_any=tuple(sf.get("tags_any", [])),
                        tags_none=tuple(sf.get("tags_none", [])),
                    )
                entries_by_role[role].append(PresetEntry(preset=preset, suited_for=suited_for, priority=priority))

        return cls(entries_by_role=entries_by_role)

    def is_empty(self, role: PluginRole) -> bool:
        return len(self.entries_by_role.get(role, [])) == 0


@dataclasses.dataclass
class SelectionResult:
    preset: PluginPreset | None
    reason: str


def select_preset_for_role(
    library: PresetLibrary,
    role: PluginRole,
    analysis: FileAnalysis,
) -> SelectionResult:
    entries = library.entries_by_role.get(role, [])
    if not entries:
        return SelectionResult(preset=None, reason=f"Aucun preset disponible pour le rôle '{role.value}'.")

    file_tags = set(analysis.summary_tags())
    scored = []
    for entry in entries:
        s = entry.score(file_tags)
        if s is not None:
            scored.append((s, entry))

    if not scored:
        return SelectionResult(
            preset=None,
            reason=f"Aucun preset '{role.value}' compatible avec les tags du fichier ({sorted(file_tags)}).",
        )

    # tri stable : meilleur score d'abord, puis nom de preset pour déterminisme
    scored.sort(key=lambda item: (-item[0], item[1].preset.name))
    best_score, best_entry = scored[0]
    reason = (
        f"Sélectionné pour rôle '{role.value}' (score={best_score:.1f}) "
        f"via tags fichier {sorted(file_tags)} ∩ preset {sorted(best_entry.suited_for.tags_any) or 'universel'}."
    )
    return SelectionResult(preset=best_entry.preset, reason=reason)


def select_chain(
    library: PresetLibrary,
    analysis: FileAnalysis,
    roles: tuple[PluginRole, ...] = (PluginRole.EQ, PluginRole.SATURATION, PluginRole.TAPE),
) -> dict[PluginRole, SelectionResult]:
    """Sélectionne, pour CE fichier (via `analysis`), le meilleur preset par rôle."""
    return {role: select_preset_for_role(library, role, analysis) for role in roles}
