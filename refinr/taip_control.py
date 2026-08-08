"""
Pilotage DYNAMIQUE de Baby Audio TAIP — pendant de `proq4_control.py` pour
le rôle SATURATION : au lieu de choisir parmi des presets figés
(`preset_mapping.py`, toujours disponible en fallback), construit
programmatiquement les 15 paramètres TAIP (`drive`, `glue`, `wear`, ...) à
partir de ce que `analysis.py` mesure sur CE fichier précis.

Contrairement au Pro-Q4 (`proq4_control.py`, format binaire
`FabFilterPluginState` reverse-engineré champ par champ) et au J37
(`preset_types` + `.aupreset` figés, format `RealWorld` non documenté),
le format TAIP est un blob `jucePluginState` = header binaire + XML
directement lisible :

    b"VC2!" + <uint32 LE: longueur XML> + XML UTF-8 (+ padding nul)

    <?xml version="1.0" encoding="UTF-8"?> <TAIP_1 scale_fac_id="..."
    PHASE_OFFSET="..." PRESET_NAME="..."><PARAM id="drive" value="4.0"/>...
    </TAIP_1>

Confirmé en inspectant `config/presets/saturation/taip_suno_tuned.aupreset`
(voir `config/presets/saturation/taip_parameter_reference.md` pour la table
des 15 id/valeur). Aucun reverse-engineering binaire nécessaire ici — les
noms de paramètres sont explicites dans le XML.

Statut : module autonome, testé, mais PAS ENCORE branché par défaut dans
`chain.py` (contrairement à l'EQ Pro-Q4, toujours pilotée dynamiquement).
La sélection de preset TAPE/SATURATION par tags (`preset_mapping.py`) reste
le chemin par défaut ; ce module est une voie parallèle plus fine, sur le
même modèle que `proq4_control.py`, prête à être activée quand souhaité.
"""

from __future__ import annotations

import dataclasses
import re
import struct
from pathlib import Path

from .analysis import FileAnalysis
from .preset_types import PluginPreset

_HEADER_MAGIC = b"VC2!"

# Valeurs par défaut ("bypass doux") — reprises du seul preset réel
# disponible (`taip_suno_tuned.aupreset`, id="host_bypass"/"power"/"link"/
# "model"/"colorID"/"mix"/"input"/"output" inchangés d'un usage à l'autre,
# voir taip_parameter_reference.md). Seuls les champs de "caractère"
# (drive, glue, wear, noise, presence, hi_shape, lo_shape) sont pilotés par
# `decide_params`.
_DEFAULT_PARAMS: dict[str, float] = {
    "colorID": 1.0,
    "drive": 0.0,
    "glue": 0.0,
    "hi_shape": 0.0,
    "host_bypass": 0.0,
    "input": 0.0,
    "link": 1.0,
    "lo_shape": 0.0,
    "mix": 100.0,
    "model": 0.0,
    "noise": 0.0,
    "output": 0.0,
    "power": 1.0,
    "presence": 0.0,
    "wear": 0.0,
}

# Bornes raisonnables observées/déduites du seul preset réel dispo — évite
# de générer des valeurs aberrantes tant qu'on n'a pas plus de presets de
# référence pour confirmer les plages exactes de chaque contrôle.
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "drive": (0.0, 10.0),
    "glue": (0.0, 20.0),
    "wear": (0.0, 30.0),
    "noise": (0.0, 15.0),
    "presence": (0.0, 20.0),
    "hi_shape": (0.0, 10.0),
    "lo_shape": (0.0, 10.0),
}

_PARAM_RE = re.compile(r'<PARAM\s+id="([^"]+)"\s+value="([^"]+)"\s*/>')
_HEADER_RE = re.compile(r"<TAIP_1\s+([^>]*)>")


@dataclasses.dataclass
class TaipTemplate:
    """Structure fixe extraite d'un `.aupreset` TAIP réel, réutilisée telle
    quelle pour ne piloter QUE les paramètres de caractère — le reste
    (scale_fac_id, PHASE_OFFSET, l'enveloppe plist complète) est copié sans
    modification, exactement comme `_GLOBAL_TAIL_TEMPLATE`/`_METADATA_TAIL`
    dans `proq4_control.py`."""

    header_attrs: str  # ex: 'scale_fac_id="1.04..." PHASE_OFFSET="5.20..." PRESET_NAME="..."'
    full_state: dict  # plist complet du preset source (name/manufacturer/type/subtype/version/...)

    @classmethod
    def from_preset(cls, preset: PluginPreset) -> TaipTemplate:
        blob = preset.full_state.get("jucePluginState")
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError(f"{preset.source_path}: pas de jucePluginState exploitable (TAIP attendu).")
        xml_text = _extract_xml(bytes(blob))
        match = _HEADER_RE.search(xml_text)
        if not match:
            raise ValueError(f"{preset.source_path}: balise <TAIP_1 ...> introuvable dans le XML.")
        return cls(header_attrs=match.group(1).strip(), full_state=dict(preset.full_state))


def _extract_xml(blob: bytes) -> str:
    if not blob.startswith(_HEADER_MAGIC):
        raise ValueError("Blob jucePluginState : magic 'VC2!' absent, format inattendu.")
    (length,) = struct.unpack_from("<I", blob, 4)
    xml_bytes = blob[8 : 8 + length]
    return xml_bytes.decode("utf-8")


def parse_taip_params(preset: PluginPreset) -> dict[str, float]:
    """Lit les 15 `<PARAM id=... value=.../>` d'un .aupreset TAIP réel."""
    blob = preset.full_state.get("jucePluginState")
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError(f"{preset.source_path}: pas de jucePluginState exploitable.")
    xml_text = _extract_xml(bytes(blob))
    return {pid: float(val) for pid, val in _PARAM_RE.findall(xml_text)}


def _build_taip_blob(params: dict[str, float], header_attrs: str) -> bytes:
    param_tags = "".join(f'<PARAM id="{pid}" value="{val}"/>' for pid, val in params.items())
    xml_text = f'<?xml version="1.0" encoding="UTF-8"?> <TAIP_1 {header_attrs}>{param_tags}</TAIP_1> '
    xml_bytes = xml_text.encode("utf-8")
    header = _HEADER_MAGIC + struct.pack("<I", len(xml_bytes))
    return header + xml_bytes


def make_dynamic_preset(name: str, params: dict[str, float], template: TaipTemplate) -> PluginPreset:
    """Construit un PluginPreset TAIP prêt pour au_host.process_chain_offline,
    en réutilisant l'enveloppe plist + les attributs `<TAIP_1 ...>` d'un vrai
    preset (`template`) mais avec les 15 `<PARAM>` fournis par `params`."""
    full_params = {**_DEFAULT_PARAMS, **params}
    blob = _build_taip_blob(full_params, template.header_attrs)
    full_state = dict(template.full_state)
    full_state["jucePluginState"] = blob
    full_state["name"] = name
    return PluginPreset(
        name=name,
        source_path=Path("<dynamic>"),
        component_type=str(full_state.get("type", "")) or "aufx",
        component_subtype=str(full_state.get("subtype", "")) or "Taip",
        component_manufacturer=str(full_state.get("manufacturer", "")) or "BABA",
        full_state=full_state,
    )


def _clamp(value: float, name: str) -> float:
    lo, hi = _PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


def decide_params(analysis: FileAnalysis) -> dict[str, float]:
    """
    Traduit les features mesurées sur CE fichier (analysis.py) en réglages
    TAIP concrets. Corrective/légère par défaut — vise à ajouter du
    caractère analogique discret, pas à imposer un son de bande épais par
    défaut sur tout ce qui passe par la chaîne.

    Heuristique actuelle (ajustable, pas calibrée sur un corpus réel — voir
    la même réserve dans `proq4_control.decide_bands`) :
      - Base légère systématique : `drive=1.5`, `wear=4.0`, `noise=2.0`
        (grain de bande subtil, quasi inaudible seul).
      - `low_end_dominant` (tag analysis.py) : plus de `glue` (cohésion
        bus-style utile sur du matériel dominé par les basses/bass DI) et
        `lo_shape` légèrement relevé.
      - `kb_metallic_4k_elevated` / `kb_hf_fizz_14k_elevated` : `presence`
        réduite et `hi_shape` augmenté — la saturation de bande adoucit
        naturellement le buzz métallique/fizz HF caractéristique des
        artefacts IA (voir config/suno_artifacts_kb.md).
      - `crest_factor_db < 8` (déjà compressé) : `drive`/`wear` réduits pour
        ne pas cumuler une saturation supplémentaire sur du matériel déjà
        dense.
      - `clipping_detected` : `drive` réduit à 0 (ne pas ajouter de
        saturation sur une source déjà écrêtée, contre-productif).
    """
    tags = set(analysis.summary_tags())
    params: dict[str, float] = {
        "drive": 1.5,
        "wear": 4.0,
        "noise": 2.0,
        "glue": 0.0,
        "presence": 0.0,
        "hi_shape": 0.0,
        "lo_shape": 0.0,
    }

    if "low_end_dominant" in tags:
        params["glue"] = 8.0
        params["lo_shape"] = 3.0

    if "kb_metallic_4k_elevated" in tags or "kb_hf_fizz_14k_elevated" in tags:
        params["presence"] = -4.0 if "kb_metallic_4k_elevated" in tags else 0.0
        params["hi_shape"] = 5.0

    if analysis.dynamics.crest_factor_db < 8:
        params["drive"] = max(0.0, params["drive"] - 1.0)
        params["wear"] = max(0.0, params["wear"] - 2.0)

    if analysis.dynamics.clipping_ratio > 0.001:
        params["drive"] = 0.0

    # presence n'a de sens que positive dans le modèle TAIP (rehaussement) —
    # la "réduction" décidée ci-dessus pour le tag metallic est en réalité
    # une absence de rehaussement, jamais une valeur négative envoyée au plugin.
    params["presence"] = max(0.0, params["presence"])

    return {name: round(_clamp(value, name), 2) if name in _PARAM_BOUNDS else value for name, value in params.items()}
