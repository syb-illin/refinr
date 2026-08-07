"""
Types partagés pour représenter un preset AU (.aupreset) et son rôle dans la
chaîne de traitement. Ce module est volontairement indépendant de PyObjC :
il ne fait que lire des fichiers plist (stdlib `plistlib`), donc il tourne
aussi bien en dev/test (Linux) qu'en prod (macOS). Le hosting réel du
plugin (au_host.py) consomme ces objets.
"""

from __future__ import annotations

import dataclasses
import enum
import plistlib
import struct
from pathlib import Path


class PluginRole(str, enum.Enum):  # noqa: UP042 - StrEnum nécessite 3.11+, compat plus large
    EQ = "eq"
    SATURATION = "saturation"
    TAPE = "tape"
    LEVELING = "leveling"


def ostype_to_str(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, int):
        raw = struct.pack(">I", value & 0xFFFFFFFF)
    else:
        return str(value)
    try:
        s = raw.decode("mac_roman")
    except UnicodeDecodeError:
        return raw.hex()
    return s if s.isprintable() else raw.hex()


@dataclasses.dataclass
class PluginPreset:
    """Un .aupreset chargé, avec ses métadonnées d'identification AU."""

    name: str
    source_path: Path
    component_type: str  # OSType 4-char, ex "aufx"
    component_subtype: str  # identifie le plugin précis, ex "Fq4A"
    component_manufacturer: str  # ex "FabF"
    full_state: dict  # le plist complet, à passer tel quel à AUAudioUnit.fullState
    role: PluginRole | None = None
    tags: tuple[str, ...] = ()  # métadonnées de sélection (voir *.meta.yaml)
    intensity: str | None = None  # "light" | "medium" | "heavy" si renseigné


def load_aupreset(path: str | Path) -> PluginPreset:
    path = Path(path)
    with open(path, "rb") as fh:
        plist = plistlib.load(fh)

    if not isinstance(plist, dict):
        raise ValueError(f"{path}: plist racine inattendu ({type(plist)})")

    return PluginPreset(
        name=str(plist.get("name", path.stem)),
        source_path=path,
        component_type=ostype_to_str(plist.get("type", 0)),
        component_subtype=ostype_to_str(plist.get("subtype", 0)),
        component_manufacturer=ostype_to_str(plist.get("manufacturer", 0)),
        full_state=plist,
    )


@dataclasses.dataclass
class ChainStepReport:
    """Trace d'une étape de la chaîne, pour le rapport final par fichier."""

    role: str
    plugin_name: str
    preset_name: str
    preset_path: str
    reason: str  # pourquoi ce preset a été choisi pour ce fichier
    pre_measurement: dict
    post_measurement: dict
    extra: dict = dataclasses.field(default_factory=dict)
