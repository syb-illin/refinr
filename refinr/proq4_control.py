"""
Pilotage DYNAMIQUE de FabFilter Pro-Q4 — pas de sélection parmi des presets
figés, mais construction programmatique de l'état binaire du plugin
(FabFilterPluginState) à partir de valeurs concrètes (fréquence, gain, Q,
forme, pente, placement stéréo, dynamique) décidées par `analysis.py` pour
CE fichier précis.

Tout l'encodage ci-dessous vient de `config/plugin_properties_mapping.yaml`,
reverse-engineered et confirmé par comparaison de presets de contrôle réels
(voir ce fichier pour le détail des preuves, champ par champ). Ce module ne
touche PAS à `preset_mapping.py` (sélection de presets par tags, toujours
disponible en fallback) — c'est une voie parallèle, plus fine.

Portable (aucune dépendance macOS) : ne fait que construire un dict Python +
des bytes, exactement le même format que ce que `preset_types.load_aupreset`
produit en lisant un vrai fichier .aupreset. Le résultat se branche
directement sur `au_host.process_chain_offline` (macOS uniquement, comme
d'habitude).
"""

from __future__ import annotations

import dataclasses
import math
import struct

from .analysis import FileAnalysis
from .preset_types import PluginPreset

# --------------------------------------------------------------------------
# Constantes d'encodage confirmées (voir config/plugin_properties_mapping.yaml)
# --------------------------------------------------------------------------

SHAPE_ENUM = {
    "bell": 0.0,
    "low_shelf": 1.0,
    "high_pass": 2.0,
    "high_shelf": 3.0,
    "low_pass": 4.0,
    "notch": 5.0,
    "band_pass": 6.0,
    "tilt_shelf": 7.0,
    "flat_tilt": 8.0,
}

STEREO_ENUM = {
    "left": 0.0,
    "right": 1.0,
    "stereo": 2.0,
    "mid": 3.0,
    "side": 4.0,
}

BAND_FLOATS = 23
N_BANDS = 24

# Bande "désactivée" par défaut (capturée sur une vraie bande inutilisée de
# Pro-Q4). On part de ce template pour chaque bande, actives ou non — seuls
# les champs confirmés sont modifiés pour les bandes actives.
_DISABLED_BAND_TEMPLATE = (
    0.0, 1.0, 9.966, 0.0, 0.5, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 1.0,
    0.667, 50.0, 50.0, 0.0, 0.0, 3.322, 14.288, 0.0, 0.0, 50.0, 0.0,
)

# Valeurs magiques observées systématiquement aux offsets 17/18 sur TOUTE
# bande active de notre échantillon, indépendamment de ses autres réglages
# (fréquence, gain, forme...). Rôle exact non identifié (possiblement lié au
# rendu graphique de la courbe EQ) mais copié tel quel par prudence — voir
# plugin_properties_mapping.yaml, champs 17/18.
_ACTIVE_MAGIC_17 = 6.644
_ACTIVE_MAGIC_18 = 11.551

# Queue de 48 floats globaux (indices absolus 552-599), capturée sur un vrai
# preset. Seul l'offset 0 (Natural Phase) est confirmé et piloté ici ; le
# reste est copié tel quel (voir plugin_properties_mapping.yaml, section
# global_params).
_GLOBAL_TAIL_TEMPLATE = (
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0,
    2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

# Métadonnées cosmétiques (nom/auteur/description/tags) qui suivent les 600
# floats dans le blob réel. N'affectent PAS le traitement audio (juste
# l'affichage dans l'UI Pro-Q4) — on les fige une fois pour toutes plutôt que
# de les reconstruire dynamiquement.
_METADATA_TAIL = bytes.fromhex(
    "4651347003000000030000006d6964ffffffff0100000006000000496e737420"
    "310100000043755356010000000400000006000000415554484f52090000004"
    "6616246696c7465720b0000004445534352495054494f4ea90000005468697320"
    "70726573657420697320206c6f61646564207768656e20796f75206f70656e20"
    "61206e657720696e7374616e63652e0a0a596f752063616e20637573746f6d69"
    "7a65207468652044656661756c742053657474696e67207072657365742061732"
    "0796f75206c696b652c20616e642073617665206974207669612074686520707"
    "26573657420"
).replace(b" ", b"")  # sûreté si un espace traîne


def _q_to_raw(q: float) -> float:
    """Q -> valeur brute Pro-Q4 (offset 4). Formule confirmée sur bande Bell."""
    return 0.5 + 0.094 * math.log2(q)


def _hz_to_raw(hz: float) -> float:
    return math.log2(hz)


@dataclasses.dataclass
class Band:
    """Une bande EQ à piloter dynamiquement, en unités humaines."""

    freq_hz: float
    gain_db: float = 0.0
    q: float = 1.0
    shape: str = "bell"                 # clé de SHAPE_ENUM
    stereo: str = "stereo"              # clé de STEREO_ENUM
    slope_db_per_oct: float | None = None   # seulement pertinent pour high_pass/low_pass
    dynamic_range_db: float = 0.0       # 0.0 = pas de dynamique sur cette bande
    dynamic_auto_threshold: bool = False

    def to_floats(self) -> tuple[float, ...]:
        block = list(_DISABLED_BAND_TEMPLATE)
        block[0] = 1.0  # band_enabled
        block[2] = _hz_to_raw(self.freq_hz)
        block[3] = self.gain_db
        block[4] = _q_to_raw(max(self.q, 0.01))
        block[5] = SHAPE_ENUM[self.shape]
        if self.slope_db_per_oct is not None:
            block[6] = self.slope_db_per_oct / 6.0
        block[7] = STEREO_ENUM[self.stereo]
        block[9] = self.dynamic_range_db
        block[11] = 0.0 if self.dynamic_auto_threshold else 1.0
        block[12] = 1.0
        block[17] = _ACTIVE_MAGIC_17
        block[18] = _ACTIVE_MAGIC_18
        return tuple(block)


def build_proq4_state(bands: list[Band], natural_phase: bool = False) -> bytes:
    """Construit le blob binaire FabFilterPluginState complet (600 floats + métadonnées)."""
    if len(bands) > N_BANDS:
        raise ValueError(f"Pro-Q4 supporte au maximum {N_BANDS} bandes, {len(bands)} demandées.")

    floats: list[float] = []
    for i in range(N_BANDS):
        if i < len(bands):
            floats.extend(bands[i].to_floats())
        else:
            floats.extend(_DISABLED_BAND_TEMPLATE)

    tail = list(_GLOBAL_TAIL_TEMPLATE)
    tail[0] = 1.0 if natural_phase else 0.0
    floats.extend(tail)

    assert len(floats) == 600, f"attendu 600 floats, obtenu {len(floats)}"

    header = b"FFBS" + struct.pack("<I", 1) + struct.pack("<I", len(floats))
    float_bytes = struct.pack(f"<{len(floats)}f", *floats)
    return header + float_bytes + _METADATA_TAIL


def make_dynamic_preset(name: str, bands: list[Band], natural_phase: bool = False) -> PluginPreset:
    """Construit un PluginPreset Pro-Q4 prêt pour au_host.process_chain_offline,
    SANS passer par un fichier .aupreset — piloté entièrement par `bands`."""
    from pathlib import Path

    state_blob = build_proq4_state(bands, natural_phase=natural_phase)
    full_state = {
        "name": name,
        "manufacturer": struct.unpack(">I", b"FabF")[0],
        "type": struct.unpack(">I", b"aumf")[0],
        "subtype": struct.unpack(">I", b"FQ4p")[0],
        "version": 0,
        "FabFilterPluginState": state_blob,
    }
    return PluginPreset(
        name=name,
        source_path=Path("<dynamic>"),
        component_type="aumf",
        component_subtype="FQ4p",
        component_manufacturer="FabF",
        full_state=full_state,
    )


# --------------------------------------------------------------------------
# Décision des réglages à partir de l'analyse du WAV
# --------------------------------------------------------------------------


def decide_bands(analysis: FileAnalysis) -> list[Band]:
    """
    Traduit les features mesurées sur CE fichier (analysis.py) en réglages
    Pro-Q4 concrets. Corrective, pas créative : vise à compenser ce que
    l'analyse détecte, pas à imposer une couleur.

    Heuristique actuelle (ajustable) :
      - Toujours : High Pass de sécurité à 30Hz/12dB-oct (retire le sub-sonique,
        n'affecte pas le contenu musical utile).
      - `tilt_db_per_octave` (pente spectrale globale) : correction via un
        High Shelf à 6kHz, gain inversement proportionnel au tilt mesuré,
        borné à ±4dB pour rester subtil (voir tags 'bright'/'dark').
      - `clipping_detected` : dynamique légère sur un Bell centré à 3kHz
        (dureté typique de sources écrêtées/saturées), range -2dB, seuil auto.
      - `already_compressed` : léger creux Bell à 300Hz (désencombrement
        bas-médium, fréquent sur du matériel déjà bien compressé).
      - `wide_stereo` : High Pass sur le canal Side à 150Hz/12dB-oct (garde
        les basses fréquences compatibles mono, évite les problèmes de phase
        en bas du spectre sur les sources très larges).
    """
    bands: list[Band] = []

    bands.append(Band(freq_hz=30.0, shape="high_pass", slope_db_per_oct=12.0, stereo="stereo"))

    tilt = analysis.spectral.tilt_db_per_octave
    if tilt > 1.0 or tilt < -2.5:
        shelf_gain = max(-4.0, min(4.0, -tilt * 1.2))
        bands.append(Band(freq_hz=6000.0, gain_db=round(shelf_gain, 2), shape="high_shelf", stereo="stereo"))

    if analysis.dynamics.clipping_ratio > 0.001:
        bands.append(
            Band(
                freq_hz=3000.0, gain_db=0.0, q=1.5, shape="bell", stereo="stereo",
                dynamic_range_db=-2.0, dynamic_auto_threshold=True,
            )
        )

    if analysis.dynamics.crest_factor_db < 8:
        bands.append(Band(freq_hz=300.0, gain_db=-1.5, q=1.2, shape="bell", stereo="stereo"))

    if analysis.dynamics.stereo_correlation < 0.2:
        bands.append(Band(freq_hz=150.0, shape="high_pass", slope_db_per_oct=12.0, stereo="side"))

    return bands
