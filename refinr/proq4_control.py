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
    0.0,
    1.0,
    9.966,
    0.0,
    0.5,
    0.0,
    2.0,
    2.0,
    1.0,
    0.0,
    1.0,
    1.0,
    0.667,
    50.0,
    50.0,
    0.0,
    0.0,
    3.322,
    14.288,
    0.0,
    0.0,
    50.0,
    0.0,
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
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    2.0,
    3.0,
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
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
).replace(
    b" ", b""
)  # sûreté si un espace traîne


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
    shape: str = "bell"  # clé de SHAPE_ENUM
    stereo: str = "stereo"  # clé de STEREO_ENUM
    slope_db_per_oct: float | None = None  # seulement pertinent pour high_pass/low_pass
    dynamic_range_db: float = 0.0  # 0.0 = pas de dynamique sur cette bande
    dynamic_auto_threshold: bool = False
    reason: str = ""  # pourquoi cette bande a été décidée (valeurs mesurées à l'appui)

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


def decide_bands(analysis: FileAnalysis, suno_mode: bool = False) -> list[Band]:
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

    `suno_mode=True` (opt-in explicite, JAMAIS activé automatiquement par
    l'analyse) ajoute deux corrections supplémentaires spécifiques aux
    artefacts connus des générateurs IA type Suno — voir
    `config/suno_artifacts_kb.md` pour les sources et le détail :
      - Shelf HF à 14kHz pour le bruit de synthèse/fizz caractéristique du
        codec de génération.
      - Bell à 4kHz pour le "buzz" métallique/robotique typique des
        formants vocaux synthétiques Suno (zone 3.5-5kHz rapportée).
    Ces deux corrections ne sont PAS déduites d'une mesure faite sur ce
    fichier précis (contrairement au reste de cette fonction) mais d'une
    connaissance a priori sur la source — d'où l'opt-in strict.
    """
    bands: list[Band] = []

    bands.append(
        Band(
            freq_hz=30.0,
            shape="high_pass",
            slope_db_per_oct=12.0,
            stereo="stereo",
            reason="Sécurité systématique : retire le sub-sonique (<30Hz, inaudible, jamais utile "
            "musicalement) pour ne pas gaspiller de headroom ni perturber le limiteur final.",
        )
    )

    tilt = analysis.spectral.tilt_db_per_octave
    if tilt > 1.0 or tilt < -2.5:
        shelf_gain = max(-4.0, min(4.0, -tilt * 1.2))
        direction = "trop brillante" if tilt > 1.0 else "trop sombre"
        bands.append(
            Band(
                freq_hz=6000.0,
                gain_db=round(shelf_gain, 2),
                shape="high_shelf",
                stereo="stereo",
                reason=(
                    f"Pente spectrale mesurée à {tilt:+.2f} dB/octave (seuils déclencheurs : "
                    f">+1.0 ou <-2.5) — source jugée {direction}. Correction : High Shelf 6kHz "
                    f"{shelf_gain:+.2f}dB (proportionnel au tilt mesuré, borné à ±4dB pour rester subtil "
                    f"et ne pas surcorriger)."
                ),
            )
        )

    clip_ratio = analysis.dynamics.clipping_ratio
    if clip_ratio > 0.001:
        bands.append(
            Band(
                freq_hz=3000.0,
                gain_db=0.0,
                q=1.5,
                shape="bell",
                stereo="stereo",
                dynamic_range_db=-2.0,
                dynamic_auto_threshold=True,
                reason=(
                    f"Écrêtage détecté sur {clip_ratio*100:.2f}% des échantillons (seuil >0.1%, proxy: "
                    f"échantillons à moins de 0.3dB de 0dBFS). La dureté/agressivité typique de ce type "
                    f"de source se concentre autour de 3kHz (zone de présence, où l'oreille est la plus "
                    f"sensible). Correction : dynamique légère (-2dB, seuil auto) plutôt qu'un cut statique, "
                    f"pour ne réagir que sur les passages réellement durs, sans assourdir le reste."
                ),
            )
        )

    crest = analysis.dynamics.crest_factor_db
    if crest < 8:
        bands.append(
            Band(
                freq_hz=300.0,
                gain_db=-1.5,
                q=1.2,
                shape="bell",
                stereo="stereo",
                reason=(
                    f"Crest factor mesuré à {crest:.1f}dB (seuil <8dB = déjà bien/trop compressé). "
                    f"Le matériel très compressé accumule typiquement de l'énergie autour de 300Hz "
                    f"(boxiness/mud dû à la compression qui égalise les dynamiques et laisse ressortir "
                    f"le bas-médium). Correction : léger creux Bell -1.5dB pour désencombrer."
                ),
            )
        )

    corr = analysis.dynamics.stereo_correlation
    if corr < 0.2:
        bands.append(
            Band(
                freq_hz=150.0,
                shape="high_pass",
                slope_db_per_oct=12.0,
                stereo="side",
                reason=(
                    f"Corrélation stéréo mesurée à {corr:.2f} (seuil <0.2 = stéréo très large, canaux "
                    f"quasi indépendants). Risque d'incompatibilité mono et de flou dans le bas du spectre "
                    f"si les basses fréquences sont présentes dans le canal Side. Correction : High Pass "
                    f"150Hz appliqué UNIQUEMENT au canal Side (le Mid n'est pas touché), pour garder les "
                    f"basses fréquences compatibles mono sans réduire la largeur stéréo perçue en aigu."
                ),
            )
        )

    if suno_mode:
        bands.append(
            Band(
                freq_hz=14000.0,
                gain_db=-2.5,
                shape="high_shelf",
                stereo="stereo",
                reason=(
                    "Mode Suno/IA activé (config/suno_artifacts_kb.md, section 2) : les générateurs "
                    "IA jettent le détail HF au-delà d'~14kHz et le remplacent par du bruit de "
                    "synthèse ('fizz'), source de fatigue d'écoute. Correction a priori (pas déduite "
                    "de l'analyse de ce fichier) : High Shelf 14kHz -2.5dB."
                ),
            )
        )
        bands.append(
            Band(
                freq_hz=4000.0,
                gain_db=-2.5,
                q=0.7,
                shape="bell",
                stereo="stereo",
                reason=(
                    "Mode Suno/IA activé (config/suno_artifacts_kb.md, section 3) : formants vocaux "
                    "synthétiques statiques dans la zone 3.5-5kHz, perçus comme un buzz "
                    "métallique/robotique typique de Suno. Correction a priori sur le mix entier "
                    "(pas de séparation vocale disponible) : Bell 4kHz -2.5dB, Q large (0.7) pour "
                    "rester modéré sur les autres éléments présents dans cette zone."
                ),
            )
        )

    return bands
