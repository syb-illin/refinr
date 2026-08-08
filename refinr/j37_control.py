"""
Pilotage DYNAMIQUE de Waves Abbey Road J37 (rôle TAPE) — pendant de
`proq4_control.py` (EQ) et `taip_control.py` (SATURATION) : construit
programmatiquement l'état interne du plugin à partir de ce que `analysis.py`
mesure sur CE fichier, au lieu de sélectionner un `.aupreset` figé parmi
`preset_mapping.py` (toujours disponible en fallback, voir `chain.py`).

## Format et reverse-engineering

Contrairement à Pro-Q4 (struct binaire documenté champ par champ) et TAIP
(XML avec noms de paramètres explicites, aucun reverse-engineering
nécessaire), le J37 stocke ses réglages dans un tableau PLAT et NON DOCUMENTÉ
de ~195 floats (`<Parameters Type="RealWorld">`), enveloppé dans un blob
binaire `Waves_XPst` :

    uint32 BE: longueur totale du blob
    16 bytes: champs constants (version/compteurs internes) + "T37S" + "setA"
    uint32 BE: longueur du texte XML qui suit
    b"XPst"
    texte XML UTF-8 (<PresetChunkXMLTree>...</PresetChunkXMLTree>, contient
    DEUX blocs <PresetData Setup="SETUP_A"|"SETUP_B">, chacun avec son propre
    tableau RealWorld — seul SETUP_A est actif, voir <ActiveSetup> ; SETUP_B
    n'est jamais modifié par ce module)

Mapping confirmé par diff (voir `tools/diff_j37_presets.py` et
`config/presets/tape/j37_parameter_reference.md` pour le détail des preuves,
presets Logic Pro ne différant QUE par un seul contrôle GUI à la fois) :

  - SCALAIRES directement pilotables (le nombre stocké EST la valeur GUI) :
    Saturation (index 69, dupliqué à l'identique en 172), Noise Level
    (index 171 — confirmé comme un flag bas/haut plutôt que continu sur les
    échantillons disponibles), Wow Rate (173), Wow Depth (175), Flutter Rate
    (177), Flutter Depth (178).
  - Speed / Bias / Formula / Modeled Tracks déclenchent chacun un RECALCUL
    PHYSIQUE complet (30 à 69 indices corrélés changent à la fois quand on
    isole un seul de ces contrôles) — typique d'une simulation de bande
    analogique où vitesse/bias/type de bande sont physiquement liés. AUCUN
    mapping "un scalaire = un contrôle" n'existe pour ces quatre-là.

## Décision de conception : pourquoi Speed/Bias/Formula/Modeled Tracks restent FIXES

Ce module pilote UNIQUEMENT les scalaires confirmés ci-dessus. Speed/Bias/
Formula/Modeled Tracks restent figés sur le bloc de référence capturé (815 /
15ips / bias 0 / 2+3 tracks — réglage par défaut du plugin,
`config/presets/tape/j37_baseline_reference.aupreset`).

Piloter dynamiquement ces quatre-là nécessiterait soit un modèle physique
équivalent à celui de Waves (hors de portée), soit choisir parmi les blocs
capturés (811/815/855/888, 7.5/15ips, bias 0/3/5, 2/2+3/3 tracks — tous
disponibles dans `config/presets/tape/`, capturés pendant cette session de
reverse-engineering). Mais Waves ne documente pas publiquement ce que chaque
formule/vitesse apporte tonalement, et inventer une corrélation ("811 = plus
sombre", par exemple) sans preuve serait contraire à la philosophie
corrective du projet (voir `proq4_control.py`, `taip_control.py` : corriger
ce qui est mesuré, jamais imposer une couleur non justifiée). Les blocs
restent disponibles pour un futur raffinement si cette documentation existe
un jour ou si un diff plus poussé permet de les caractériser tonalement.
"""

from __future__ import annotations

import dataclasses
import re
import struct
from pathlib import Path

from .analysis import FileAnalysis
from .preset_types import PluginPreset

# --------------------------------------------------------------------------
# Indices confirmés dans le tableau RealWorld (195 floats plats) — voir
# config/presets/tape/j37_parameter_reference.md pour les preuves de diff.
# --------------------------------------------------------------------------

IDX_SATURATION = 69
IDX_SATURATION_MIRROR = 172  # dupliqué à l'identique dans tous les échantillons observés
IDX_NOISE = 171
IDX_WOW_RATE = 173
IDX_WOW_DEPTH = 175
IDX_FLUTTER_RATE = 177
IDX_FLUTTER_DEPTH = 178

# Valeurs confirmées par échantillon réel (voir j37_parameter_reference.md) :
# le contrôle Noise Level se comporte comme un flag bas/haut sur les
# échantillons disponibles ("off" et affichage "2" -> 2.0 ; affichages "100"
# et "240" -> 24.0 tous les deux), PAS une valeur continûment proportionnelle
# à l'affichage GUI — voir la note dans decide_params.
NOISE_OFF_VALUE = 2.0
NOISE_ON_VALUE = 24.0

# --------------------------------------------------------------------------
# Format binaire Waves_XPst — voir docstring du module.
# --------------------------------------------------------------------------

_XPST_HEADER_LEN = 28
_XPST_TAG = b"XPst"
_PARAMS_RE = re.compile(
    r'(<Parameters\s+Type="RealWorld"[^>]*>)(.*?)(</Parameters>)',
    re.DOTALL,
)
_FLOAT_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _format_value(value: float) -> str:
    """Représentation texte d'un float pour injection dans le tableau
    RealWorld — n'importe quelle représentation décimale valide convient
    (le parseur XML du plugin lit juste `float(token)`), `repr` donne la
    représentation la plus courte qui round-trip exactement."""
    return repr(float(value))


@dataclasses.dataclass
class J37Template:
    """
    Structure fixe extraite d'un `.aupreset` J37 réel, réutilisée telle
    quelle pour ne piloter QUE les scalaires confirmés — le reste (header
    binaire, SETUP_B, métadonnées XML, `*` de padding) est copié SANS
    modification, même principe que `_GLOBAL_TAIL_TEMPLATE` dans
    `proq4_control.py` ou `TaipTemplate` dans `taip_control.py`.
    """

    xml_text: str  # texte XML complet (les deux <PresetData Setup=...>)
    realworld_span: tuple[int, int]  # offsets (dans xml_text) du contenu interne du bloc RealWorld de SETUP_A
    header_middle: bytes  # blob[4:20] : champs constants (version/compteurs + "T37S" + "setA"), copiés verbatim
    full_state: dict  # plist complet du preset source (Waves_XPst sera remplacé, le reste copié tel quel)

    @classmethod
    def from_preset(cls, preset: PluginPreset) -> J37Template:
        blob = preset.full_state.get("Waves_XPst")
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError(f"{preset.source_path}: pas de clé 'Waves_XPst' exploitable (J37 attendu).")
        blob = bytes(blob)
        (xml_len,) = struct.unpack_from(">I", blob, 20)
        xml_bytes = blob[_XPST_HEADER_LEN : _XPST_HEADER_LEN + xml_len]
        xml_text = xml_bytes.decode("utf-8")

        match = _PARAMS_RE.search(xml_text)
        if not match:
            raise ValueError(f'{preset.source_path}: bloc <Parameters Type="RealWorld"> (SETUP_A) introuvable.')

        return cls(
            xml_text=xml_text,
            realworld_span=(match.start(2), match.end(2)),
            header_middle=blob[4:20],
            full_state=dict(preset.full_state),
        )


def _read_realworld_tokens(text: str) -> list[re.Match]:
    return list(_FLOAT_TOKEN_RE.finditer(text))


def parse_realworld_values(preset: PluginPreset) -> list[float]:
    """Lit le tableau plat de floats RealWorld (SETUP_A) d'un .aupreset J37
    réel — utilisé pour les tests et le debug, même rôle que
    `taip_control.parse_taip_params`."""
    template = J37Template.from_preset(preset)
    start, end = template.realworld_span
    inner = template.xml_text[start:end]
    return [float(m.group(0)) for m in _read_realworld_tokens(inner)]


@dataclasses.dataclass
class J37Params:
    """Réglages scalaires pilotés dynamiquement (voir docstring du module
    pour pourquoi Speed/Bias/Formula/Modeled Tracks n'en font pas partie)."""

    saturation: float = 0.0  # 0-100, valeur GUI directe (voir IDX_SATURATION)
    noise_on: bool = False  # voir NOISE_OFF_VALUE/NOISE_ON_VALUE
    wow_rate: float = 0.0
    wow_depth: float = 0.0
    flutter_rate: float = 0.0
    flutter_depth: float = 0.0


def make_dynamic_preset(name: str, params: J37Params, template: J37Template) -> PluginPreset:
    """
    Construit un PluginPreset J37 prêt pour `au_host.process_chain_offline`,
    en réutilisant l'enveloppe XML + le header binaire d'un vrai preset
    (`template`) mais avec les scalaires confirmés remplacés par `params`.

    Remplacement PAR SUBSTITUTION DE SPANS DE TEXTE (pas de re-génération du
    XML depuis une structure de données) : le tableau RealWorld contient des
    `*` de padding et des nombres à précision flottante variable qu'un
    simple `join(" ")` ne reproduirait pas à l'identique — on modifie donc
    UNIQUEMENT les tokens numériques ciblés, tout le reste (espaces, retours
    à la ligne, `*`, SETUP_B, métadonnées) reste strictement inchangé.
    """
    start, end = template.realworld_span
    inner = template.xml_text[start:end]
    tokens = _read_realworld_tokens(inner)

    overrides: dict[int, float] = {
        IDX_SATURATION: params.saturation,
        IDX_SATURATION_MIRROR: params.saturation,
        IDX_NOISE: NOISE_ON_VALUE if params.noise_on else NOISE_OFF_VALUE,
        IDX_WOW_RATE: params.wow_rate,
        IDX_WOW_DEPTH: params.wow_depth,
        IDX_FLUTTER_RATE: params.flutter_rate,
        IDX_FLUTTER_DEPTH: params.flutter_depth,
    }

    max_index = max(overrides)
    if len(tokens) <= max_index:
        raise ValueError(
            f"Tableau RealWorld du template trop court ({len(tokens)} tokens, "
            f"index {max_index} requis) — template J37 inattendu."
        )

    # Reconstruction par découpage : on avance span par span dans le texte
    # original, en substituant uniquement les tokens dont l'index est ciblé.
    pieces: list[str] = []
    cursor = 0
    for idx, m in enumerate(tokens):
        pieces.append(inner[cursor : m.start()])
        if idx in overrides:
            pieces.append(_format_value(overrides[idx]))
        else:
            pieces.append(m.group(0))
        cursor = m.end()
    pieces.append(inner[cursor:])
    new_inner = "".join(pieces)

    new_xml_text = template.xml_text[:start] + new_inner + template.xml_text[end:]
    new_xml_bytes = new_xml_text.encode("utf-8")

    new_blob = (
        struct.pack(">I", _XPST_HEADER_LEN + len(new_xml_bytes))
        + template.header_middle
        + struct.pack(">I", len(new_xml_bytes))
        + _XPST_TAG
        + new_xml_bytes
    )

    full_state = dict(template.full_state)
    full_state["Waves_XPst"] = new_blob
    full_state["name"] = name

    return PluginPreset(
        name=name,
        source_path=Path("<dynamic>"),
        component_type=str(full_state.get("type", "")) or "aufx",
        component_subtype=str(full_state.get("subtype", "")) or "T37S",
        component_manufacturer=str(full_state.get("manufacturer", "")) or "Wave",
        full_state=full_state,
    )


# --------------------------------------------------------------------------
# Décision des réglages à partir de l'analyse du WAV
# --------------------------------------------------------------------------

# Base légère systématique (même esprit que taip_control._DEFAULT_PARAMS,
# un grain de bande discret plutôt qu'un caractère marqué par défaut).
DEFAULT_SATURATION = 8.0

# Mêmes seuils que proq4_control.decide_bands / transient_shaping (cohérence
# de projet) : matériel déjà compressé ou déjà écrêté ne doit pas recevoir
# davantage de saturation.
CREST_FACTOR_COMPRESSED_DB = 8.0
SATURATION_REDUCTION_ON_COMPRESSED = 3.0


def decide_params(analysis: FileAnalysis) -> J37Params:
    """
    Traduit les features mesurées sur CE fichier en réglages J37 concrets.
    Corrective, pas créative :
      - Base légère systématique de Saturation (grain de bande discret),
        réduite si le matériel est déjà bien compressé (crest factor bas,
        même heuristique que `proq4_control`/`taip_control`), coupée à zéro
        si de l'écrêtage est détecté dans la source (ajouter de la
        saturation sur une source déjà écrêtée est contre-productif).
      - Noise Level : activé UNIQUEMENT si une bande KB de fizz HF est
        mesurée comme élevée (`kb_hf_fizz_14k_elevated`, voir
        config/suno_artifacts_kb.md) — un soupçon de bruit de bande peut
        aider à masquer le bruit de synthèse HF typique des générateurs IA,
        un usage corrective plausible. Off par défaut.
      - Wow/Flutter : JAMAIS activés automatiquement. Introduire une
        modulation de hauteur est un choix de COULEUR, pas une correction —
        aucune mesure de `analysis.py` ne justifie de l'imposer à un fichier
        précis. Les champs restent pilotables (voir `J37Params`) pour un
        futur mode créatif explicite, mais `decide_params` les laisse
        toujours à 0.0.
    """
    saturation = DEFAULT_SATURATION

    if analysis.dynamics.crest_factor_db < CREST_FACTOR_COMPRESSED_DB:
        saturation = max(0.0, saturation - SATURATION_REDUCTION_ON_COMPRESSED)

    if analysis.dynamics.clipping_ratio > 0.001:
        saturation = 0.0

    noise_on = "kb_hf_fizz_14k_elevated" in analysis.summary_tags()

    return J37Params(saturation=round(saturation, 2), noise_on=noise_on)
