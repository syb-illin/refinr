"""
Transient designer — nouveau stage de la chaîne, absent jusqu'ici
(chaîne actuelle : EQ -> saturation -> tape -> leveling, rien ne traite
explicitement l'attaque/le punch). Les générateurs IA type Suno produisent
parfois des attaques "molles"/lissées (artefact connu) ; ce module les
renforce (ou, en principe, les atténue — `sustain_amount_db` existe pour
ça mais n'est pas piloté automatiquement pour l'instant, seul l'attack
boost l'est).

Détecteur de transitoire classique à double enveloppe :
  - enveloppe "rapide" (attaque très courte, relâchement court) : suit
    fidèlement les pics/attaques.
  - enveloppe "lente" (attaque et relâchement plus longs) : suit le corps/
    sustain du son, ignore les pics brefs.
  - détecteur = écart en dB entre les deux ; positif = transitoire présent.

Le gain appliqué (`attack_amount_db` à l'endroit où le détecteur est actif)
est lié entre canaux (calculé sur la somme mono, comme le limiteur de
`loudness.py`) pour ne jamais déséquilibrer l'image stéréo.

Traitement offline uniquement (boucle Python par blocs) : Refinr traite des
fichiers entiers en batch, jamais en temps réel, donc le coût par bloc
(quelques centaines d'échantillons) reste largement acceptable pour un
morceau de quelques minutes.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .analysis import FileAnalysis
from .audio_io import AudioBuffer

# Constantes de l'enveloppe rapide (suit les attaques) et lente (suit le
# corps du son). Valeurs typiques d'un transient designer logiciel standard.
FAST_ATTACK_MS = 1.0
FAST_RELEASE_MS = 10.0
SLOW_ATTACK_MS = 30.0
SLOW_RELEASE_MS = 150.0

# Lissage supplémentaire du gain de sortie (évite le "zipper noise" d'un
# gain qui varie trop brutalement échantillon par échantillon).
GAIN_SMOOTH_MS = 3.0

# Le détecteur (écart fast/slow en dB) est normalisé sur cette plage avant
# d'être multiplié par `attack_amount_db` — au-delà, on considère que le
# transitoire est déjà "plein" (gain maximal atteint).
DETECTOR_FLOOR_DB = 0.0
DETECTOR_CEILING_DB = 8.0

_EPS = 1e-9

# --------------------------------------------------------------------------
# Décision automatique (analysis.py + activité transitoire mesurée ici)
# --------------------------------------------------------------------------

# Crest factor en dessous duquel le matériel est jugé "déjà bien compressé"
# (même seuil que proq4_control.decide_bands, cohérence de projet).
CREST_FACTOR_COMPRESSED_DB = 8.0

# Fraction de temps (0..1) où le détecteur est actif (>2dB) en dessous de
# laquelle on juge les transitoires "molles/lissées" — pas calibré sur un
# corpus réel, ajustable comme les autres seuils heuristiques du projet.
ACTIVITY_DB_THRESHOLD = 2.0
LOOSE_TRANSIENTS_ACTIVITY_MAX = 0.12

TRANSIENT_BOOST_DB = 3.0


def _envelope_follower(rectified: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Enveloppe à attaque/relâchement asymétriques, calculée par blocs
    (bloc = plusieurs échantillons vectorisés, boucle Python seulement sur
    les blocs) pour rester praticable sur un morceau entier."""
    n = rectified.shape[0]
    if n == 0:
        return rectified.copy()

    attack_coeff = float(np.exp(-1.0 / max(1.0, attack_ms * 1e-3 * sample_rate)))
    release_coeff = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * sample_rate)))

    block = 256
    env = np.empty(n, dtype=np.float64)
    current = 0.0
    for start in range(0, n, block):
        chunk = rectified[start : start + block]
        for i, x in enumerate(chunk):
            coeff = attack_coeff if x > current else release_coeff
            current = coeff * current + (1.0 - coeff) * x
            env[start + i] = current
    return env


def _mono_sum(buffer: AudioBuffer) -> np.ndarray:
    stereo = buffer.as_stereo() if buffer.n_channels > 1 else buffer.samples[:, None]
    return stereo.mean(axis=1).astype(np.float64)


def _detector_db(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    rectified = np.abs(mono)
    fast_env = _envelope_follower(rectified, sample_rate, FAST_ATTACK_MS, FAST_RELEASE_MS)
    slow_env = _envelope_follower(rectified, sample_rate, SLOW_ATTACK_MS, SLOW_RELEASE_MS)
    return 20.0 * np.log10((fast_env + _EPS) / (slow_env + _EPS))


def measure_transient_activity(buffer: AudioBuffer) -> float:
    """Fraction de temps (0..1) où le détecteur fast/slow dépasse
    `ACTIVITY_DB_THRESHOLD` — proxy de "combien ce fichier a déjà d'attaques
    marquées". Bas = transitoires molles/lissées (candidat au renforcement)."""
    mono = _mono_sum(buffer)
    if mono.size == 0:
        return 0.0
    detector = _detector_db(mono, buffer.sample_rate)
    return float(np.mean(detector > ACTIVITY_DB_THRESHOLD))


@dataclasses.dataclass
class TransientShapingDecision:
    attack_amount_db: float  # 0.0 = pas de correction
    reason: str
    transient_activity: float


def decide_attack_amount_db(buffer: AudioBuffer, analysis: FileAnalysis) -> TransientShapingDecision:
    """
    Corrective, pas créative : ne renforce l'attaque QUE si le matériel est
    déjà compressé (crest factor bas, comme le désencombrement 300Hz de
    `proq4_control.decide_bands`) ET que les transitoires mesurées sur CE
    fichier sont effectivement molles (activité transitoire basse) — les
    deux conditions ensemble, pas l'une ou l'autre, pour éviter de renforcer
    l'attaque d'un matériel déjà punchy juste parce qu'il est compressé
    (ex: EDM à la limite déjà bien transitoire).
    """
    activity = measure_transient_activity(buffer)
    crest = analysis.dynamics.crest_factor_db

    if crest < CREST_FACTOR_COMPRESSED_DB and activity < LOOSE_TRANSIENTS_ACTIVITY_MAX:
        return TransientShapingDecision(
            attack_amount_db=TRANSIENT_BOOST_DB,
            reason=(
                f"Crest factor {crest:.1f}dB (<{CREST_FACTOR_COMPRESSED_DB}dB, déjà compressé) ET activité "
                f"transitoire mesurée à {activity*100:.1f}% (<{LOOSE_TRANSIENTS_ACTIVITY_MAX*100:.0f}%, "
                f"attaques molles/lissées, artefact IA connu). Correction : transient designer, "
                f"+{TRANSIENT_BOOST_DB:.1f}dB sur l'attaque (gain lié stéréo)."
            ),
            transient_activity=activity,
        )
    return TransientShapingDecision(
        attack_amount_db=0.0,
        reason=(
            f"Crest factor {crest:.1f}dB, activité transitoire mesurée à {activity*100:.1f}% — "
            "hors des conditions de correction (voir decide_attack_amount_db), pas de transient shaping."
        ),
        transient_activity=activity,
    )


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def apply_transient_shaping(
    buffer: AudioBuffer,
    attack_amount_db: float,
    sustain_amount_db: float = 0.0,
) -> AudioBuffer:
    """Applique un gain lié stéréo dérivé du détecteur fast/slow (calculé sur
    la somme mono) : `attack_amount_db` sur les zones transitoires détectées,
    `sustain_amount_db` (optionnel, pas piloté automatiquement pour
    l'instant) sur le reste."""
    if attack_amount_db == 0.0 and sustain_amount_db == 0.0:
        return buffer

    mono = _mono_sum(buffer)
    if mono.size == 0:
        return buffer

    detector = _detector_db(mono, buffer.sample_rate)
    normalized = np.clip((detector - DETECTOR_FLOOR_DB) / (DETECTOR_CEILING_DB - DETECTOR_FLOOR_DB), 0.0, 1.0)

    gain_db = normalized * attack_amount_db + (1.0 - normalized) * sustain_amount_db

    # Lissage du gain pour éviter le zipper noise (même principe que le
    # release smoothing du limiteur dans loudness.py).
    smooth_coeff = float(np.exp(-1.0 / max(1.0, GAIN_SMOOTH_MS * 1e-3 * buffer.sample_rate)))
    smoothed = np.empty_like(gain_db)
    current = 0.0
    for i, g in enumerate(gain_db):
        current = smooth_coeff * current + (1.0 - smooth_coeff) * g
        smoothed[i] = current

    gain_linear = (10.0 ** (smoothed / 20.0))[:, None]

    stereo = buffer.as_stereo() if buffer.n_channels > 1 else buffer.samples[:, None]
    out = stereo.astype(np.float64) * gain_linear
    out_samples = out[:, 0].astype(np.float32) if buffer.n_channels == 1 else out.astype(np.float32)

    return AudioBuffer(samples=out_samples, sample_rate=buffer.sample_rate, source_path=buffer.source_path)
