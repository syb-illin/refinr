"""
Contrôle ACTIF de la largeur stéréo — jusqu'ici, `analysis.py`
(`dynamics.stereo_correlation`, tags `wide_stereo`/`mono_leaning`) se contente
de FLAGGER un problème potentiel ; `proq4_control.decide_bands` réagit
seulement au cas extrême (corrélation <0.2 -> High Pass Side à 150Hz).
Ce module va plus loin : il resserre ou élargit réellement le canal Side,
en encodage Mid/Side standard, avec un crossover "mono-safe" — les basses
fréquences (<`MONO_SAFE_CROSSOVER_HZ`) ne sont JAMAIS resserrées ni
élargies, seulement le contenu au-dessus, pour ne jamais dégrader la
compatibilité mono (lecture téléphone/Bluetooth) quel que soit le facteur
appliqué en aigu.

Traitement volontairement simple et sûr (un seul facteur de largeur
large-bande au-dessus du crossover, pas de traitement multi-bande fin) :
corrige un problème mesuré sur CE fichier, ne cherche pas à "designer" un
son. Cohérent avec la philosophie du reste de la chaîne (proq4_control,
taip_control) : correctif, pas créatif.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .analysis import FileAnalysis
from .audio_io import AudioBuffer

# En dessous de cette fréquence, le canal Side est toujours laissé intact
# (facteur 1.0), quel que soit le facteur de largeur appliqué au-dessus —
# évite qu'un resserrement/élargissement HF ne s'accompagne d'un
# déséquilibre de phase basse fréquence, la zone la plus critique pour la
# compatibilité mono (voir aussi le futur mono fold-down check, backlog).
MONO_SAFE_CROSSOVER_HZ = 150.0

# Seuils de corrélation stéréo (dynamics.stereo_correlation, -1..1) —
# cohérents en ordre de grandeur avec proq4_control.decide_bands (qui
# déclenche son High Pass Side au même seuil bas de 0.2).
NARROW_CORRELATION_THRESHOLD = 0.2
WIDEN_CORRELATION_THRESHOLD = 0.9

# Facteurs de largeur appliqués au Side filtré (>150Hz) quand un seuil est
# franchi. Volontairement modérés (pas de resserrement total ni
# d'élargissement agressif) — un premier passage correctif, pas un design
# stéréo créatif.
NARROW_WIDTH_FACTOR = 0.6
WIDEN_WIDTH_FACTOR = 1.3

# Marge de sécurité anti-clip après traitement M/S (widening peut créer un
# pic plus haut que l'entrée sur les canaux L/R reconstruits).
_SAFETY_CEILING_LINEAR = 0.999


@dataclasses.dataclass
class StereoWidthDecision:
    width_factor: float  # 1.0 = pas de correction
    reason: str


def decide_width_factor(analysis: FileAnalysis) -> StereoWidthDecision:
    """
    Décide s'il faut resserrer ou élargir le Side à partir de la corrélation
    stéréo mesurée sur CE fichier (analysis.dynamics.stereo_correlation).
    Ne fait rien (facteur 1.0) dans la plage normale.
    """
    corr = analysis.dynamics.stereo_correlation
    if corr < NARROW_CORRELATION_THRESHOLD:
        return StereoWidthDecision(
            width_factor=NARROW_WIDTH_FACTOR,
            reason=(
                f"Corrélation stéréo mesurée à {corr:.2f} (seuil <{NARROW_CORRELATION_THRESHOLD} = stéréo "
                f"très large, canaux quasi indépendants). Correction : resserrement du canal Side au-dessus "
                f"de {MONO_SAFE_CROSSOVER_HZ:.0f}Hz (facteur {NARROW_WIDTH_FACTOR}), le grave reste intact "
                "(mono-safe)."
            ),
        )
    if corr > WIDEN_CORRELATION_THRESHOLD:
        return StereoWidthDecision(
            width_factor=WIDEN_WIDTH_FACTOR,
            reason=(
                f"Corrélation stéréo mesurée à {corr:.2f} (seuil >{WIDEN_CORRELATION_THRESHOLD} = quasi mono). "
                f"Correction : élargissement léger du canal Side au-dessus de {MONO_SAFE_CROSSOVER_HZ:.0f}Hz "
                f"(facteur {WIDEN_WIDTH_FACTOR}), le grave reste intact (mono-safe)."
            ),
        )
    return StereoWidthDecision(
        width_factor=1.0,
        reason=(
            f"Corrélation stéréo mesurée à {corr:.2f}, dans la plage normale "
            f"({NARROW_CORRELATION_THRESHOLD}-{WIDEN_CORRELATION_THRESHOLD}) — pas de correction de largeur."
        ),
    )


def apply_stereo_width(
    buffer: AudioBuffer,
    width_factor: float,
    mono_safe_hz: float = MONO_SAFE_CROSSOVER_HZ,
) -> AudioBuffer:
    """Applique `width_factor` au canal Side (encodage M/S), UNIQUEMENT
    au-dessus de `mono_safe_hz`. `width_factor == 1.0` ou source mono :
    retourne le buffer inchangé (pas de coût de traitement inutile)."""
    if buffer.n_channels < 2 or width_factor == 1.0:
        return buffer

    stereo = buffer.as_stereo()
    left, right = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    nyquist = buffer.sample_rate / 2.0
    safe_crossover = min(mono_safe_hz, nyquist * 0.9)
    sos = butter(4, safe_crossover, btype="highpass", fs=buffer.sample_rate, output="sos")
    side_high = sosfiltfilt(sos, side)
    side_low = side - side_high  # complément mono-safe, jamais retouché

    side_shaped = side_low + side_high * width_factor
    new_left = mid + side_shaped
    new_right = mid - side_shaped
    out = np.stack([new_left, new_right], axis=1)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > _SAFETY_CEILING_LINEAR:
        out = out * (_SAFETY_CEILING_LINEAR / peak)

    return AudioBuffer(
        samples=out.astype(np.float32),
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
    )
