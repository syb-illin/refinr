"""
Correction MACRO-dynamique, pilotée par la Loudness Range (LRA) — pas
seulement l'intégré. Jusqu'ici le seul filet de sécurité dynamique en fin
de chaîne était `loudness.limit_true_peak` : un limiteur "brickwall" à
lookahead de quelques millisecondes, qui ne réagit qu'aux pics individuels.
Rien ne traitait les écarts de niveau ENTRE SECTIONS d'un même morceau
(couplet calme vs refrain fort, par ex.) — un fichier pouvait être
parfaitement conforme en LUFS intégré tout en ayant des sauts de volume
section à section gênants à l'écoute, invisibles pour le gate QC actuel
(qui ne regarde que l'intégré + le true peak).

Ce module ajoute le stage manquant : un "rider" de gain macro, à constante
de temps LENTE (secondes, pas millisecondes), piloté par la courbe de
loudness court-terme (fenêtre glissante ~3s — même fenêtre que
`loudness.measure_loudness_range`, pour rester cohérent avec la mesure de
LRA déjà utilisée ailleurs dans le projet). Ce n'est PAS un limiteur et PAS
un compresseur à l'échantillon : un cousin lent du limiteur existant, qui
tire doucement les sections trop écartées de la loudness intégrée globale
vers elle.

Volontairement PARTIEL : `ratio` < 1.0 réduit le LRA mesuré, ne l'aplatit
jamais à zéro — un master doit garder de la dynamique musicale. L'objectif
est de retirer les sauts vraiment GÊNANTS (au-delà du seuil de
déclenchement), pas de niveler tout le morceau au même volume perçu.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyloudnorm as pyln

from .analysis import FileAnalysis
from .audio_io import AudioBuffer

# Fenêtre/hop de la courbe de loudness court-terme — mêmes valeurs que
# `loudness.measure_loudness_range` (cohérence : c'est cette même mesure de
# LRA qui pilote la décision de déclenchement ci-dessous).
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.0

# Seuil de LRA (LU) au-delà duquel les écarts section à section sont jugés
# "gênants" pour un master de diffusion streaming (pas calibré sur un corpus
# réel — même réserve que les autres seuils heuristiques du projet ; une LRA
# élevée est parfaitement légitime pour du classique/jazz, ce seuil vise le
# cas Suno/pop/électro que ce projet cible).
LRA_JARRING_THRESHOLD_LU = 10.0

# Fraction de l'écart à la loudness intégrée corrigée par section (0.0 = pas
# de correction, 1.0 = ramène chaque section exactement à l'intégré). 0.5 =
# "colle" les sections sans les aplatir complètement.
DEFAULT_CORRECTION_RATIO = 0.5

# Plafond de correction par section, pour éviter un "pompage" audible sur
# une section anormalement calme/forte isolée (ex: silence de 2 secondes).
MAX_CORRECTION_DB = 6.0

# Constante de temps du lissage du gain (secondes) — volontairement lente
# (macro), pour ne jamais réagir comme un compresseur à l'échantillon.
GAIN_SMOOTHING_SECONDS = 1.5

_EPS = 1e-9


@dataclasses.dataclass
class MacroCompressionDecision:
    enabled: bool
    ratio: float
    reason: str


def decide_macro_compression(
    analysis: FileAnalysis, ratio: float = DEFAULT_CORRECTION_RATIO
) -> MacroCompressionDecision:
    """Corrective : ne se déclenche QUE si la LRA mesurée sur CE fichier
    dépasse `LRA_JARRING_THRESHOLD_LU` (voir analysis.py, `dynamics.loudness_range_lu`,
    déjà calculé pour tous les fichiers)."""
    lra = analysis.dynamics.loudness_range_lu
    if lra is None:
        return MacroCompressionDecision(
            enabled=False,
            ratio=0.0,
            reason="LRA non mesurable sur ce fichier (signal trop court ou trop silencieux) — pas de correction macro-dynamique.",
        )
    if lra > LRA_JARRING_THRESHOLD_LU:
        return MacroCompressionDecision(
            enabled=True,
            ratio=ratio,
            reason=(
                f"LRA mesurée à {lra:.1f}LU (seuil >{LRA_JARRING_THRESHOLD_LU:.0f}LU = sauts de volume "
                f"section à section jugés gênants pour un master de diffusion). Correction : rider de gain "
                f"macro (constante de temps {GAIN_SMOOTHING_SECONDS:.1f}s), ratio {ratio:.2f} — réduit l'écart "
                "sans aplatir la dynamique musicale."
            ),
        )
    return MacroCompressionDecision(
        enabled=False,
        ratio=0.0,
        reason=f"LRA mesurée à {lra:.1f}LU, dans la plage normale (<= {LRA_JARRING_THRESHOLD_LU:.0f}LU) — pas de correction macro-dynamique.",
    )


def compute_short_term_loudness_curve(
    buffer: AudioBuffer,
    window_s: float = WINDOW_SECONDS,
    hop_s: float = HOP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Courbe de loudness court-terme (K-weighted, ITU-R BS.1770 via
    pyloudnorm, même pondération que le reste du projet) : fenêtres de
    `window_s` glissant par pas de `hop_s`, centre de fenêtre en secondes.
    Retourne (times_sec, lufs_values) — les fenêtres non mesurables
    (silence complet) sont omises."""
    stereo = buffer.as_stereo() if buffer.n_channels > 1 else buffer.samples
    win = int(window_s * buffer.sample_rate)
    hop = int(hop_s * buffer.sample_rate)
    if win <= 0 or stereo.shape[0] < win:
        return np.array([]), np.array([])

    meter = pyln.Meter(buffer.sample_rate)
    times: list[float] = []
    values: list[float] = []
    for start in range(0, stereo.shape[0] - win + 1, hop):
        chunk = stereo[start : start + win]
        try:
            lufs = meter.integrated_loudness(chunk)
        except Exception:
            continue
        if np.isfinite(lufs):
            center_sec = (start + win / 2.0) / buffer.sample_rate
            times.append(center_sec)
            values.append(float(lufs))

    return np.array(times), np.array(values)


def apply_macro_compression(
    buffer: AudioBuffer,
    ratio: float,
    window_s: float = WINDOW_SECONDS,
    hop_s: float = HOP_SECONDS,
    max_correction_db: float = MAX_CORRECTION_DB,
    smoothing_s: float = GAIN_SMOOTHING_SECONDS,
) -> AudioBuffer:
    """Applique le rider de gain macro. `ratio == 0.0` ou signal trop court
    pour mesurer une courbe court-terme : retourne le buffer inchangé."""
    if ratio <= 0.0:
        return buffer

    times, short_term = compute_short_term_loudness_curve(buffer, window_s=window_s, hop_s=hop_s)
    if times.size == 0:
        return buffer

    meter = pyln.Meter(buffer.sample_rate)
    stereo_full = buffer.as_stereo() if buffer.n_channels > 1 else buffer.samples[:, None]
    try:
        reference_lufs = float(meter.integrated_loudness(stereo_full))
    except Exception:
        return buffer
    if not np.isfinite(reference_lufs):
        return buffer

    # Gain correctif par fenêtre : tire chaque section vers la loudness
    # intégrée globale, proportionnellement à `ratio`, plafonné à
    # `max_correction_db` pour éviter tout pompage sur une section extrême.
    correction_db = np.clip((reference_lufs - short_term) * ratio, -max_correction_db, max_correction_db)

    n = stereo_full.shape[0]
    sample_times = np.arange(n) / buffer.sample_rate
    # Interpolation linéaire de la courbe de correction (résolution hop) vers
    # la résolution échantillon, avec extrapolation constante aux bornes.
    gain_db_per_sample = np.interp(sample_times, times, correction_db, left=correction_db[0], right=correction_db[-1])

    # Lissage exponentiel supplémentaire (constante de temps macro) — évite
    # les à-coups de gain à chaque frontière de fenêtre.
    smooth_coeff = float(np.exp(-1.0 / max(1.0, smoothing_s * buffer.sample_rate)))
    smoothed = np.empty_like(gain_db_per_sample)
    current = gain_db_per_sample[0]
    for i, g in enumerate(gain_db_per_sample):
        current = smooth_coeff * current + (1.0 - smooth_coeff) * g
        smoothed[i] = current

    gain_linear = (10.0 ** (smoothed / 20.0))[:, None]
    out = stereo_full.astype(np.float64) * gain_linear
    out_samples = out[:, 0].astype(np.float32) if buffer.n_channels == 1 else out.astype(np.float32)

    return AudioBuffer(samples=out_samples, sample_rate=buffer.sample_rate, source_path=buffer.source_path)
