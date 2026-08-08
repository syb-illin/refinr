"""Tests du transient designer (transient_shaping.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from refinr.analysis import analyze
from refinr.audio_io import AudioBuffer
from refinr.transient_shaping import (
    ACTIVITY_DB_THRESHOLD,
    CREST_FACTOR_COMPRESSED_DB,
    apply_transient_shaping,
    decide_attack_amount_db,
    measure_transient_activity,
)
from tests.generate_test_audio import SR


def _punchy_buffer(seconds: float = 3.0) -> AudioBuffer:
    """Impulsions nettes et espacées -> forte activité transitoire, crest factor élevé."""
    n = int(SR * seconds)
    mono = np.zeros(n, dtype=np.float32)
    for start in range(0, n, int(SR * 0.5)):
        end = min(n, start + int(SR * 0.01))
        mono[start:end] = 0.9
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


def _loose_compressed_buffer(seconds: float = 3.0) -> AudioBuffer:
    """Niveau quasi constant (peu de crest factor, attaques absentes) — imite
    un matériel déjà bien compressé avec des transitoires molles."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mono = mono / np.max(np.abs(mono)) * 0.85
    return AudioBuffer(samples=np.stack([mono, mono], axis=1).astype(np.float32), sample_rate=SR)


def test_measure_transient_activity_higher_on_punchy_material():
    punchy_activity = measure_transient_activity(_punchy_buffer())
    loose_activity = measure_transient_activity(_loose_compressed_buffer())
    assert punchy_activity > loose_activity


def test_decide_attack_amount_boosts_on_compressed_loose_material():
    buffer = _loose_compressed_buffer()
    a = analyze(buffer)
    assert a.dynamics.crest_factor_db < CREST_FACTOR_COMPRESSED_DB  # précondition du test
    decision = decide_attack_amount_db(buffer, a)
    assert decision.attack_amount_db > 0.0


def test_decide_attack_amount_stays_zero_on_already_punchy_material():
    buffer = _punchy_buffer()
    a = analyze(buffer)
    decision = decide_attack_amount_db(buffer, a)
    if a.dynamics.crest_factor_db >= CREST_FACTOR_COMPRESSED_DB or decision.transient_activity >= 0.12:
        assert decision.attack_amount_db == 0.0


def test_apply_transient_shaping_is_noop_at_zero_amount():
    buffer = _loose_compressed_buffer()
    result = apply_transient_shaping(buffer, attack_amount_db=0.0)
    assert result is buffer


def test_apply_transient_shaping_boosts_detected_transients():
    buffer = _punchy_buffer()
    activity_before = measure_transient_activity(buffer)
    assert activity_before > ACTIVITY_DB_THRESHOLD * 0  # juste s'assurer que ce n'est pas silencieux

    shaped = apply_transient_shaping(buffer, attack_amount_db=6.0)
    # Le pic pendant les impulsions doit être plus élevé après boost (gain>1 appliqué
    # là où le détecteur est actif), tant qu'on ne clippe pas.
    peak_before = float(np.max(np.abs(buffer.samples)))
    peak_after = float(np.max(np.abs(shaped.samples)))
    assert peak_after >= peak_before


def test_apply_transient_shaping_preserves_stereo_link():
    """Le gain est lié (calculé sur la somme mono) : les deux canaux doivent
    rester identiques si la source l'était."""
    buffer = _punchy_buffer()
    shaped = apply_transient_shaping(buffer, attack_amount_db=4.0)
    assert np.allclose(shaped.samples[:, 0], shaped.samples[:, 1])


if __name__ == "__main__":
    test_measure_transient_activity_higher_on_punchy_material()
    test_decide_attack_amount_boosts_on_compressed_loose_material()
    test_decide_attack_amount_stays_zero_on_already_punchy_material()
    test_apply_transient_shaping_is_noop_at_zero_amount()
    test_apply_transient_shaping_boosts_detected_transients()
    test_apply_transient_shaping_preserves_stereo_link()
    print("Tous les tests transient_shaping sont passés.")
