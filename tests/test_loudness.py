"""
Tests du module loudness — tournent sur n'importe quel OS (pas de PyObjC).

Lancer:  python3 -m pytest tests/ -v
(ou, sans pytest installé:  python3 tests/test_loudness.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import refinr.loudness as loudness_module
from refinr.audio_io import AudioBuffer
from refinr.loudness import gain_to_target_lufs, limit_true_peak, measure
from tests.generate_test_audio import SR, make_calm_dynamic, make_hot_clipped_source


def _buffer_from(samples):
    return AudioBuffer(samples=samples, sample_rate=SR)


def test_gain_staging_reaches_target():
    buf = _buffer_from(make_hot_clipped_source())
    gained, gain_db = gain_to_target_lufs(buf, target_lufs=-18.0)
    m = measure(gained)
    assert abs(m.integrated_lufs - (-18.0)) < 0.5, f"LUFS après gain staging hors cible: {m.integrated_lufs}"


def test_gain_staging_respects_true_peak_ceiling():
    buf = _buffer_from(make_hot_clipped_source())
    gained, _ = gain_to_target_lufs(buf, target_lufs=0.0, ceiling_dbtp=-1.0)  # cible volontairement irréaliste
    m = measure(gained)
    assert m.true_peak_dbtp <= -1.0 + 1e-6, f"True peak dépasse le plafond: {m.true_peak_dbtp}"


def test_limiter_enforces_ceiling():
    buf = _buffer_from(make_hot_clipped_source())
    limited = limit_true_peak(buf, ceiling_dbtp=-1.0)
    m = measure(limited)
    assert m.true_peak_dbtp <= -1.0 + 0.05, f"Limiteur laisse passer un peak trop haut: {m.true_peak_dbtp}"


def test_calm_signal_is_quiet_and_dynamic():
    buf = _buffer_from(make_calm_dynamic())
    m = measure(buf)
    assert m.integrated_lufs < -15.0
    assert m.loudness_range_lu is not None


def test_lra_falls_back_to_approximation_when_ebur128_unavailable():
    """measure_loudness_range doit rester utilisable même sans pyebur128
    (dégradation gracieuse, voir _measure_loudness_range_approx) — on force
    le flag pour couvrir ce chemin indépendamment de l'environnement de test."""
    buf = _buffer_from(make_calm_dynamic())
    stereo = buf.as_stereo()
    original_flag = loudness_module._PYEBUR128_AVAILABLE
    try:
        loudness_module._PYEBUR128_AVAILABLE = False
        lra_via_public_api = loudness_module.measure_loudness_range(stereo, buf.sample_rate)
    finally:
        loudness_module._PYEBUR128_AVAILABLE = original_flag

    lra_direct_approx = loudness_module._measure_loudness_range_approx(stereo, buf.sample_rate)
    assert lra_via_public_api == lra_direct_approx
    assert lra_via_public_api is not None


def test_lra_ebur128_path_falls_back_on_failure():
    """Si pyebur128 est marqué disponible mais que le calcul échoue (signal
    dégénéré, erreur de la lib C...), on doit retomber sur l'approximation
    plutôt que de propager l'exception ou retourner None sans raison."""
    buf = _buffer_from(make_calm_dynamic())
    stereo = buf.as_stereo()
    original_flag = loudness_module._PYEBUR128_AVAILABLE
    original_fn = loudness_module._measure_loudness_range_ebur128
    try:
        loudness_module._PYEBUR128_AVAILABLE = True
        loudness_module._measure_loudness_range_ebur128 = lambda *_args, **_kwargs: None
        lra = loudness_module.measure_loudness_range(stereo, buf.sample_rate)
    finally:
        loudness_module._PYEBUR128_AVAILABLE = original_flag
        loudness_module._measure_loudness_range_ebur128 = original_fn

    assert lra is not None


if __name__ == "__main__":
    test_gain_staging_reaches_target()
    test_gain_staging_respects_true_peak_ceiling()
    test_limiter_enforces_ceiling()
    test_calm_signal_is_quiet_and_dynamic()
    test_lra_falls_back_to_approximation_when_ebur128_unavailable()
    test_lra_ebur128_path_falls_back_on_failure()
    print("Tous les tests loudness sont passés.")
