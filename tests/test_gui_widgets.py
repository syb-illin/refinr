"""
Tests "smoke" des widgets GUI (gui/analyzer_widgets.py, gui/main_window.py)
via pytest-qt, en mode `QT_QPA_PLATFORM=offscreen` — pas besoin d'un vrai
écran, tourne aussi bien dans un runner GitHub Actions ubuntu-latest headless
que sur une machine de dev locale.

Ce fichier tourne dans un job CI SÉPARÉ de la suite principale (voir
.github/workflows/ci.yml, job `gui-tests`) car PyQt6 n'est PAS dans
`requirements.txt` sur Linux (marqueur `sys_platform == "darwin"` — la GUI
est macOS-only en usage réel). Ici on installe PyQt6 explicitement pour
vérifier que les widgets s'instancient, réagissent aux données, et ne
plantent pas — pas pour tester le rendu pixel-perfect (hors de portée d'un
test automatisé), ni le hosting AU réel (macOS uniquement, voir
tools/au_host_smoketest.py).

`pytest.importorskip` en tête : si PyQt6 n'est pas installé (cas normal en
dev sur Linux/CI principal), ce fichier est silencieusement skip plutôt que
de faire échouer toute la suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtGui import QColor  # noqa: E402

from gui.analyzer_widgets import (  # noqa: E402
    RAW_COLOR,
    REFINED_COLOR,
    ABPlayerWidget,
    CorrelationMeterWidget,
    FileInfoBar,
    GoniometerWidget,
    SpectrumCurveWidget,
    StereoWidthTimelineWidget,
)
from refinr.audio_io import AudioBuffer  # noqa: E402
from refinr.spectrum import (  # noqa: E402
    compute_correlation_timeline,
    compute_goniometer_points,
    compute_spectrum_db,
    file_info,
)

SR = 44100


def _tone_buffer(freq: float = 440.0, seconds: float = 1.0, amplitude: float = 0.3) -> AudioBuffer:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioBuffer(samples=np.stack([mono, mono], axis=1), sample_rate=SR)


# ------------------------------------------------------------ SpectrumCurveWidget


def test_spectrum_curve_widget_starts_empty(qtbot):
    widget = SpectrumCurveWidget()
    qtbot.addWidget(widget)
    assert widget._raw is None
    assert widget._refined is None


def test_spectrum_curve_widget_accepts_real_data(qtbot):
    widget = SpectrumCurveWidget()
    qtbot.addWidget(widget)
    raw = compute_spectrum_db(_tone_buffer(440.0))
    refined = compute_spectrum_db(_tone_buffer(880.0))
    widget.set_curves(raw, refined)
    assert widget._raw is raw
    assert widget._refined is refined
    widget.resize(400, 160)
    widget.repaint()  # ne doit pas lever d'exception (paintEvent complet)


def test_spectrum_curve_widget_clear_resets_state(qtbot):
    widget = SpectrumCurveWidget()
    qtbot.addWidget(widget)
    widget.set_curves(compute_spectrum_db(_tone_buffer()), None)
    widget.clear()
    assert widget._raw is None
    assert widget._refined is None


# ------------------------------------------------------------ GoniometerWidget


def test_goniometer_widget_accepts_points_and_paints(qtbot):
    widget = GoniometerWidget()
    qtbot.addWidget(widget)
    points = compute_goniometer_points(_tone_buffer())
    widget.set_points(points, REFINED_COLOR)
    assert widget._points is points
    widget.resize(150, 150)
    widget.repaint()


def test_goniometer_widget_handles_none_and_empty(qtbot):
    widget = GoniometerWidget()
    qtbot.addWidget(widget)
    widget.set_points(None)
    widget.repaint()
    widget.set_points(np.zeros((0, 2), dtype=np.float32))
    widget.repaint()


# ------------------------------------------------------------ CorrelationMeterWidget


def test_correlation_meter_accepts_full_range(qtbot):
    widget = CorrelationMeterWidget()
    qtbot.addWidget(widget)
    for value in (-1.0, -0.3, 0.0, 0.5, 1.0):
        widget.set_value(value)
        assert widget._value == value
        widget.repaint()


def test_correlation_meter_accepts_none(qtbot):
    widget = CorrelationMeterWidget()
    qtbot.addWidget(widget)
    widget.set_value(None)
    widget.repaint()


# ------------------------------------------------------------ FileInfoBar


def test_file_info_bar_displays_metadata(qtbot, tmp_path):
    import soundfile as sf

    mono = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, SR))).astype(np.float32)
    path = tmp_path / "probe.wav"
    sf.write(str(path), np.stack([mono, mono], axis=1), SR, subtype="PCM_24")

    widget = FileInfoBar()
    qtbot.addWidget(widget)
    info = file_info(path)
    widget.set_info(info.sample_rate, info.bit_depth_label, info.channels, info.duration_seconds)
    assert "44.1kHz" in widget.label.text()
    assert "24-bit" in widget.label.text()

    widget.clear()
    assert widget.label.text() == "—"


# ------------------------------------------------------------ ABPlayerWidget


def test_ab_player_widget_constructs_and_toggles_buttons(qtbot):
    widget = ABPlayerWidget()
    qtbot.addWidget(widget)
    assert widget.raw_btn.isChecked()
    assert not widget.refined_btn.isEnabled()

    widget.set_sources("/tmp/does_not_need_to_exist_raw.wav", None)
    assert not widget.refined_btn.isEnabled()

    widget.set_sources(
        "/tmp/does_not_need_to_exist_raw.wav",
        "/tmp/does_not_need_to_exist_refined.wav",
        raw_lufs=-14.0,
        refined_lufs=-9.0,
    )
    assert widget.refined_btn.isEnabled()


def test_ab_player_widget_level_matching_attenuates_louder_source(qtbot):
    """Le raffiné (souvent plus fort après gain staging) doit être atténué,
    pas le brut — voir ABPlayerWidget._apply_level_matching."""
    widget = ABPlayerWidget()
    qtbot.addWidget(widget)
    widget.set_sources(
        "/tmp/raw.wav",
        "/tmp/refined.wav",
        raw_lufs=-14.0,
        refined_lufs=-9.0,  # 5 LU plus fort
    )
    # écoute brut par défaut -> pas plus fort que le raffiné -> volume neutre
    assert widget._audio_out.volume() == pytest.approx(1.0, abs=1e-6)

    widget._set_listening(refined=True)
    # écoute raffiné, qui est plus fort -> doit être atténué (<1.0)
    assert widget._audio_out.volume() < 1.0


def test_ab_player_widget_match_levels_checkbox_disables_compensation(qtbot):
    widget = ABPlayerWidget()
    qtbot.addWidget(widget)
    widget.set_sources("/tmp/raw.wav", "/tmp/refined.wav", raw_lufs=-14.0, refined_lufs=-9.0)
    widget._set_listening(refined=True)
    assert widget._audio_out.volume() < 1.0

    widget.match_levels_checkbox.setChecked(False)
    assert widget._audio_out.volume() == pytest.approx(1.0, abs=1e-6)


def test_ab_player_widget_live_update_signal_carries_expected_types(qtbot):
    widget = ABPlayerWidget()
    qtbot.addWidget(widget)
    buffer = _tone_buffer(seconds=1.0)
    widget.set_sources("/tmp/raw.wav", None, raw_buffer=buffer, raw_lufs=-14.0)

    received = []
    widget.live_update.connect(lambda *args: received.append(args))
    widget._emit_live_update()  # simule un tick du timer sans dépendre du vrai décodage média

    assert len(received) == 1
    spectrum, gonio, correlation, is_refined = received[0]
    assert len(spectrum) == 2
    assert gonio.shape[1] == 2
    assert isinstance(correlation, float)
    assert is_refined is False


# ------------------------------------------------------------ StereoWidthTimelineWidget


def test_stereo_timeline_widget_starts_empty(qtbot):
    widget = StereoWidthTimelineWidget()
    qtbot.addWidget(widget)
    assert widget._raw is None
    assert widget._refined is None
    widget.repaint()  # doit afficher "Pas encore d'analyse" sans planter


def test_stereo_timeline_widget_accepts_both_curves_and_paints(qtbot):
    widget = StereoWidthTimelineWidget()
    qtbot.addWidget(widget)
    raw = compute_correlation_timeline(_tone_buffer(seconds=1.0), n_windows=30)
    refined = compute_correlation_timeline(_tone_buffer(seconds=1.0), n_windows=30)
    widget.set_timeline(raw, refined)
    assert widget._duration_sec > 0
    widget.resize(300, 80)
    widget.repaint()


def test_stereo_timeline_widget_playhead_and_clear(qtbot):
    widget = StereoWidthTimelineWidget()
    qtbot.addWidget(widget)
    widget.set_timeline(compute_correlation_timeline(_tone_buffer(seconds=1.0)), None)
    widget.set_playhead_fraction(0.5)
    assert widget._playhead_fraction == 0.5
    widget.repaint()

    widget.clear()
    assert widget._raw is None
    assert widget._playhead_fraction is None


def test_ab_player_widget_emits_playhead_fraction_signal(qtbot):
    """Vérifie la formule de fraction (position/duration), pas le vrai
    décodage média (hors de portée d'un test offscreen sans backend audio
    garanti) : simule directement _on_position_changed."""
    widget = ABPlayerWidget()
    qtbot.addWidget(widget)

    received = []
    widget.playhead_changed.connect(lambda frac, is_refined: received.append((frac, is_refined)))

    class _FakeDuration:
        @staticmethod
        def duration():
            return 10000

    widget._player.duration = _FakeDuration.duration
    widget._on_position_changed(2500)

    assert len(received) == 1
    fraction, is_refined = received[0]
    assert fraction == pytest.approx(0.25)
    assert is_refined is False


# ------------------------------------------------------------ Couleurs (sanity)


def test_raw_and_refined_colors_are_distinct_qcolors():
    assert isinstance(RAW_COLOR, QColor)
    assert isinstance(REFINED_COLOR, QColor)
    assert RAW_COLOR.name() != REFINED_COLOR.name()
