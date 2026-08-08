"""
Widgets d'analyse temps réel façon studio pro : overlay spectral brut/raffiné
(courbe EQ style FabFilter Pro-Q4), goniomètre stéréo (Lissajous), mètre de
corrélation, et lecteur A/B brut vs raffiné.

Tout le calcul (numpy/scipy) vit dans `refinr/spectrum.py` — ce module ne
fait QUE dessiner (QPainter) et jouer l'audio (QtMultimedia). Aucune de ces
classes n'importe `refinr.chain`/`refinr.batch` : elles reçoivent des
données déjà calculées via `set_curves()`/`set_points()`/`set_value()`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import QPointF, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from refinr.audio_io import AudioBuffer
from refinr.spectrum import (
    compute_correlation,
    compute_goniometer_points,
    compute_spectrum_db,
    extract_window,
)

# Fenêtre d'analyse "live" pendant la lecture : ~200ms à 44.1kHz — assez
# court pour réagir vite, assez long pour une FFT/corrélation stable.
LIVE_WINDOW_SAMPLES = 8192
LIVE_UPDATE_INTERVAL_MS = 120

RAW_COLOR = QColor("#7d80a0")  # gris-bleu discret : signal brut, référence
REFINED_COLOR = QColor("#7c5cff")  # violet primaire : signal raffiné, mis en avant
GRID_COLOR = QColor("#2f313a")
BG_COLOR = QColor("#16171b")
TEXT_COLOR = QColor("#8a8d99")

_AXIS_FREQS_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
_AXIS_DB = [-60, -40, -20, 0]


class SpectrumCurveWidget(QWidget):
    """
    Overlay spectral brut vs raffiné, axe fréquentiel log (20Hz-20kHz), façon
    analyseur d'EQ pro (Pro-Q4, etc.) — deux courbes de couleurs différentes
    superposées sur la même grille, pas de traitement ici, juste le rendu.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._raw: tuple[np.ndarray, np.ndarray] | None = None
        self._refined: tuple[np.ndarray, np.ndarray] | None = None
        self._min_hz = 20.0
        self._max_hz = 20000.0
        self._min_db = -60.0
        self._max_db = 6.0

    def set_curves(
        self,
        raw: tuple[np.ndarray, np.ndarray] | None,
        refined: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self._raw = raw
        self._refined = refined
        self.update()

    def clear(self) -> None:
        self.set_curves(None, None)

    def _x_for_freq(self, freq_hz: float, w: int) -> float:
        log_min, log_max = math.log10(self._min_hz), math.log10(self._max_hz)
        t = (math.log10(max(freq_hz, self._min_hz)) - log_min) / (log_max - log_min)
        return t * w

    def _y_for_db(self, db: float, h: int) -> float:
        t = (db - self._min_db) / (self._max_db - self._min_db)
        return h - t * h

    def _path_for(self, freqs: np.ndarray, dbs: np.ndarray, w: int, h: int) -> QPainterPath:
        path = QPainterPath()
        for i, (f, d) in enumerate(zip(freqs, dbs, strict=True)):
            x, y = self._x_for_freq(f, w), self._y_for_db(float(d), h)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        return path

    def paintEvent(self, event) -> None:  # noqa: N802 (override Qt)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), BG_COLOR)

        # Grille fréquentielle (verticale) + niveau (horizontale)
        painter.setPen(QPen(GRID_COLOR, 1))
        for f in _AXIS_FREQS_HZ:
            x = self._x_for_freq(f, w)
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
        for db in _AXIS_DB:
            y = self._y_for_db(db, h)
            painter.drawLine(QPointF(0, y), QPointF(w, y))

        painter.setPen(QPen(TEXT_COLOR))
        for f in (20, 100, 1000, 10000, 20000):
            x = self._x_for_freq(f, w)
            label = f"{f // 1000}k" if f >= 1000 else str(f)
            painter.drawText(QPointF(min(max(x - 10, 2), w - 24), h - 4), label)

        if self._raw is None and self._refined is None:
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Pas encore d'analyse")
            painter.end()
            return

        if self._raw is not None:
            freqs, dbs = self._raw
            painter.setPen(QPen(RAW_COLOR, 1.5))
            painter.drawPath(self._path_for(freqs, dbs, w, h))

        if self._refined is not None:
            freqs, dbs = self._refined
            painter.setPen(QPen(REFINED_COLOR, 2.0))
            painter.drawPath(self._path_for(freqs, dbs, w, h))

        painter.end()


class GoniometerWidget(QWidget):
    """Goniomètre stéréo (Lissajous 45°) : mono parfait = ligne verticale,
    phase inversée = ligne horizontale, image stéréo large = nuage ouvert."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self._points: np.ndarray | None = None
        self._color = REFINED_COLOR

    def set_points(self, points: np.ndarray | None, color: QColor | None = None) -> None:
        self._points = points
        if color is not None:
            self._color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        cx, cy = self.width() / 2, self.height() / 2
        radius = side / 2 - 6

        painter.fillRect(self.rect(), BG_COLOR)
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))
        painter.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))

        if self._points is None or len(self._points) == 0:
            painter.end()
            return

        scale = radius * 0.9
        color = QColor(self._color)
        color.setAlpha(140)
        painter.setPen(QPen(color, 1))
        for x, y in self._points:
            px = cx + float(x) * scale
            py = cy - float(y) * scale
            painter.drawPoint(QPointF(px, py))
        painter.end()


class CorrelationMeterWidget(QWidget):
    """Barre horizontale -1..+1 (corrélation stéréo Pearson, voir
    `analysis.DynamicsProfile.stereo_correlation`) : rouge = risque de phase
    (mono-incompatible), vert = corrélé/mono-safe."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._value: float | None = None

    def set_value(self, value: float | None) -> None:
        self._value = value
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), BG_COLOR)

        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0.0, QColor("#e05c5c"))  # -1 : anti-corrélé, danger phase
        gradient.setColorAt(0.5, QColor("#d9a441"))  # 0 : neutre
        gradient.setColorAt(1.0, QColor("#5cc26a"))  # +1 : mono-safe
        painter.fillRect(2, 2, w - 4, h - 4, gradient)

        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawLine(QPointF(w / 2, 0), QPointF(w / 2, h))  # repère "0"

        if self._value is not None:
            x = (self._value + 1.0) / 2.0 * w
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
            painter.drawText(QPointF(4, h - 6), f"corr {self._value:+.2f}")
        painter.end()


class StereoWidthTimelineWidget(QWidget):
    """
    Largeur stéréo (corrélation) AU FIL DU MORCEAU, pas un score global —
    répond à "où dans le morceau la stéréo se resserre / la phase pose
    problème ?" (voir `spectrum.compute_correlation_timeline`). Un curseur
    de lecture (playhead) se déplace dessus pendant l'écoute A/B, façon
    timeline de DAW, pour situer visuellement ce qu'on entend.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self._raw: tuple[np.ndarray, np.ndarray] | None = None
        self._refined: tuple[np.ndarray, np.ndarray] | None = None
        self._duration_sec: float = 0.0
        self._playhead_fraction: float | None = None

    def set_timeline(
        self,
        raw: tuple[np.ndarray, np.ndarray] | None,
        refined: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self._raw = raw
        self._refined = refined
        pairs = [p for p in (raw, refined) if p is not None and len(p[0])]
        self._duration_sec = max(t[-1] for t, _c in pairs) if pairs else 0.0
        self.update()

    def set_playhead_fraction(self, fraction: float | None) -> None:
        self._playhead_fraction = fraction
        self.update()

    def clear(self) -> None:
        self.set_timeline(None, None)
        self.set_playhead_fraction(None)

    def _path_for(self, times: np.ndarray, correlations: np.ndarray, w: int, h: int) -> QPainterPath:
        path = QPainterPath()
        for i, (t, corr) in enumerate(zip(times, correlations, strict=True)):
            x = (t / self._duration_sec) * w if self._duration_sec > 0 else 0.0
            y = h - ((float(corr) + 1.0) / 2.0) * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        return path

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), BG_COLOR)

        mid_y = h / 2.0
        painter.setPen(QPen(GRID_COLOR, 1))
        painter.drawLine(QPointF(0, mid_y), QPointF(w, mid_y))  # repère corrélation = 0
        painter.setPen(QPen(TEXT_COLOR))
        painter.drawText(QPointF(4, 12), "+1")
        painter.drawText(QPointF(4, h - 4), "-1")

        if self._raw is None and self._refined is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Pas encore d'analyse")
            painter.end()
            return

        if self._raw is not None and len(self._raw[0]):
            painter.setPen(QPen(RAW_COLOR, 1.2))
            painter.drawPath(self._path_for(*self._raw, w, h))

        if self._refined is not None and len(self._refined[0]):
            painter.setPen(QPen(REFINED_COLOR, 1.6))
            painter.drawPath(self._path_for(*self._refined, w, h))

        if self._playhead_fraction is not None:
            x = max(0.0, min(1.0, self._playhead_fraction)) * w
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawLine(QPointF(x, 0), QPointF(x, h))

        painter.end()


class ABPlayerWidget(QWidget):
    """
    Lecteur A/B brut vs raffiné : un seul `QMediaPlayer`, on change juste la
    source selon le bouton actif (pas deux lecteurs synchronisés — plus
    simple et suffisant pour une écoute comparative, pas un vrai crossfade
    sample-accurate). Le bouton "Raffiné" reste désactivé tant que le
    fichier n'a pas été traité (pas de sortie à écouter).

    Deux points essentiels pour que la comparaison A/B soit honnête :

    1. ÉGALISATION DE NIVEAU ("Égaliser les niveaux", cochée par défaut) :
       le fichier raffiné est presque toujours plus fort (gain staging vers
       la cible LUFS du profil) — sans compensation, l'oreille perçoit "plus
       fort = meilleur" indépendamment du contenu réel du traitement, biais
       classique en A/B audio. On atténue le PLUS FORT des deux (via
       `QAudioOutput.setVolume`, qui ne peut qu'atténuer, pas amplifier
       au-delà de l'unité) pour ramener les deux sources au même niveau
       perçu avant de comparer. Décochable si on veut explicitement entendre
       l'effet du gain staging lui-même.
    2. ANALYSE "LIVE" : pendant la lecture, un timer extrait une fenêtre
       courte du buffer en mémoire autour de la position de lecture
       courante et émet `live_update` (spectre/goniomètre/corrélation de
       CETTE fenêtre, pas une moyenne figée du fichier entier) — la GUI
       s'abonne à ce signal pour faire bouger les visualisations en temps
       réel, façon analyseur de studio.
    """

    live_update = pyqtSignal(object, object, float, bool)  # (spectrum, gonio_points, correlation, is_refined)
    playback_state_changed = pyqtSignal(bool)  # True = en lecture (utile pour revenir aux courbes statiques à l'arrêt)
    playhead_changed = pyqtSignal(float, bool)  # (fraction 0..1 de la durée, is_refined) — pour le curseur de timeline

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._raw_path: str | None = None
        self._refined_path: str | None = None
        self._raw_buffer: AudioBuffer | None = None
        self._refined_buffer: AudioBuffer | None = None
        self._raw_lufs: float | None = None
        self._refined_lufs: float | None = None
        self._listening_refined = False
        self._match_levels = True

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(LIVE_UPDATE_INTERVAL_MS)
        self._live_timer.timeout.connect(self._emit_live_update)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(32)
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("color: #8a8d99;")
        row.addWidget(self.time_label)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._player.setPosition)
        row.addWidget(self.position_slider, stretch=1)

        self.raw_btn = QPushButton("Écouter : Brut")
        self.raw_btn.setCheckable(True)
        self.raw_btn.setChecked(True)
        self.raw_btn.clicked.connect(lambda: self._set_listening(refined=False))
        row.addWidget(self.raw_btn)

        self.refined_btn = QPushButton("Écouter : Raffiné")
        self.refined_btn.setCheckable(True)
        self.refined_btn.setEnabled(False)
        self.refined_btn.clicked.connect(lambda: self._set_listening(refined=True))
        row.addWidget(self.refined_btn)
        outer.addLayout(row)

        match_row = QHBoxLayout()
        self.match_levels_checkbox = QCheckBox("Égaliser les niveaux (comparaison honnête)")
        self.match_levels_checkbox.setChecked(True)
        self.match_levels_checkbox.setToolTip(
            "Compense la différence de loudness entre brut et raffiné avant de comparer — "
            "sans ça, le fichier le plus fort semble presque toujours 'meilleur' à l'oreille, "
            "indépendamment du traitement réel. Décoche pour entendre le gain staging brut."
        )
        self.match_levels_checkbox.toggled.connect(self._on_match_levels_toggled)
        match_row.addWidget(self.match_levels_checkbox)
        self.level_match_label = QLabel("")
        self.level_match_label.setStyleSheet("color: #6f7280; font-size: 11px;")
        match_row.addWidget(self.level_match_label)
        match_row.addStretch(1)
        outer.addLayout(match_row)

    # ------------------------------------------------------------ Sources

    def set_sources(
        self,
        raw_path: str | None,
        refined_path: str | None = None,
        raw_buffer: AudioBuffer | None = None,
        refined_buffer: AudioBuffer | None = None,
        raw_lufs: float | None = None,
        refined_lufs: float | None = None,
    ) -> None:
        """
        À appeler quand on sélectionne un fichier (raw_path/raw_buffer/raw_lufs)
        et, plus tard, quand son traitement se termine (refined_*). Les
        buffers servent à l'analyse "live" (fenêtres extraites en mémoire,
        pas de re-décodage disque à chaque tick) ; les LUFS servent à
        l'égalisation de niveau.
        """
        was_playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self._player.stop()
        self._raw_path = raw_path
        self._refined_path = refined_path
        self._raw_buffer = raw_buffer
        self._refined_buffer = refined_buffer
        self._raw_lufs = raw_lufs
        self._refined_lufs = refined_lufs
        self.refined_btn.setEnabled(refined_path is not None)
        if refined_path is None and self._listening_refined:
            self._listening_refined = False
            self.raw_btn.setChecked(True)
            self.refined_btn.setChecked(False)
        self._load_current_source()
        self._apply_level_matching()
        if was_playing and self._current_path():
            self._player.play()

    def _current_path(self) -> str | None:
        return self._refined_path if self._listening_refined else self._raw_path

    def _current_buffer(self) -> AudioBuffer | None:
        return self._refined_buffer if self._listening_refined else self._raw_buffer

    def _load_current_source(self) -> None:
        path = self._current_path()
        self._player.setSource(QUrl.fromLocalFile(path) if path else QUrl())

    def _set_listening(self, refined: bool) -> None:
        if refined and self._refined_path is None:
            self.refined_btn.setChecked(False)
            return
        self._listening_refined = refined
        self.raw_btn.setChecked(not refined)
        self.refined_btn.setChecked(refined)
        was_playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        position = self._player.position()  # on garde la position pour comparer au même instant
        self._load_current_source()
        self._apply_level_matching()
        self._player.setPosition(position)
        if was_playing:
            self._player.play()

    # -------------------------------------------------- Égalisation niveau

    def _on_match_levels_toggled(self, checked: bool) -> None:
        self._match_levels = checked
        self._apply_level_matching()

    def _apply_level_matching(self) -> None:
        """
        N'atténue JAMAIS en dessous de ce qui est nécessaire, et ne peut
        qu'atténuer (QAudioOutput ne sait pas amplifier au-delà de 1.0) : le
        plus fort des deux LUFS est ramené au niveau du plus faible. Si l'un
        des deux LUFS est inconnu (raffiné pas encore mesuré) ou si la case
        est décochée, volume neutre (1.0).
        """
        if not self._match_levels or self._raw_lufs is None or self._refined_lufs is None:
            self._audio_out.setVolume(1.0)
            self.level_match_label.setText("")
            return

        diff_db = self._refined_lufs - self._raw_lufs  # >0 si le raffiné est plus fort
        # atténue celui qui est le plus fort des deux, jamais le contraire
        gain_db = min(0.0, -diff_db) if self._listening_refined else min(0.0, diff_db)
        volume = 10 ** (gain_db / 20.0)
        self._audio_out.setVolume(max(0.0, min(1.0, volume)))
        if abs(diff_db) > 0.2:
            self.level_match_label.setText(f"Δ niveau brut/raffiné : {diff_db:+.1f} LU — compensé")
        else:
            self.level_match_label.setText("")

    # -------------------------------------------------------- Lecture

    def _toggle_play(self) -> None:
        if self._current_path() is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def stop(self) -> None:
        self._player.stop()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setText("⏸" if is_playing else "▶")
        if is_playing and self._current_buffer() is not None:
            self._live_timer.start()
        else:
            self._live_timer.stop()
        self.playback_state_changed.emit(is_playing)

    def _on_position_changed(self, position_ms: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position_ms)
        self.position_slider.blockSignals(False)
        self.time_label.setText(f"{_fmt_ms(position_ms)} / {_fmt_ms(self._player.duration())}")

        duration_ms = self._player.duration()
        if duration_ms > 0:
            self.playhead_changed.emit(position_ms / duration_ms, self._listening_refined)

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.position_slider.setRange(0, duration_ms)

    # -------------------------------------------------------- Analyse live

    def _emit_live_update(self) -> None:
        buffer = self._current_buffer()
        if buffer is None or buffer.n_frames == 0:
            return
        position_ms = self._player.position()
        center_sample = int(position_ms / 1000.0 * buffer.sample_rate)
        window = extract_window(buffer, center_sample, LIVE_WINDOW_SAMPLES)
        if window.n_frames == 0:
            return
        spectrum = compute_spectrum_db(window, n_points=150)
        gonio = compute_goniometer_points(window, max_points=800)
        correlation = compute_correlation(window)
        self.live_update.emit(spectrum, gonio, correlation, self._listening_refined)


def _fmt_ms(ms: int) -> str:
    total_s = max(0, ms) // 1000
    return f"{total_s // 60}:{total_s % 60:02d}"


class FileInfoBar(QWidget):
    """Ligne d'info fichier brut : sample rate / bit depth / durée — visible
    immédiatement à l'ajout, avant tout traitement."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("—")
        self.label.setStyleSheet("color: #c9c9ce; font-family: Menlo, monospace; font-size: 11px;")
        layout.addWidget(self.label)
        layout.addStretch(1)

    def set_info(self, sample_rate: int, bit_depth_label: str, channels: int, duration_seconds: float) -> None:
        ch = "stéréo" if channels >= 2 else "mono"
        mins, secs = divmod(int(duration_seconds), 60)
        self.label.setText(f"{sample_rate / 1000:g}kHz / {bit_depth_label} / {ch} / {mins}:{secs:02d}")

    def clear(self) -> None:
        self.label.setText("—")
