"""
Fenêtre principale de l'app.

Flux :
  1. Glisser-déposer, ou menu Fichier > Ouvrir des WAV… (plusieurs fichiers
     possibles), pour ajouter des WAV.
  2. Pour chaque fichier ajouté, on calcule immédiatement son analyse et un
     aperçu de ce qui sera décidé : EQ Pro-Q4 pilotée dynamiquement (jamais
     de preset EQ, voir refinr.proq4_control) + presets Saturation/Tape
     choisis par tags (ces plugins n'ont pas encore été reverse-engineered
     pour un pilotage fin).
  3. Choix du profil de destination (Spotify, YouTube, Apple Music, ...).
  4. "Lancer le batch" traite tous les fichiers en parallèle
     (refinr.batch, process pool), avec barre de progression et statut
     par ligne.
  5. À la fin : ouverture ou export du rapport HTML détaillé.

Mode debug (menu Réglages) : affiche un panneau de log en bas de la
fenêtre, alimenté par le module `logging` standard — utile pour voir le
détail des erreurs sans repasser par le Terminal.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from refinr import __version__
from refinr.analysis import analyze
from refinr.audio_io import load_wav
from refinr.preset_mapping import PresetLibrary, select_chain
from refinr.preset_types import PluginRole
from refinr.profiles import DEFAULT_PROFILES_PATH, load_profiles
from refinr.proq4_control import decide_bands
from refinr.spectrum import compute_correlation_timeline, compute_goniometer_points, compute_spectrum_db, file_info

from .analyzer_widgets import (
    RAW_COLOR,
    REFINED_COLOR,
    ABPlayerWidget,
    CorrelationMeterWidget,
    FileInfoBar,
    GoniometerWidget,
    SpectrumCurveWidget,
    StereoWidthTimelineWidget,
)
from .worker import BatchWorker, RefinedAnalysisWorker

DEFAULT_PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"
DEFAULT_OUTPUT_DIR = Path.home() / "Refinr_Output"

COLUMNS = ["Fichier", "Tags analyse", "EQ Pro-Q4 (auto)", "Saturation", "Tape (J37)", "Statut"]

CUSTOM_PROFILE_KEY = "__custom__"
# Sliders en dixièmes (QSlider ne fait que des int) : LUFS -24.0..-6.0, true peak -3.0..0.0.
LUFS_SLIDER_RANGE = (-240, -60)
TRUE_PEAK_SLIDER_RANGE = (-30, 0)
LUFS_SLIDER_DEFAULT = -140  # -14.0 LUFS, cible la plus commune (streaming musique)
TRUE_PEAK_SLIDER_DEFAULT = -10  # -1.0 dBTP

# Bornes "raisonnables" pour un avertissement (pas un blocage — l'utilisateur
# reste libre, mais ces valeurs sortent des pratiques standards de diffusion).
LUFS_WARN_LOUDER_THAN = -9.0  # plus fort que ça -> quasi aucune plateforme ne vise ça, risque de limiting excessif
LUFS_WARN_QUIETER_THAN = -20.0  # plus bas -> presque toutes les plateformes vont sur-normaliser à la lecture
TRUE_PEAK_WARN_ABOVE = -0.3  # au-dessus -> risque réel d'intersample clipping après reconversion lossy

# Sliders "kHz"/"bits" de livraison (cibles personnalisées) : valeurs discrètes
# standards, pas un continuum — le slider avance par index dans ces listes.
SAMPLE_RATE_OPTIONS = [44100, 48000, 88200, 96000]
BIT_DEPTH_OPTIONS = [("PCM_16", "16-bit"), ("PCM_24", "24-bit"), ("PCM_32", "32-bit int"), ("FLOAT", "32-bit float")]
SAMPLE_RATE_DEFAULT_INDEX = 0  # 44100Hz
BIT_DEPTH_DEFAULT_INDEX = 1  # PCM_24

_STYLESHEET = """
QMainWindow, QWidget { background-color: #1e1f24; color: #e8e8ea; font-size: 12.5px; }
QLabel { color: #c9c9ce; }
QLabel#versionLabel { color: #6f7280; font-size: 11px; }
QPushButton {
  background-color: #34363f; color: #f0f0f2; border: 1px solid #45474f;
  border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { background-color: #3d3f49; }
QPushButton:pressed { background-color: #2a2c33; }
QPushButton:disabled { color: #6f7280; background-color: #2a2c33; }
QPushButton#primary { background-color: #4e6bff; border: 1px solid #4e6bff; }
QPushButton#primary:hover { background-color: #5f7aff; }
QPushButton#primary:disabled { background-color: #33385c; border: 1px solid #33385c; }
QComboBox, QSpinBox {
  background-color: #2a2c33; border: 1px solid #45474f; border-radius: 6px; padding: 4px 8px;
}
QTableWidget {
  background-color: #24262c; alternate-background-color: #26282f; gridline-color: #34363f;
  border: 1px solid #34363f; border-radius: 8px;
}
QHeaderView::section {
  background-color: #2a2c33; color: #c9c9ce; border: none; padding: 6px; font-weight: 600;
}
QProgressBar {
  background-color: #2a2c33; border: 1px solid #45474f; border-radius: 6px; text-align: center; color: #e8e8ea;
}
QProgressBar::chunk { background-color: #4e6bff; border-radius: 6px; }
QPlainTextEdit {
  background-color: #16171b; color: #9fe89f; border: 1px solid #34363f; font-family: Menlo, monospace; font-size: 11px;
}
QMenuBar { background-color: #1e1f24; color: #e8e8ea; }
QMenuBar::item:selected { background-color: #34363f; }
QMenu { background-color: #24262c; color: #e8e8ea; border: 1px solid #34363f; }
QMenu::item:selected { background-color: #34363f; }
QSlider::groove:horizontal { background-color: #2a2c33; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
  background-color: #7c5cff; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
}
QSlider::handle:horizontal:disabled { background-color: #4a4c56; }
QFrame#detailPanel { background-color: #1a1b20; border: 1px solid #2a2c33; border-radius: 8px; }
QSplitter::handle { background-color: #1e1f24; }
"""


class _QtLogHandler(logging.Handler):
    """Relaie les logs Python vers un QPlainTextEdit (panneau de debug)."""

    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Refinr v{__version__} — retraitement WAV via AU")
        self.resize(1250, 680)
        self.setAcceptDrops(True)
        self.setStyleSheet(_STYLESHEET)

        self.library = PresetLibrary.load(DEFAULT_PRESETS_ROOT)
        self.catalog = load_profiles(DEFAULT_PROFILES_PATH)

        self.row_data: dict[int, dict] = {}  # row -> {"path": ..., "analysis":..., "eq_bands":..., "selection":...}
        self.worker: BatchWorker | None = None
        self.refined_analysis_worker: RefinedAnalysisWorker | None = None
        self._last_report_paths: dict | None = None

        # État du panneau d'analyse pour la ligne sélectionnée — voir
        # _restore_static_analyzer_view / _on_live_update. Initialisé ici
        # (avant _build_ui) car _on_playback_state_changed peut être
        # déclenché par des signaux Qt avant toute sélection de ligne.
        self._selected_row: int | None = None
        self._static_spectrum: tuple = (None, None)
        self._static_goniometer: tuple = (None, RAW_COLOR)
        self._static_correlation: float | None = None

        self._build_menu()
        self._build_ui()
        self._build_log_dock()
        self._refresh_library_warning()

        self.logger = logging.getLogger("refinr.gui")

    # --------------------------------------------------------------- Menus

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&Fichier")

        open_action = QAction("Ouvrir des WAV…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_add_files_clicked)
        file_menu.addAction(open_action)

        clear_action = QAction("Vider la liste", self)
        clear_action.triggered.connect(self._on_clear_clicked)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        self.export_report_action = QAction("Exporter le rapport…", self)
        self.export_report_action.setEnabled(False)
        self.export_report_action.triggered.connect(self._on_export_report_clicked)
        file_menu.addAction(self.export_report_action)

        self.open_report_action = QAction("Ouvrir le rapport", self)
        self.open_report_action.setEnabled(False)
        self.open_report_action.triggered.connect(self._on_open_report_clicked)
        file_menu.addAction(self.open_report_action)

        file_menu.addSeparator()

        quit_action = QAction("Quitter", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        settings_menu = menubar.addMenu("&Réglages")

        self.debug_action = QAction("Mode debug (afficher les logs)", self)
        self.debug_action.setCheckable(True)
        self.debug_action.toggled.connect(self._on_debug_toggled)
        settings_menu.addAction(self.debug_action)

        settings_menu.addSeparator()

        self.suno_mode_action = QAction("Source Suno / IA générée (corrections spécifiques)", self)
        self.suno_mode_action.setCheckable(True)
        self.suno_mode_action.setToolTip(
            "Ajoute des corrections EQ supplémentaires ciblant les artefacts connus des "
            "générateurs IA type Suno (fizz HF, buzz métallique vocal) — voir "
            "config/suno_artifacts_kb.md. Désactivé par défaut : ne s'applique jamais "
            "automatiquement, seulement si tu sais que la source vient d'un générateur IA."
        )
        settings_menu.addAction(self.suno_mode_action)

        self.export_presets_action = QAction("Exporter les presets .aupreset (réutilisables dans Logic Pro)", self)
        self.export_presets_action.setCheckable(True)
        self.export_presets_action.setToolTip(
            "Écrit le preset Pro-Q4 dynamique décidé pour chaque fichier en vrai .aupreset "
            "dans <sortie>/presets_aupreset/, réutilisable tel quel dans Logic Pro."
        )
        settings_menu.addAction(self.export_presets_action)

        help_menu = menubar.addMenu("&Aide")
        about_action = QAction("À propos de Refinr", self)
        about_action.triggered.connect(self._on_about_clicked)
        help_menu.addAction(about_action)

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header_bar = QHBoxLayout()
        title = QLabel("Refinr")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #ffffff;")
        header_bar.addWidget(title)
        version_label = QLabel(f"v{__version__}")
        version_label.setObjectName("versionLabel")
        header_bar.addWidget(version_label)
        header_bar.addStretch(1)
        layout.addLayout(header_bar)

        top_bar = QHBoxLayout()
        add_btn = QPushButton("Ajouter des fichiers…")
        add_btn.clicked.connect(self._on_add_files_clicked)
        top_bar.addWidget(add_btn)

        clear_btn = QPushButton("Vider la liste")
        clear_btn.clicked.connect(self._on_clear_clicked)
        top_bar.addWidget(clear_btn)

        top_bar.addStretch(1)

        top_bar.addWidget(QLabel("Profil de destination :"))
        self.profile_combo = QComboBox()
        for key, profile in sorted(self.catalog.profiles.items()):
            self.profile_combo.addItem(f"{profile.label}", userData=key)
        self.profile_combo.addItem("Cibles personnalisées…", userData=CUSTOM_PROFILE_KEY)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        top_bar.addWidget(self.profile_combo)

        top_bar.addWidget(QLabel("Workers parallèles :"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        top_bar.addWidget(self.workers_spin)

        layout.addLayout(top_bar)

        # Sliders "Cibles personnalisées" : n'ont d'effet sur le traitement
        # QUE si "Cibles personnalisées…" est sélectionné dans le combo
        # ci-dessus (voir _on_profile_combo_changed) — sinon le profil YAML
        # choisi fait foi et les sliders restent désactivés, en lecture seule.
        custom_bar = QHBoxLayout()
        custom_bar.addWidget(QLabel("Target LUFS :"))
        self.lufs_slider = QSlider(Qt.Orientation.Horizontal)
        self.lufs_slider.setRange(*LUFS_SLIDER_RANGE)
        self.lufs_slider.setValue(LUFS_SLIDER_DEFAULT)
        self.lufs_slider.setEnabled(False)
        self.lufs_value_label = QLabel(f"{LUFS_SLIDER_DEFAULT / 10:.1f}")
        self.lufs_value_label.setFixedWidth(48)
        self.lufs_slider.valueChanged.connect(self._on_custom_targets_changed)
        custom_bar.addWidget(self.lufs_slider, stretch=1)
        custom_bar.addWidget(self.lufs_value_label)

        custom_bar.addSpacing(16)
        custom_bar.addWidget(QLabel("True Peak Limit :"))
        self.true_peak_slider = QSlider(Qt.Orientation.Horizontal)
        self.true_peak_slider.setRange(*TRUE_PEAK_SLIDER_RANGE)
        self.true_peak_slider.setValue(TRUE_PEAK_SLIDER_DEFAULT)
        self.true_peak_slider.setEnabled(False)
        self.true_peak_value_label = QLabel(f"{TRUE_PEAK_SLIDER_DEFAULT / 10:.1f} dB")
        self.true_peak_value_label.setFixedWidth(56)
        self.true_peak_slider.valueChanged.connect(self._on_custom_targets_changed)
        custom_bar.addWidget(self.true_peak_slider, stretch=1)
        custom_bar.addWidget(self.true_peak_value_label)
        layout.addLayout(custom_bar)

        # Sliders kHz/bits de livraison — mêmes règles d'activation que
        # LUFS/true peak ci-dessus, mais valeurs DISCRÈTES (index dans
        # SAMPLE_RATE_OPTIONS/BIT_DEPTH_OPTIONS) plutôt qu'un continuum,
        # puisque seules quelques valeurs standards ont un sens ici.
        format_bar = QHBoxLayout()
        format_bar.addWidget(QLabel("Sample rate :"))
        self.sample_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.sample_rate_slider.setRange(0, len(SAMPLE_RATE_OPTIONS) - 1)
        self.sample_rate_slider.setValue(SAMPLE_RATE_DEFAULT_INDEX)
        self.sample_rate_slider.setEnabled(False)
        self.sample_rate_value_label = QLabel(f"{SAMPLE_RATE_OPTIONS[SAMPLE_RATE_DEFAULT_INDEX] / 1000:g}kHz")
        self.sample_rate_value_label.setFixedWidth(56)
        self.sample_rate_slider.valueChanged.connect(self._on_custom_targets_changed)
        format_bar.addWidget(self.sample_rate_slider, stretch=1)
        format_bar.addWidget(self.sample_rate_value_label)

        format_bar.addSpacing(16)
        format_bar.addWidget(QLabel("Bit depth :"))
        self.bit_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.bit_depth_slider.setRange(0, len(BIT_DEPTH_OPTIONS) - 1)
        self.bit_depth_slider.setValue(BIT_DEPTH_DEFAULT_INDEX)
        self.bit_depth_slider.setEnabled(False)
        self.bit_depth_value_label = QLabel(BIT_DEPTH_OPTIONS[BIT_DEPTH_DEFAULT_INDEX][1])
        self.bit_depth_value_label.setFixedWidth(70)
        self.bit_depth_slider.valueChanged.connect(self._on_custom_targets_changed)
        format_bar.addWidget(self.bit_depth_slider, stretch=1)
        format_bar.addWidget(self.bit_depth_value_label)
        layout.addLayout(format_bar)

        self.custom_targets_warning = QLabel("")
        self.custom_targets_warning.setStyleSheet("color: #d9a441; font-size: 11px;")
        layout.addWidget(self.custom_targets_warning)

        self.library_warning = QLabel("")
        self.library_warning.setStyleSheet("color: #d9a441;")
        layout.addWidget(self.library_warning)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table)

        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 380])
        layout.addWidget(splitter, stretch=1)

        bottom_bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        bottom_bar.addWidget(self.progress, stretch=1)

        self.run_btn = QPushButton("Lancer le batch")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        bottom_bar.addWidget(self.run_btn)

        self.open_report_btn = QPushButton("Ouvrir le rapport")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self._on_open_report_clicked)
        bottom_bar.addWidget(self.open_report_btn)

        self.export_report_btn = QPushButton("Exporter…")
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.clicked.connect(self._on_export_report_clicked)
        bottom_bar.addWidget(self.export_report_btn)

        layout.addLayout(bottom_bar)

    def _build_log_dock(self) -> None:
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(5000)

        self.log_dock = QDockWidget("Logs (mode debug)", self)
        self.log_dock.setWidget(self.log_widget)
        self.log_dock.setVisible(False)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        self._log_handler = _QtLogHandler(self.log_widget)
        # attaché au logger racine 'refinr' uniquement quand le mode debug
        # est activé, pour ne pas polluer les logs hors debug.

    def _build_detail_panel(self) -> QWidget:
        """Panneau d'analyse du fichier sélectionné dans le tableau : infos
        fichier (rate/bits/durée), overlay spectral brut/raffiné façon
        Pro-Q4, goniomètre + corrélation stéréo, et lecteur A/B. Vide tant
        qu'aucune ligne n'est sélectionnée."""
        panel = QFrame()
        panel.setObjectName("detailPanel")
        panel.setMinimumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.detail_title = QLabel("Sélectionne un fichier dans la liste")
        self.detail_title.setStyleSheet("font-weight: 600; color: #f0f0f2;")
        layout.addWidget(self.detail_title)

        self.detail_file_info = FileInfoBar()
        layout.addWidget(self.detail_file_info)

        self.detail_qc_label = QLabel("")
        self.detail_qc_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.detail_qc_label)

        legend = QHBoxLayout()
        raw_dot = QLabel("●")
        raw_dot.setStyleSheet(f"color: {RAW_COLOR.name()};")
        legend.addWidget(raw_dot)
        legend.addWidget(QLabel("Brut"))
        refined_dot = QLabel("●")
        refined_dot.setStyleSheet(f"color: {REFINED_COLOR.name()};")
        legend.addWidget(refined_dot)
        legend.addWidget(QLabel("Raffiné"))
        legend.addStretch(1)
        layout.addLayout(legend)

        self.spectrum_widget = SpectrumCurveWidget()
        layout.addWidget(self.spectrum_widget, stretch=1)

        gonio_row = QHBoxLayout()
        self.goniometer_widget = GoniometerWidget()
        gonio_row.addWidget(self.goniometer_widget)

        meter_col = QVBoxLayout()
        meter_col.addWidget(QLabel("Corrélation stéréo"))
        self.correlation_meter = CorrelationMeterWidget()
        meter_col.addWidget(self.correlation_meter)
        meter_col.addStretch(1)
        gonio_row.addLayout(meter_col, stretch=1)
        layout.addLayout(gonio_row)

        layout.addWidget(QLabel("Largeur stéréo sur la durée du morceau"))
        self.stereo_timeline_widget = StereoWidthTimelineWidget()
        layout.addWidget(self.stereo_timeline_widget)

        self.ab_player = ABPlayerWidget()
        self.ab_player.live_update.connect(self._on_live_update)
        self.ab_player.playback_state_changed.connect(self._on_playback_state_changed)
        self.ab_player.playhead_changed.connect(self._on_playhead_changed)
        layout.addWidget(self.ab_player)

        return panel

    def _clear_detail_panel(self) -> None:
        self.detail_title.setText("Sélectionne un fichier dans la liste")
        self.detail_file_info.clear()
        self.detail_qc_label.setText("")
        self.spectrum_widget.clear()
        self.goniometer_widget.set_points(None)
        self.correlation_meter.set_value(None)
        self.stereo_timeline_widget.clear()
        self.ab_player.stop()
        self.ab_player.set_sources(None, None)
        self._selected_row = None

    def _on_row_selected(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if len(rows) != 1:
            self._clear_detail_panel()
            return
        row = next(iter(rows))
        data = self.row_data.get(row)
        if data is None or data.get("analysis") is None:
            self._clear_detail_panel()
            return

        self._selected_row = row
        self.detail_title.setText(Path(data["path"]).name)

        info = data.get("file_info")
        if info is not None:
            self.detail_file_info.set_info(info.sample_rate, info.bit_depth_label, info.channels, info.duration_seconds)

        ai_score = data.get("ai_score")
        qc_passed = data.get("qc_passed")
        if qc_passed is True:
            self.detail_qc_label.setStyleSheet("color: #5cc26a; font-size: 11px;")
            qc_text = "✓ QC passé"
        elif qc_passed is False:
            self.detail_qc_label.setStyleSheet("color: #e05c5c; font-size: 11px;")
            qc_text = "✗ QC échoué"
        else:
            self.detail_qc_label.setStyleSheet("color: #6f7280; font-size: 11px;")
            qc_text = "QC : pas encore traité"
        if ai_score is not None:
            qc_text += f"   ·   Score suspicion IA : {ai_score:.1f}/10"
        self.detail_qc_label.setText(qc_text)

        # Courbes "statiques" (fichier entier) — référence par défaut, et ce
        # sur quoi on revient à l'arrêt de la lecture (voir _on_playback_state_changed).
        self._static_spectrum = (data.get("spectrum_raw"), data.get("spectrum_refined"))
        self._static_goniometer = (
            (data["goniometer_refined"], REFINED_COLOR)
            if data.get("goniometer_refined") is not None
            else (data.get("goniometer_raw"), RAW_COLOR)
        )
        self._static_correlation = data.get("correlation_refined")
        if self._static_correlation is None and data.get("analysis") is not None:
            self._static_correlation = data["analysis"].dynamics.stereo_correlation
        self._restore_static_analyzer_view()

        self.stereo_timeline_widget.set_timeline(data.get("stereo_timeline_raw"), data.get("stereo_timeline_refined"))
        self.stereo_timeline_widget.set_playhead_fraction(None)

        self.ab_player.set_sources(
            data["path"],
            data.get("output_path"),
            raw_buffer=data.get("raw_buffer"),
            refined_buffer=data.get("refined_buffer"),
            raw_lufs=data["analysis"].loudness.integrated_lufs,
            refined_lufs=data.get("refined_lufs"),
        )

    def _restore_static_analyzer_view(self) -> None:
        self.spectrum_widget.set_curves(*self._static_spectrum)
        points, color = self._static_goniometer
        self.goniometer_widget.set_points(points, color)
        self.correlation_meter.set_value(self._static_correlation)

    def _on_live_update(self, spectrum, gonio_points, correlation: float, is_refined: bool) -> None:
        """Pendant la lecture (voir ABPlayerWidget.live_update) : remplace
        SEULEMENT le côté (brut ou raffiné) actuellement écouté par la
        fenêtre live, garde l'autre côté en référence statique — laisse le
        goniomètre/corrélation suivre la fenêtre en cours d'écoute."""
        static_raw, static_refined = self._static_spectrum
        if is_refined:
            self.spectrum_widget.set_curves(static_raw, spectrum)
        else:
            self.spectrum_widget.set_curves(spectrum, static_refined)
        self.goniometer_widget.set_points(gonio_points, REFINED_COLOR if is_refined else RAW_COLOR)
        self.correlation_meter.set_value(correlation)

    def _on_playback_state_changed(self, is_playing: bool) -> None:
        if not is_playing:
            self._restore_static_analyzer_view()
            self.stereo_timeline_widget.set_playhead_fraction(None)

    def _on_playhead_changed(self, fraction: float, _is_refined: bool) -> None:
        # Position dans le morceau identique quel que soit le côté écouté
        # (brut/raffiné) : le curseur avance à la même fraction sur les
        # deux courbes superposées de la timeline.
        self.stereo_timeline_widget.set_playhead_fraction(fraction)

    def _on_profile_combo_changed(self) -> None:
        is_custom = self.profile_combo.currentData() == CUSTOM_PROFILE_KEY
        self.lufs_slider.setEnabled(is_custom)
        self.true_peak_slider.setEnabled(is_custom)
        self.sample_rate_slider.setEnabled(is_custom)
        self.bit_depth_slider.setEnabled(is_custom)
        self._on_custom_targets_changed()

    def _on_custom_targets_changed(self, _value: int | None = None) -> None:
        """
        Avertissement non-bloquant si les cibles personnalisées sortent des
        pratiques standards de diffusion (trop fort/trop faible en LUFS,
        true peak trop proche de 0dBTP -> risque d'intersample clipping
        après réencodage lossy côté plateforme) — l'utilisateur reste libre
        de forcer ces valeurs, on prévient juste du risque.
        """
        lufs = self.lufs_slider.value() / 10.0
        true_peak = self.true_peak_slider.value() / 10.0
        self.lufs_value_label.setText(f"{lufs:.1f}")
        self.true_peak_value_label.setText(f"{true_peak:.1f} dB")

        sample_rate = SAMPLE_RATE_OPTIONS[self.sample_rate_slider.value()]
        bit_depth_code, bit_depth_label = BIT_DEPTH_OPTIONS[self.bit_depth_slider.value()]
        self.sample_rate_value_label.setText(f"{sample_rate / 1000:g}kHz")
        self.bit_depth_value_label.setText(bit_depth_label)

        if self.profile_combo.currentData() != CUSTOM_PROFILE_KEY:
            self.custom_targets_warning.setText("")
            return

        warnings = []
        if lufs > LUFS_WARN_LOUDER_THAN:
            warnings.append(
                f"{lufs:.1f} LUFS est très fort — presque aucune plateforme ne vise ça, limiting agressif probable"
            )
        elif lufs < LUFS_WARN_QUIETER_THAN:
            warnings.append(
                f"{lufs:.1f} LUFS est très bas — la plupart des plateformes vont sur-normaliser à la lecture"
            )
        if true_peak > TRUE_PEAK_WARN_ABOVE:
            warnings.append(
                f"true peak {true_peak:+.1f}dBTP est proche de 0 — risque d'intersample clipping après ré-encodage"
            )

        self.custom_targets_warning.setText("⚠ " + " · ".join(warnings) if warnings else "")

    def _refresh_library_warning(self) -> None:
        missing_roles = [role.value for role in (PluginRole.SATURATION, PluginRole.TAPE) if self.library.is_empty(role)]
        if missing_roles:
            self.library_warning.setText(
                f"⚠ Aucun preset trouvé pour: {', '.join(missing_roles)}. "
                f"Ajoute des .aupreset dans config/presets/<role>/ (voir config/presets/README.md). "
                f"L'EQ Pro-Q4 est pilotée automatiquement, aucun preset requis pour ce rôle."
            )
        else:
            self.library_warning.setText("")

    # ----------------------------------------------------------- Drag&Drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith(".wav")]
        self._add_files(paths)

    # ------------------------------------------------------------ Actions

    def _on_debug_toggled(self, checked: bool) -> None:
        self.log_dock.setVisible(checked)
        root_logger = logging.getLogger("refinr")
        if checked:
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(self._log_handler)
            logging.getLogger("refinr.gui").debug("Mode debug activé.")
        else:
            root_logger.removeHandler(self._log_handler)

    def _on_about_clicked(self) -> None:
        QMessageBox.information(
            self,
            "À propos de Refinr",
            f"Refinr v{__version__}\n\n"
            "Retraitement audio WAV via plugins Audio Unit locaux "
            "(FabFilter Pro-Q4 piloté dynamiquement, Saturn2/HG2, Waves J37).\n\n"
            "github.com/syb-illin/refinr",
        )

    def _on_add_files_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choisir des WAV", "", "WAV (*.wav)")
        self._add_files(paths)

    def _on_clear_clicked(self) -> None:
        self.table.setRowCount(0)
        self.row_data.clear()
        self._clear_detail_panel()

    def _add_files(self, paths: list[str]) -> None:
        for path in paths:
            self._add_one_file(path)

    def _add_one_file(self, path: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(Path(path).name))

        try:
            buffer = load_wav(path)
            file_analysis = analyze(buffer)
            eq_bands = decide_bands(file_analysis, suno_mode=self.suno_mode_action.isChecked())
            selection = select_chain(self.library, file_analysis, roles=(PluginRole.SATURATION, PluginRole.TAPE))

            # Données pour le panneau d'analyse (overlay spectral/goniomètre/
            # infos fichier) — calculées une fois à l'ajout, pas à chaque
            # sélection de ligne, pour rester réactif sur de gros batches.
            spectrum_raw = compute_spectrum_db(buffer)
            goniometer_raw = compute_goniometer_points(buffer)
            stereo_timeline_raw = compute_correlation_timeline(buffer)
            info = file_info(path)

            tags_text = ", ".join(file_analysis.summary_tags())
            if eq_bands:
                eq_text = f"{len(eq_bands)} bande(s) auto"
                eq_tooltip = "\n".join(
                    f"{b.shape} {b.freq_hz:.0f}Hz" + (f" {b.gain_db:+.1f}dB" if b.gain_db else "") for b in eq_bands
                )
            else:
                eq_text = "aucune correction"
                eq_tooltip = "Diagnostic dans les limites normales."
            sat_text = selection[PluginRole.SATURATION].preset.name if selection[PluginRole.SATURATION].preset else "—"
            tape_text = selection[PluginRole.TAPE].preset.name if selection[PluginRole.TAPE].preset else "—"

            self.table.setItem(row, 1, QTableWidgetItem(tags_text))
            eq_item = QTableWidgetItem(eq_text)
            eq_item.setToolTip(eq_tooltip)
            self.table.setItem(row, 2, eq_item)
            self.table.setItem(row, 3, QTableWidgetItem(sat_text))
            self.table.setItem(row, 4, QTableWidgetItem(tape_text))
            self.table.setItem(row, 5, QTableWidgetItem("Prêt"))

            self.row_data[row] = {
                "path": path,
                "analysis": file_analysis,
                "eq_bands": eq_bands,
                "selection": selection,
                "spectrum_raw": spectrum_raw,
                "spectrum_refined": None,
                "goniometer_raw": goniometer_raw,
                "goniometer_refined": None,
                "correlation_refined": None,
                "stereo_timeline_raw": stereo_timeline_raw,
                "stereo_timeline_refined": None,
                "file_info": info,
                "output_path": None,
                "raw_buffer": buffer,  # gardé en mémoire pour l'analyse "live" pendant la lecture A/B
                "refined_buffer": None,
                "refined_lufs": None,
                "ai_score": None,
                "qc_passed": None,
            }
            self.logger.debug("Fichier ajouté: %s (%d bande(s) EQ auto)", path, len(eq_bands))
        except Exception as exc:  # noqa: BLE001
            for col in range(1, 5):
                self.table.setItem(row, col, QTableWidgetItem("—"))
            self.table.setItem(row, 5, QTableWidgetItem(f"Erreur analyse: {exc}"))
            self.row_data[row] = {"path": path, "analysis": None, "eq_bands": None, "selection": None}
            self.logger.exception("Erreur d'analyse sur %s", path)

    def _row_for_path(self, path: str) -> int | None:
        for row, data in self.row_data.items():
            if data["path"] == path:
                return row
        return None

    def _on_run_clicked(self) -> None:
        if not self.row_data:
            QMessageBox.information(self, "Rien à faire", "Ajoute d'abord des fichiers WAV.")
            return

        if self.worker is not None and self.worker.isRunning():
            # Protection contre un double-lancement (double-clic, ou clic
            # pendant qu'un batch précédent tourne encore) : un nouveau
            # ProcessPoolExecutor par-dessus l'ancien doublerait la charge
            # CPU pour rien et compliquerait le rapport final.
            QMessageBox.information(
                self, "Batch en cours", "Un traitement est déjà en cours, attends qu'il se termine."
            )
            return

        profile_key = self.profile_combo.currentData()
        input_paths = [data["path"] for data in self.row_data.values()]

        custom_profile = None
        if profile_key == CUSTOM_PROFILE_KEY:
            bit_depth_code, _label = BIT_DEPTH_OPTIONS[self.bit_depth_slider.value()]
            custom_profile = {
                "target_lufs": self.lufs_slider.value() / 10.0,
                "true_peak_ceiling_dbtp": self.true_peak_slider.value() / 10.0,
                "output_sample_rate": SAMPLE_RATE_OPTIONS[self.sample_rate_slider.value()],
                "output_bit_depth": bit_depth_code,
            }
            profile_key = "custom"  # nom de fichier de sortie, voir batch._process_one

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.open_report_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
        self.open_report_action.setEnabled(False)
        self.export_report_action.setEnabled(False)
        self.progress.setMaximum(len(input_paths))
        self.progress.setValue(0)

        self.logger.info("Lancement du batch sur %d fichier(s), profil=%s", len(input_paths), profile_key)

        self.worker = BatchWorker(
            input_paths=input_paths,
            output_dir=str(DEFAULT_OUTPUT_DIR),
            presets_root=str(DEFAULT_PRESETS_ROOT),
            profile_key=profile_key,
            profiles_path=str(DEFAULT_PROFILES_PATH),
            max_workers=self.workers_spin.value(),
            suno_mode=self.suno_mode_action.isChecked(),
            export_eq_presets=self.export_presets_action.isChecked(),
            custom_profile=custom_profile,
        )
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_ok.connect(self._on_batch_finished)
        self.worker.failed.connect(self._on_batch_failed)
        self.worker.start()

    def _on_file_done(self, path: str, success: bool, message: str) -> None:
        row = self._row_for_path(path)
        if row is not None:
            self.table.setItem(row, 5, QTableWidgetItem(message))
        self.progress.setValue(self.progress.value() + 1)
        self.logger.debug("Terminé: %s -> %s", path, message)

    def _on_batch_finished(self, result, report_paths: dict) -> None:
        self.run_btn.setEnabled(True)
        self._last_report_paths = report_paths
        has_report = report_paths.get("html_summary") is not None
        self.open_report_btn.setEnabled(has_report)
        self.export_report_btn.setEnabled(has_report)
        self.open_report_action.setEnabled(has_report)
        self.export_report_action.setEnabled(has_report)
        self.logger.info("Batch terminé: %d réussi(s), %d échec(s)", len(result.succeeded), len(result.failed))

        self._start_refined_analysis(result)

        QMessageBox.information(
            self,
            "Batch terminé",
            f"{len(result.succeeded)} fichier(s) réussi(s), {len(result.failed)} échec(s).\n"
            f"Sortie: {DEFAULT_OUTPUT_DIR}",
        )

    def _start_refined_analysis(self, result) -> None:
        """Lance en arrière-plan (QThread dédié, voir gui/worker.py) le
        recalcul des courbes spectrales/goniomètre/LUFS/QC "raffiné" pour
        chaque fichier réussi — PAS sur le thread GUI, pour ne pas geler la
        fenêtre en rechargeant/analysant chaque sortie une par une sur un
        gros batch."""
        succeeded = result.succeeded
        if not succeeded:
            return
        self.refined_analysis_worker = RefinedAnalysisWorker(succeeded)
        self.refined_analysis_worker.file_analyzed.connect(self._on_refined_file_analyzed)
        self.refined_analysis_worker.start()

    def _on_refined_file_analyzed(self, input_path: str, data: dict) -> None:
        row = self._row_for_path(input_path)
        if row is None:
            return
        self.row_data[row]["output_path"] = data["output_path"]
        self.row_data[row]["refined_buffer"] = data["buffer"]
        self.row_data[row]["spectrum_refined"] = data["spectrum"]
        self.row_data[row]["goniometer_refined"] = data["goniometer"]
        self.row_data[row]["stereo_timeline_refined"] = data["stereo_timeline"]
        self.row_data[row]["correlation_refined"] = data["correlation"]
        self.row_data[row]["refined_lufs"] = data["lufs"]
        self.row_data[row]["ai_score"] = data["ai_score"]
        self.row_data[row]["qc_passed"] = data["qc_passed"]

        if row == self._selected_row:
            self._on_row_selected()  # rafraîchit le panneau si cette ligne est actuellement affichée

    def _on_batch_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.logger.error("Échec du batch: %s", message)
        QMessageBox.critical(self, "Erreur batch", message)

    def _on_open_report_clicked(self) -> None:
        html_path = (self._last_report_paths or {}).get("html_summary")
        if html_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    # ------------------------------------------------------------ Cycle de vie

    def closeEvent(self, event) -> None:  # noqa: N802 (override Qt)
        """
        Fermeture propre : si un batch tourne encore, on attend sa fin (avec
        timeout) plutôt que de laisser un QThread/ProcessPoolExecutor orphelin
        derrière nous — source classique de crash silencieux ou de process
        zombies sur macOS si on quitte brutalement pendant un traitement.
        """
        if self.worker is not None and self.worker.isRunning():
            self.logger.info("Fermeture demandée pendant un batch en cours — attente de fin (max 5s)...")
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self.logger.warning("Le worker n'a pas terminé dans les temps, arrêt forcé.")
                self.worker.terminate()
                self.worker.wait(1000)

        if (
            self.refined_analysis_worker is not None
            and self.refined_analysis_worker.isRunning()
            and not self.refined_analysis_worker.wait(3000)
        ):
            self.refined_analysis_worker.terminate()
            self.refined_analysis_worker.wait(1000)

        # Désinscrit le handler de log AVANT que le widget Qt ne soit détruit :
        # sinon un log émis après fermeture (ex: depuis un thread encore en
        # train de finir) planterait en écrivant dans un objet Qt mort.
        root_logger = logging.getLogger("refinr")
        if self._log_handler in root_logger.handlers:
            root_logger.removeHandler(self._log_handler)

        super().closeEvent(event)

    def _on_export_report_clicked(self) -> None:
        html_path = (self._last_report_paths or {}).get("html_summary")
        if not html_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Exporter le rapport", str(Path.home() / "refinr_rapport.html"), "HTML (*.html)"
        )
        if not dest:
            return
        try:
            shutil.copyfile(html_path, dest)
            self.logger.info("Rapport exporté vers %s", dest)
            QMessageBox.information(self, "Export réussi", f"Rapport exporté vers :\n{dest}")
        except OSError as exc:
            self.logger.exception("Échec export du rapport")
            QMessageBox.critical(self, "Erreur export", str(exc))
