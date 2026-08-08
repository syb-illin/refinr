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
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
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

from .worker import BatchWorker

DEFAULT_PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"
DEFAULT_OUTPUT_DIR = Path.home() / "Refinr_Output"

COLUMNS = ["Fichier", "Tags analyse", "EQ Pro-Q4 (auto)", "Saturation", "Tape (J37)", "Statut"]

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
        self._last_report_paths: dict | None = None

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
        top_bar.addWidget(self.profile_combo)

        top_bar.addWidget(QLabel("Workers parallèles :"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        top_bar.addWidget(self.workers_spin)

        layout.addLayout(top_bar)

        self.library_warning = QLabel("")
        self.library_warning.setStyleSheet("color: #d9a441;")
        layout.addWidget(self.library_warning)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

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
        QMessageBox.information(
            self,
            "Batch terminé",
            f"{len(result.succeeded)} fichier(s) réussi(s), {len(result.failed)} échec(s).\n"
            f"Sortie: {DEFAULT_OUTPUT_DIR}",
        )

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
