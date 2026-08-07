"""
Fenêtre principale de l'app.

Flux :
  1. Glisser-déposer (ou "Ajouter des fichiers...") des WAV.
  2. Pour chaque fichier ajouté, on calcule immédiatement son analyse et la
     preview des presets qui seraient choisis (EQ/Saturation/Tape) — visible
     et modifiable avant de lancer quoi que ce soit (traitement JAMAIS
     générique : chaque ligne peut avoir une chaîne différente).
  3. Choix du profil de destination (Spotify, YouTube, Apple Music, ...).
  4. "Lancer le batch" traite tous les fichiers en parallèle
     (refinr.batch, process pool), avec barre de progression et statut
     par ligne.
  5. À la fin : lien vers le rapport HTML détaillé + dossier de sortie.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from refinr.analysis import analyze
from refinr.audio_io import load_wav
from refinr.preset_mapping import PresetLibrary, select_chain
from refinr.preset_types import PluginRole
from refinr.profiles import DEFAULT_PROFILES_PATH, load_profiles

from .worker import BatchWorker

DEFAULT_PRESETS_ROOT = Path(__file__).resolve().parent.parent / "config" / "presets"
DEFAULT_OUTPUT_DIR = Path.home() / "Refinr_Output"

COLUMNS = ["Fichier", "Tags analyse", "EQ (Pro-Q4)", "Saturation", "Tape (J37)", "Statut"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Refinr — retraitement WAV via AU")
        self.resize(1150, 620)
        self.setAcceptDrops(True)

        self.library = PresetLibrary.load(DEFAULT_PRESETS_ROOT)
        self.catalog = load_profiles(DEFAULT_PROFILES_PATH)

        self.row_data: dict[int, dict] = {}  # row -> {"path": ..., "analysis":..., "selection":...}
        self.worker: BatchWorker | None = None

        self._build_ui()
        self._refresh_library_warning()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

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
        self.library_warning.setStyleSheet("color: #a86500;")
        layout.addWidget(self.library_warning)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        bottom_bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        bottom_bar.addWidget(self.progress, stretch=1)

        self.run_btn = QPushButton("Lancer le batch")
        self.run_btn.clicked.connect(self._on_run_clicked)
        bottom_bar.addWidget(self.run_btn)

        self.open_report_btn = QPushButton("Ouvrir le rapport")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self._on_open_report_clicked)
        bottom_bar.addWidget(self.open_report_btn)

        layout.addLayout(bottom_bar)

        self._last_report_path: str | None = None

    def _refresh_library_warning(self) -> None:
        missing_roles = [
            role.value for role in (PluginRole.EQ, PluginRole.SATURATION, PluginRole.TAPE)
            if self.library.is_empty(role)
        ]
        if missing_roles:
            self.library_warning.setText(
                f"⚠ Aucun preset trouvé pour: {', '.join(missing_roles)}. "
                f"Ajoute des .aupreset dans config/presets/<role>/ (voir config/presets/README.md)."
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
            selection = select_chain(self.library, file_analysis)

            tags_text = ", ".join(file_analysis.summary_tags())
            eq_text = selection[PluginRole.EQ].preset.name if selection[PluginRole.EQ].preset else "—"
            sat_text = selection[PluginRole.SATURATION].preset.name if selection[PluginRole.SATURATION].preset else "—"
            tape_text = selection[PluginRole.TAPE].preset.name if selection[PluginRole.TAPE].preset else "—"

            self.table.setItem(row, 1, QTableWidgetItem(tags_text))
            self.table.setItem(row, 2, QTableWidgetItem(eq_text))
            self.table.setItem(row, 3, QTableWidgetItem(sat_text))
            self.table.setItem(row, 4, QTableWidgetItem(tape_text))
            self.table.setItem(row, 5, QTableWidgetItem("Prêt"))

            self.row_data[row] = {"path": path, "analysis": file_analysis, "selection": selection}
        except Exception as exc:  # noqa: BLE001
            for col in range(1, 5):
                self.table.setItem(row, col, QTableWidgetItem("—"))
            self.table.setItem(row, 5, QTableWidgetItem(f"Erreur analyse: {exc}"))
            self.row_data[row] = {"path": path, "analysis": None, "selection": None}

    def _row_for_path(self, path: str) -> int | None:
        for row, data in self.row_data.items():
            if data["path"] == path:
                return row
        return None

    def _on_run_clicked(self) -> None:
        if not self.row_data:
            QMessageBox.information(self, "Rien à faire", "Ajoute d'abord des fichiers WAV.")
            return

        profile_key = self.profile_combo.currentData()
        input_paths = [data["path"] for data in self.row_data.values()]

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.open_report_btn.setEnabled(False)
        self.progress.setMaximum(len(input_paths))
        self.progress.setValue(0)

        self.worker = BatchWorker(
            input_paths=input_paths,
            output_dir=str(DEFAULT_OUTPUT_DIR),
            presets_root=str(DEFAULT_PRESETS_ROOT),
            profile_key=profile_key,
            profiles_path=str(DEFAULT_PROFILES_PATH),
            max_workers=self.workers_spin.value(),
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

    def _on_batch_finished(self, result, report_paths: dict) -> None:
        self.run_btn.setEnabled(True)
        self._last_report_path = report_paths.get("html_summary")
        self.open_report_btn.setEnabled(self._last_report_path is not None)
        QMessageBox.information(
            self,
            "Batch terminé",
            f"{len(result.succeeded)} fichier(s) réussi(s), {len(result.failed)} échec(s).\n"
            f"Sortie: {DEFAULT_OUTPUT_DIR}",
        )

    def _on_batch_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Erreur batch", message)

    def _on_open_report_clicked(self) -> None:
        if self._last_report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_report_path))
