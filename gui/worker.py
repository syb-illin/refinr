"""
Worker QThread pour lancer le batch en arrière-plan sans geler l'UI.

Le vrai parallélisme du traitement se fait dans `refinr.batch`
(ProcessPoolExecutor). Ce QThread sert uniquement à ne pas bloquer la
boucle d'événements Qt pendant que ce batch tourne, et à relayer la
progression via des signaux.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refinr.batch import BatchResult, run_batch  # noqa: E402
from refinr.report import write_reports  # noqa: E402


class BatchWorker(QThread):
    file_done = pyqtSignal(str, bool, str)  # input_path, success, message court
    finished_ok = pyqtSignal(object, dict)  # BatchResult, chemins des rapports
    failed = pyqtSignal(str)

    def __init__(
        self,
        input_paths: list[str],
        output_dir: str,
        presets_root: str,
        profile_key: str,
        profiles_path: str,
        max_workers: int,
    ):
        super().__init__()
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.presets_root = presets_root
        self.profile_key = profile_key
        self.profiles_path = profiles_path
        self.max_workers = max_workers

    def run(self) -> None:
        try:

            def _on_progress(outcome):
                if outcome.success:
                    lufs = outcome.report.final_measurement.get("integrated_lufs")
                    msg = f"OK — {lufs} LUFS final" if lufs is not None else "OK"
                else:
                    msg = "ÉCHEC"
                self.file_done.emit(outcome.input_path, outcome.success, msg)

            result: BatchResult = run_batch(
                input_paths=self.input_paths,
                output_dir=self.output_dir,
                presets_root=self.presets_root,
                profile_key=self.profile_key,
                profiles_path=self.profiles_path,
                max_workers=self.max_workers,
                on_progress=_on_progress,
            )
            report_paths = write_reports(result, self.output_dir, self.profile_key)
            self.finished_ok.emit(result, report_paths)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
