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

from refinr.analysis import analyze  # noqa: E402
from refinr.audio_io import load_wav  # noqa: E402
from refinr.batch import BatchResult, run_batch  # noqa: E402
from refinr.report import write_reports  # noqa: E402
from refinr.spectrum import compute_correlation_timeline, compute_goniometer_points, compute_spectrum_db  # noqa: E402


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
        suno_mode: bool = False,
        export_eq_presets: bool = False,
        custom_profile: dict | None = None,
    ):
        super().__init__()
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.presets_root = presets_root
        self.profile_key = profile_key
        self.profiles_path = profiles_path
        self.max_workers = max_workers
        self.suno_mode = suno_mode
        self.export_eq_presets = export_eq_presets
        self.custom_profile = custom_profile

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
                suno_mode=self.suno_mode,
                export_eq_presets=self.export_eq_presets,
                custom_profile=self.custom_profile,
            )
            report_paths = write_reports(result, self.output_dir, self.profile_key)
            self.finished_ok.emit(result, report_paths)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class RefinedAnalysisWorker(QThread):
    """
    Recharge chaque WAV de sortie d'un batch pour calculer les données du
    panneau d'analyse "raffiné" (courbe spectrale, goniomètre, corrélation,
    LUFS) — dans un QThread séparé plutôt que dans `_on_batch_finished`
    directement, pour ne pas geler l'UI en rechargeant/analysant chaque
    fichier de sortie un par un sur le thread principal (notable sur un
    batch de plusieurs dizaines de fichiers).
    """

    file_analyzed = pyqtSignal(
        str, dict
    )  # input_path, {"spectrum":..., "goniometer":..., "correlation":..., "lufs":...}
    finished_all = pyqtSignal()

    def __init__(self, outcomes: list):
        super().__init__()
        self.outcomes = outcomes  # list[BatchFileOutcome], réussis uniquement (filtré par l'appelant)

    def run(self) -> None:
        for outcome in self.outcomes:
            if outcome.report is None:
                continue
            try:
                output_buffer = load_wav(outcome.report.output_path)
            except OSError:
                continue
            refined_analysis = analyze(output_buffer)
            data = {
                "output_path": outcome.report.output_path,
                "buffer": output_buffer,  # gardé pour l'analyse "live" pendant la lecture A/B
                "spectrum": compute_spectrum_db(output_buffer),
                "goniometer": compute_goniometer_points(output_buffer),
                "stereo_timeline": compute_correlation_timeline(output_buffer),
                "correlation": refined_analysis.dynamics.stereo_correlation,
                "lufs": refined_analysis.loudness.integrated_lufs,
                "ai_score": outcome.report.output_diagnostic.get("ai_score"),
                "qc_passed": outcome.report.qc_passed,
            }
            self.file_analyzed.emit(outcome.input_path, data)
        self.finished_all.emit()
