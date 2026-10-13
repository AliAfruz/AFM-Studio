"""Background jobs with immutable inputs; widgets are touched only by UI slots."""
from __future__ import annotations
from copy import deepcopy
from PySide6.QtCore import QThread, Signal
from batch_analysis import BatchResult, analyze_file


class BatchWorker(QThread):
    file_started = Signal(int, str)
    file_finished = Signal(int, object)

    def __init__(self, paths, settings, parent=None):
        super().__init__(parent)
        self.paths = tuple(paths)
        self.settings = deepcopy(settings)
        self.cancelled = False

    def run(self):
        for index, path in enumerate(self.paths, start=1):
            if self.isInterruptionRequested():
                self.cancelled = True
                break
            self.file_started.emit(index, path)
            try:
                result = analyze_file(path, self.settings)
            except Exception as exc:
                from pathlib import Path
                result = BatchResult(str(path), Path(path).name, "Error", error=str(exc))
            self.file_finished.emit(index, result)
        self.cancelled = self.cancelled or self.isInterruptionRequested()
