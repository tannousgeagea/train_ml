import os
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log("=== Training started: {} ===".format(datetime.now()))

    def _log(self, msg: str):
        with open(self.log_file, "a") as f:
            f.write(msg.strip() + "\n")
        print(msg)

    def _tail(self, n=10):
        try:
            with open(self.log_file, "r") as f:
                return "".join(f.readlines()[-n:])
        except:
            return ""

    def info(self, msg: str):
        self._log(f"[INFO] {msg}")

    def progress(self, percent: float):
        self._log(f"[PROGRESS] {percent:.1f}%")

    def epoch(self, epoch_num: int, metrics: dict):
        self._log(f"[EPOCH {epoch_num}] {metrics}")

    def error(self, msg: str):
        self._log(f"[ERROR] {msg}")

    def complete(self):
        self._log("=== Training completed ===")
