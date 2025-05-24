import os
from pathlib import Path
from common_utils.ml_models.core.base import AbstractBaseModel
from ultralytics import YOLO
from ultralytics.utils import downloads

def no_download_asset(path, *args, **kwargs):
    print(f"❌ [BLOCKED] attempt_download_asset('{path}')")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found locally: {path}")
    return path

downloads.attempt_download_asset = no_download_asset
os.environ["YOLO_CONFIG_DIR"] = "~/src/ultralytics_config"

class YOLOTrainer(AbstractBaseModel):
    def __init__(self, weights, dataset, config, logger, on_event=None):
        super().__init__(weights, dataset, config, logger, on_event)
        self.model = None
        self.results = {}

    def init_model(self):
        if not os.path.exists(self.weights):
            raise FileExistsError(f"Model not found: {self.weights}")

        return YOLO(self.weights)

    def train(self,):
        try:
            self.model = self.init_model()
            def on_epoch_end(trainer):
                self.logger.epoch(trainer.epoch + 1, trainer.metrics)
                if self.on_event:
                    self.on_event(
                        event={
                            "type": "epoch_end",
                            "epoch": trainer.epoch + 1,
                            "progress": (trainer.epoch + 1) / trainer.epochs * 100,
                            "metrics": trainer.metrics,
                            "logs": f"{trainer.epoch + 1} / {trainer.epochs}: {trainer.metrics}",
                        }
                    )
            
            self.model.add_callback("on_fit_epoch_end", on_epoch_end)
            self.results = self.model.train(
                data=self.dataset,
                imgsz=self.config.get("imgsz", 640),
                optimizer=self.config.get("optimizer", "SGD"),
                lr0=self.config.get("lr0", 0.001),
                lrf=self.config.get("lrf", 0.00001),
                epochs=self.config.get("epochs", 100),
                batch=self.config.get("batch", 16),
                augment=self.config.get("augment", False),
                name=self.config.get("name", "Yolo"),
                workers=0,
                project= '/media/runs/train',
            )

            self.logger.complete()
            if self.on_event:
                self.on_event({
                    "type": "complete",
                    "progress": 100.0,
                    "metrics": self.results.get("metrics", {}),
                    "status": "completed",
                    "logs": "=== Training completed ===",
                })

            return self.results
        except Exception as e:
            self.logger.error(str(e))
            if self.on_event:
                self.on_event({
                    "type": "error",
                    "status": "failed",
                    "error_message": str(e)
                })
            raise e

    def get_results(self):
        return {
            "metrics": self.results.get("metrics", {}),
            "artifacts": {
                "weights": self.weights_path,
                "logs": self.log_path
            }
        }

