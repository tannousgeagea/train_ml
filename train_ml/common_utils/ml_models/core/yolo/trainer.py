import os
import shutil
import pandas as pd
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
        self.save_dir = None
        self.results = {}

    def init_model(self):
        if not os.path.exists(self.weights):
            raise FileExistsError(f"Model not found: {self.weights}")

        return YOLO(self.weights)

    def train(self,):
        try:
            self.model = self.init_model()
            def on_epoch_end(trainer):
                if not self.save_dir:
                    self.save_dir = trainer.save_dir
    
                self.logger.epoch(trainer.epoch + 1, trainer.metrics)
                if self.on_event:
                    raw_metrics = self.get_train_results_csv(csv_file=Path(trainer.save_dir) / "results.csv")
                    metrics_summary = {
                        "iou": round(raw_metrics.get("metrics/box", 0.0), 3),
                        "mAP": round(raw_metrics.get("metrics/mAP50(B)", 0.0), 3),
                        "mAP50-95": round(raw_metrics.get("metrics/mAP50-95(B)", 0.0), 3),
                        "recall": round(raw_metrics.get("metrics/recall(B)", 0.0), 3),
                        "precision": round(raw_metrics.get("metrics/precision(B)", 0.0), 3),
                        "val/box_loss": round(raw_metrics.get("val/box_loss", 0.0), 3),
                        "val/cls_loss": round(raw_metrics.get("val/cls_loss", 0.0), 3),
                        "val/dfl_loss": round(raw_metrics.get("val/dfl_loss", 0.0), 3),
                        "train/box_loss": round(raw_metrics.get("train/box_loss", 0.0), 3),
                        "train/cls_loss": round(raw_metrics.get("train/cls_loss", 0.0), 3),
                        "train/dfl_loss": round(raw_metrics.get("train/dfl_loss", 0.0), 3),
                        "epoch": raw_metrics.get("epoch"),
                        "time": round(raw_metrics.get("time", 0.0), 3),
                    }
                    
                    self.on_event(
                        event={
                            "type": "epoch_end",
                            "epoch": trainer.epoch + 1,
                            "progress": (trainer.epoch + 1) / trainer.epochs * 100,
                            "metrics": metrics_summary,
                            "logs": f"Epoch {trainer.epoch + 1} / {trainer.epochs}: {trainer.metrics}",
                        }
                    )
            
            self.model.add_callback("on_fit_epoch_end", on_epoch_end)
            self.results = self.model.train(
                data=self.dataset,
                imgsz=self.config.get("imgsz", 640),
                optimizer=self.config.get("optimizer", "SGD"),
                lr0=self.config.get("lr0", 0.001) or self.config.get('learning_rate'),
                lrf=self.config.get("lrf", 0.00001),
                epochs=self.config.get("epochs", 100),
                batch=self.config.get("batch", 16) or self.config.get("batch_size"),
                augment=self.config.get("augment", False),
                name=self.config.get("name", "Yolo"),
                workers=0,
                project= '/media/runs/train',
            )

            self.logger.complete()

            raw_metrics = self.results.results_dict
            metrics_summary = {
                "iou": round(raw_metrics.get("metrics/box", 0.0), 3),
                "mAP": round(raw_metrics.get("metrics/mAP50(B)", 0.0), 3),
                "mAP50-95": round(raw_metrics.get("metrics/mAP50-95(B)", 0.0), 3),
                "recall": round(raw_metrics.get("metrics/recall(B)", 0.0), 3),
                "precision": round(raw_metrics.get("metrics/precision(B)", 0.0), 3),
            }

            if self.on_event:
                self.on_event({
                    "type": "complete",
                    "progress": 100.0,
                    "metrics": metrics_summary,
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


    def get_train_results_csv(self, csv_file:Path):
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            return df.to_dict(orient="records")[-1]
        return {}
    
    def get_results(self):
        return {
            "metrics": self.results.results_dict.get("metrics", {}),
            "artifacts": {
                "weights": self.weights_path,
                "logs": self.log_path
            }
        }
    
    def save(self, output_file:str):
        best_ckpt = Path(self.save_dir / "weights" / "best.pt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if best_ckpt.exists():
            shutil.copy(best_ckpt, output_file)
            return output_file

