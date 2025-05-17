import os
from pathlib import Path
from common_utils.ml_models.core.base import AbstractBaseModel
from ultralytics import YOLO


os.environ["YOLO_CONFIG_DIR"] = "~/src/ultralytics_config"
class YOLOModel(AbstractBaseModel):
    def __init__(self, weights, dataset, config):
        super().__init__(weights, dataset, config)
        self.model = None
        self.results = {}
        self.log_path = "/media/models/logs/yolo.log"

    def init_model(self):
        if not os.path.exists(self.weights):
            raise FileExistsError(f"Model not found: {self.weights}")

        return YOLO(self.weights)

    def train(self):
        self.model = self.init_model()

        log_path = Path(self.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, "w") as f:
            f.write(f"[YOLOPlugin] Training started \n")

        def epoch_logger(trainer):
            epoch = trainer.epoch
            metrics = trainer.metrics
            with open(self.log_path, "a") as f:
                f.write(f"Epoch {epoch + 1}: {metrics}\n")

        self.model.add_callback("on_fit_epoch_end", epoch_logger)
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
        )

        return self.results
    
    def get_results(self):
        return {
            "metrics": self.results.get("metrics", {}),
            "artifacts": {
                "weights": self.weights_path,
                "logs": self.log_path
            }
        }

