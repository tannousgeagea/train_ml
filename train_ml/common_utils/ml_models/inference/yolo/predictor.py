from .base import BaseInferencePlugin
from PIL import Image
import numpy as np
from ultralytics import YOLO
from common_utils.detection.core import Detections

class YOLOInferencePlugin(BaseInferencePlugin):
    def __init__(self, weights, config=None):
        super().__init__(weights, config)
        self.model = None
        
    def init_model(self):
        if not os.path.exists(self.weights):
            raise FileExistsError(f"Model not found: {self.weights}")

        return YOLO(self.weights)

    def predict(self, image: Image.Image, confidence_threshold:float=0.25) -> Detections:
        img_array = np.array(image)
        results = self.model(img_array, conf=confidence_threshold)
        detections = Detections.from_ultralytics(
            ultralytics_results=results[0]
        )

        return detections
