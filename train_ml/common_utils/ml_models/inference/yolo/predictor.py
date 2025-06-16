
import cv2
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
from common_utils.detection.core import Detections
from common_utils.ml_models.inference.base import BaseInferencePlugin

class YOLOInferencePlugin(BaseInferencePlugin):
    def __init__(self, weights, config=None):
        super().__init__(weights, config)
        self.model = self.init_model()
        
    def init_model(self):
        if not os.path.exists(self.weights):
            raise FileExistsError(f"Model not found: {self.weights}")

        return YOLO(self.weights)

    def predict(self, image: Image.Image, confidence_threshold:float=0.25) -> Detections:
        image = image.convert('RGB')
        img_array = np.array(image)
        cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = self.model(cv_image, conf=confidence_threshold)
        detections = Detections.from_ultralytics(
            ultralytics_results=results[0]
        )

        return detections
