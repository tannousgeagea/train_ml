from abc import ABC, abstractmethod
from PIL import Image

class BaseInferencePlugin(ABC):
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {}

    @abstractmethod
    def predict(self, image: Image.Image, confidence_threshold:float=0.25) -> dict:
        pass
