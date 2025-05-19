from abc import ABC, abstractmethod
from common_utils.logger.core import TrainingLogger

class AbstractBaseModel(ABC):
    def __init__(self, weights, dataset, config, logger: TrainingLogger, on_event=None):
        self.weights = weights
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.on_event = on_event

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def get_results(self) -> dict: pass

    def evaluate(self): pass  # Optional override

    def prepare_config(self): pass  # Optional override

    def save_outputs(self): pass  # Optional override
