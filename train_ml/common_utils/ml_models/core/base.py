from abc import ABC, abstractmethod

class AbstractBaseModel(ABC):
    def __init__(self, weights, dataset, config):
        self.weights = weights
        self.dataset = dataset
        self.config = config

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def get_results(self) -> dict: pass

    def evaluate(self): pass  # Optional override

    def prepare_config(self): pass  # Optional override

    def save_outputs(self): pass  # Optional override
