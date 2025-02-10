import os
import cv2
import time
import logging
import numpy as np
# from .mlflow.core import pull
from ultralytics import YOLO
from pathlib import Path

import sys

base_dir = Path(__file__).parent
sys.path.append(str(base_dir / 'mlflow'))

class BaseModels:
    def __init__(
            self,
            config_params=None,
            weights=None,
            task=None,
            mlflow=False,

    ):
        self.weights = weights
        self.task = task
        self.mlflow = mlflow

        if not config_params is None:
            self.weights = config_params['weights']

        self.model = self.init_model()

    def init_model(self):
        if self.mlflow:
            return pull(self.weights)
            
        if not os.path.exists(self.weights):
            logging.warning("⚠️ Warning: Model weights %s does not exists" % self.weights)
            if not os.path.exists(f"{base_dir}/weights/base.{self.task}.pt"):  
                return None
            
            logging.info(f"Loading base model: {base_dir}/weights/base.{self.task}.pt")
            return YOLO(f"{base_dir}/weights/base.{self.task}.pt")
            
        
        logging.info(f'Model weights: {self.weights} successfully loaded! ✅')
        return YOLO(self.weights)

    def classify_one(self, image, conf=0.25, mode='detect', classes=None):
        final_results = {}
        if self.model:
            # results = self.model.track(image, persist=True, conf=conf, classes=classes) if mode=='track' else self.model.predict(image, conf=conf, classes=classes)
            results = self.track(image, conf=conf, classes=classes) if mode=="track" else self.predict(image, conf=conf)
            
            final_results = self.write_result(final_results, 'class_names', results[0].names)
            if not results[0].probs is None:
                final_results = self.write_result(final_results, 'probabilities', results[0].probs.data.cpu().numpy().tolist())
            
            if not results[0].boxes is None:
                final_results = self.write_result(final_results, 'xyxy', results[0].boxes.xyxy.cpu().numpy().astype(int).tolist())
                final_results = self.write_result(final_results, 'xyxyn', results[0].boxes.xyxyn.cpu().numpy().tolist())
                final_results = self.write_result(final_results, 'confidence_score', results[0].boxes.conf.cpu().numpy().tolist())
                final_results = self.write_result(final_results, 'class_id', results[0].boxes.cls.cpu().numpy().astype(int).tolist())
                if not results[0].boxes.id is None:
                    final_results = self.write_result(final_results, 'tracker_id', results[0].boxes.id.cpu().numpy().astype(int).tolist())
            
            if not results[0].masks is None:
                final_results = self.write_result(final_results, 'xy', results[0].masks.xy)
                final_results = self.write_result(final_results, 'xyn', results[0].masks.xyn)
            else:
                final_results = self.write_result(final_results, 'xy', [])
                final_results = self.write_result(final_results, 'xyn', [])
                
        return final_results
    
    def write_result(self, result, key, value):
        if key not in result.keys():
            result[key] = None
        
        result[key] = value

        return result
    
    def predict(self, image, conf:float=0.25):
        if self.mlflow:
            return self.model.unwrap_python_model().predict(None, image, conf=conf)
        
        return self.model.predict(image, conf=conf)
    
    def track(self, image, conf:float=0.25, classes=None):
        if self.mlflow:
            return self.model.unwrap_python_model().track(image, conf=conf)
        
        return self.model.track(image, persist=True, conf=conf, classes=classes)
    
    def train(self, 
              data:str, 
              imgsz:int=None, 
              optimizer:str=None, 
              lr0:float=None, 
              lrf:float=None, 
              epochs:int=100, 
              batch:int=8, 
              augment:bool=False, 
              name:str=""
              ):
        
        if lr0 is None:
            logging.info('setting lr0 to default: lr0 = 0.0001')
            lr0 = 0.0001
        
        if imgsz is None:
            logging.info('setting imgsz to default: imgsz = 640')
            imgsz = 640
        
        if lrf is None:
            logging.info('setting lrf to default: lrf = 0.000001') 
            lrf = 0.000001
            
        if lrf >= lr0:
            lrf = lr0 * 1e-2
        
        if epochs is None:
            logging.info('setting epochs to default: epochs = 100')
            epochs = 100

        if batch is None:
            logging.info('setting batch to default: batch = 4')
            batch = 4
            
        if optimizer is None:
            logging.info('setting optimizer to default: optimizer = SGD')
            optimizer = "SGD"
            
        return self.model.train(
            data=data,
            imgsz=imgsz,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            epochs=epochs,
            batch=batch,
            augment=augment,
            name=name,
        )