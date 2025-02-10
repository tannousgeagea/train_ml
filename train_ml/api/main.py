
import cv2
import django
import random
django.setup()
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from common_utils.detection.convertor import copy_and_paste
from common_utils.detection.core import Detections
from common_utils.model.base import BaseModels
from models.models import (
    ModelVersion
)

seg_model = ModelVersion.objects.filter(
    model__name="waste-segmentation-gate"
).order_by('-version').first()


seg_model = BaseModels(
    weights=seg_model.checkpoint.path,
)

model = BaseModels(
    weights='/home/appuser/src/train_ml/runs/detect/waste_material_classification_synthetic/weights/best.pt'
    )

app = FastAPI()

class PredictionResponse(BaseModel):
    predictions: list


# colors = [
#     "#" + hex(random.randrange(0, 2**24))[2:] for _ in range(20)
# ]

colors = [
    "#540bbb",
    "#39f985",
    "#969035",
    "#dddc06",
    "#185705",
    "#c2da7b",
    "#baf1ed",
    "#4c6ccc",
    "#214da3",
    "#4186c",
    "#cfc1e4",
    "#6f68cb",
    "#f53b33",
    "#e1e2ab",
    "#66b100",
    "#4b1cad",
    "#7ee4e2",
    "#656d91",
    "#aceb71",
    "#891c7c"
]

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    # Read image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = Detections.from_dict({})
    results = seg_model.classify_one(
        image=img,
        conf=0.25,
        mode="detect"
    )
    detections = detections.from_dict(results)
    xy = detections.xy
    xyxy = detections.xyxyn


    classes = []
    class_ids = []
    confidences = []
    for i, polygon in enumerate(xy):        
        res = model.classify_one(
            copy_and_paste(
                img=img,
                polygon=polygon
            )
        )
        
        detections = Detections.from_dict(res)
        if len(detections):
            cls_id = int(detections.class_id[0])
            class_ids.append(cls_id)
            classes.append(res.get("class_names")[cls_id])
            confidences.append(float(detections.confidence[0]))
        else:
            class_ids.append(10)
            classes.append('unknown')
            confidences.append(0)
    
    predictions = [
        {
            "x": _xyxy[0],
            "y": _xyxy[1],
            "width": _xyxy[2] - _xyxy[0],
            "height": _xyxy[3] - _xyxy[1], 
            "confidence": confidences[i],
            "class_id": class_ids[i],
            "class": classes[i], 
            "id": str(i),
            "color": colors[class_ids[i]]
        } for i, _xyxy in enumerate(xyxy)
    ]
    
    return {"predictions": predictions}