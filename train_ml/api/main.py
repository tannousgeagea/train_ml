from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import random

model = YOLO('/home/appuser/src/train_ml/runs/detect/garbage_classification.v1.2/weights/best.pt')
app = FastAPI()

class PredictionResponse(BaseModel):
    predictions: list


colors = [
    "#" + hex(random.randrange(0, 2**24))[2:] for _ in range(20)
]
@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    # Read image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)
    results = results[0]
    boxes = results.boxes
    xyxy = boxes.xyxyn.cpu().numpy().tolist()
    
    predictions = [
        {
            "x": _xyxy[0],
            "y": _xyxy[1],
            "width": _xyxy[2] - _xyxy[0],
            "height": _xyxy[3] - _xyxy[1], 
            "confidence": boxes.conf.cpu().numpy().tolist()[i],
            "class_id": boxes.cls.cpu().numpy().astype(int).tolist()[i],
            "class": results.names[boxes.cls.cpu().numpy().astype(int).tolist()[i]],
            "id": str(i),
            "color": colors[boxes.cls.cpu().numpy().astype(int).tolist()[i]],
        } for i, _xyxy in enumerate(xyxy)
    ]
    return {"predictions": predictions}
