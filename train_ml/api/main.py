from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import django
django.setup()
import numpy as np
from ultralytics import YOLO
import random
from models.models import (
    ModelVersion
)

from typing import Tuple

seg_model = ModelVersion.objects.filter(
    model__name="waste-segmentation-gate"
).order_by('-version').first()


seg_model = YOLO(seg_model.checkpoint.path)

model = YOLO('/home/appuser/src/train_ml/runs/detect/waste_material_classification_synthetic/weights/best.pt')
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



def polygon_to_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], color=255)
    return mask

def rescale_polygon(polygon: np.ndarray, wh0: Tuple[int, int], wh: Tuple[int, int]) -> np.ndarray:
    xyxyn = polygon / np.array([wh0[0], wh0[1]])
    xyxy = xyxyn * np.array([wh[0], wh[1]])
    return xyxy.astype(np.int32).squeeze()

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    # Read image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = seg_model.predict(img, conf=0.25)
    results = results[0]
    boxes = results.boxes
    xyxy = boxes.xyxyn.cpu().numpy().tolist()
    xy = results.masks.xy


    classes = []
    class_ids = []
    
    for i, polygon in enumerate(xy):
        kernel = np.ones((5, 5), np.uint8)
        polygon = polygon.astype(np.int32)
        epsilon = 0.01 * cv2.arcLength(polygon, True)
        polygon = cv2.approxPolyDP(polygon, epsilon, True)
        w0, h0 = 640, 640
        
        polygon = rescale_polygon(polygon, wh0=(img.shape[1], img.shape[0]), wh=(w0, h0))
        mask = polygon_to_mask(polygon, resolution_wh=(w0, h0))
        mask = cv2.dilate(mask, kernel, iterations=5) 
        
        resized = cv2.resize(img.copy(), (w0, h0))
        extracted = cv2.bitwise_and(resized, resized, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        object_cropped = extracted[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]
        
        background = np.ones((h0, w0, 3), dtype=np.uint8) * 255
        
        center_x = (w0 - w) // 2
        center_y = (h0 - h) // 2
        
        background[center_y:center_y+h, center_x:center_x+w] = cv2.bitwise_and(
            background[center_y:center_y+h, center_x:center_x+w],
            background[center_y:center_y+h, center_x:center_x+w],
            mask=cv2.bitwise_not(mask_cropped)
        )
        
        background[center_y:center_y+h, center_x:center_x+w] += object_cropped
        res = model.predict(background)
        mboxes = res[0].boxes
        if not mboxes is None:
            if len(mboxes):
                cls_id = mboxes.cls.cpu().numpy().astype(int).tolist()[0]
                class_ids.append(cls_id)
                classes.append(res[0].names[cls_id])
            else:
                class_ids.append(10)
                classes.append('unknown')
        else:
            class_ids.append(10)
            classes.append('unknown')
    
    predictions = [
        {
            "x": _xyxy[0],
            "y": _xyxy[1],
            "width": _xyxy[2] - _xyxy[0],
            "height": _xyxy[3] - _xyxy[1], 
            "confidence": boxes.conf.cpu().numpy().tolist()[i],
            "class_id": class_ids[i], #boxes.cls.cpu().numpy().astype(int).tolist()[i],
            "class": classes[i], #results.names[boxes.cls.cpu().numpy().astype(int).tolist()[i]],
            "id": str(i),
            "color": colors[class_ids[i]] #colors[boxes.cls.cpu().numpy().astype(int).tolist()[i]],
        } for i, _xyxy in enumerate(xyxy)
    ]
    return {"predictions": predictions}
