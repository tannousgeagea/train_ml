
import io
import cv2
import time
import random
import django
django.setup()
from PIL import Image
from asgiref.sync import sync_to_async
import numpy as np
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request
from fastapi import Response
from django.db.models import F
from fastapi.routing import APIRoute, APIRouter
from django.db import transaction
from fastapi import UploadFile, File
from ml_models.models import ModelVersion
from common_utils.ml_models.inference import get_inference_plugin

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)


@sync_to_async
def get_model(model_version_id):
    mv = ModelVersion.objects.filter(model_version_id=model_version_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"model version {model_version_id} not found")
    
    return {
        "weights": mv.checkpoint.path,
        "framework": mv.model.framework.name
    }
        

@router.api_route(
    "/infer/{model_version_id}", methods=["POST"]
    )
async def infer(
    model_version_id: int, 
    file: UploadFile = File(...),
    confidence_threshold:Optional[float] = 0.25
    ):
    try:
        mv = await get_model(model_version_id)
        predictor_cls = get_inference_plugin(framework=mv.model.framework.name)

        predictor = predictor_cls(weights=mv.checkpoint.path)
        image_bytes =  await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        detections = predictor.predict(image=image, confidence_threshold=confidence_threshold)

        predictions = [
            {
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "width":int(xyxy[2] - xyxy[0]),
                "height":int( xyxy[3] - xyxy[1]),
                "xyxyn": detections.xyxyn.tolist()[i],
                "xyxy": xyxy.tolist(),
                "confidence": float(detections.confidence.astype(float)[i]),
                "class_id": int(detections.class_id.astype(int)[i]),
                "class_label": detections.data['class_name'][i],
                "id": str(i),
            } for i, xyxy in enumerate(detections.xyxy.astype(int))
        ]


        return {
            "width": image.size[0],
            "height": image.size[1],
            "predictions": predictions
        }

    except HTTPException as e:
        raise e
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {err}"
        )