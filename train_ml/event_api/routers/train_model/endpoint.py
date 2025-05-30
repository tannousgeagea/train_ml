# routes/models.py
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response, Header
from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute
from typing import List, Literal
from pydantic import BaseModel
from typing_extensions import Annotated

import time
import django
django.setup()

from django.shortcuts import get_object_or_404
from django.shortcuts import get_object_or_404
from ml_models.models import ModelVersion, Model, ModelTask, ModelFramework
from datasets.models import Dataset
from projects.models import Project, Visibility, ProjectType
from training.models import TrainingSession
from event_api.tasks.train_model.core import train_model

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
    prefix="/api/v1",
    tags=["Training"],
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

class TrainingRequest(BaseModel):
    project_id: str
    base_version: Optional[int] = None
    model_name: str
    dataset_name: str
    dataset_version: int
    framework: str
    task: str
    model_id: int
    model_version_id: int
    dataset_id: int
    session_id: int
    config: dict = {}

@router.post("/train")
def trigger_training(
    req: TrainingRequest,
    x_request_id: Annotated[Optional[str], Header()] = None,
    ):
    try:

        task = get_object_or_404(ModelTask, name=req.task)
        framework = get_object_or_404(ModelFramework, name=req.framework)
        project_type = get_object_or_404(ProjectType, name=req.task)
        visibility = get_object_or_404(Visibility, name="private")
        project = Project.objects.filter(name=req.project_id).first()
        if not project:
            project, _ = Project.objects.get_or_create(
                name=req.project_id,
                visibility=visibility,
                project_type=project_type,
            )

        model = Model.objects.filter(name=req.model_name, project=project).first()
        if not model:
            model, _ = Model.objects.get_or_create(
                project=project,
                task=task,
                framework=framework,
                name=req.model_name
            )

        last_version = ModelVersion.objects.filter(model=model).order_by("-version").first()
        version_number = int(last_version.version) + 1 if last_version else 1

        dataset = Dataset.objects.filter(name=req.dataset_name, version=req.dataset_version, project=project).first()
        if not dataset:
            dataset, _ = Dataset.objects.get_or_create(
                name=req.dataset_name,
                project=project,
                version=req.dataset_version,
                dataset_id=req.dataset_id,
            )

        if ModelVersion.objects.filter(model=model, version=version_number).exists():
            raise HTTPException(409, detail=f"Model Version {version_number} already exists for this model.")
        
        model_version = ModelVersion.objects.create(
            model=model,
            model_version_id=req.model_version_id,
            version=version_number,
            dataset_version=dataset,
            config=req.config,
            status="training",
        )

        if TrainingSession.objects.filter(model_version=model_version).exists():
            raise HTTPException(409, detail="Training session already exists for this model version.")


        TrainingSession.objects.create(
            model_version=model_version, 
            session_id=req.session_id, 
            config=req.config,
            logs=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Training Session Initialized ... ... \n",
            )
        
        task = train_model.apply_async(args=(model_version.id, req.base_version,), task_id=x_request_id)

        return {
            "message": "Training triggered",
            "model_version_id": model_version.id,
            "training_session_id": model_version.training_session.id,
            "task_id": task.id,
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(500, str(e))
