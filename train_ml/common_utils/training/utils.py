import os
import requests
import django
django.setup()
from typing import Optional
from ml_models.models import Model, ModelVersion, get_model_artifact_path
from training.models import TrainingSession
from django.conf import settings

API_URL = os.getenv("CORE_API_URL")
SAVE_DIR = settings.MEDIA_ROOT

def get_model_weights(model_version_id, model_version):
    try:

        base_model_version = ModelVersion.objects.filter(model_version_id=model_version_id).first()
        if base_model_version and base_model_version.checkpoint:
            return base_model_version.checkpoint.path
            
        response = requests.get(f"{API_URL}/api/v1/model-versions/{model_version_id}", headers={"accept": "application/json"})
        response.raise_for_status()
        download_url = response.json().get("artifacts", {}).get("weights")

        if not download_url:
            raise Exception("No download URL returned by the API.")

        filename = get_model_artifact_path(model_version, os.path.basename(download_url.split("?")[0]))
        filepath = os.path.join(SAVE_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"Downloading from: {download_url}")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File downloaded successfully: {filepath}")
        model_version.checkpoint = filename
        model_version.save(update_fields=["checkpoint"])
        
        return filepath
    
    except Exception as e:
        raise Exception(f"Error: {e}")

def update_core_session(session_id, progress, logs, status, metrics, error_message=None):
    try:
        payload = {
            "progress": progress,
            "logs": logs,
            "status": status,
            "metrics": metrics,
            "error_message": error_message,
        }

        url = f"{API_URL}/api/v1/training-sessions/{session_id}"
        res = requests.patch(url, json=payload, timeout=5)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to send progress update to core: {e}")

def update_core_model(version_id, status, metrics, error_message=None):
    try:
        payload = {
            "status": status,
            "metrics": metrics,
            "error_message": error_message,
        }

        url = f"{API_URL}/api/v1/model-versions/{version_id}"
        res = requests.patch(url, json=payload, timeout=5)
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to send progress update to core: {e}")

def upload_model_artifact(
    core_api_url: str,
    version_id: int,
    file_path: str,
    artifact_type: str = "checkpoint",
    token: Optional[str] = None
) -> dict:
    """
    Uploads a checkpoint or logs file to the core backend.

    Args:
        core_api_url (str): Base URL of the core API (e.g., "http://core-backend:8000").
        version_id (int): ModelVersion ID to upload to.
        file_path (str): Path to the file to upload.
        artifact_type (str): Either "checkpoint" or "logs".
        token (Optional[str]): Bearer token for authentication, if needed.

    Returns:
        dict: Response JSON from the core backend.

    Raises:
        Exception: If upload fails.
    """
    url = f"{core_api_url}/api/v1/model-versions/{version_id}/upload-artifact"
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    with open(file_path, "rb") as f:
        files = {
            "file": (file_path.split("/")[-1], f),
        }
        data = {
            "type": artifact_type
        }

        response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code != 200:
        raise Exception(f"Failed to upload {artifact_type}. "
                        f"Status code: {response.status_code}, Response: {response.text}")

    return response.json()


def training_session_callback_factory(session_id: int):
    def callback(event: dict):
        try:
            session = TrainingSession.objects.get(id=session_id)
            if session.metrics is None:
                session.metrics = []

            if event["type"] == "epoch_end":
                session.progress = event.get("progress") or session.progress
                session.metrics.append(event.get("metrics"))
                session.logs += event.get("logs") + "\n"
                session.save(update_fields=["progress", "metrics", "logs"])

                if session.model_version:
                    session.model_version.metrics = event["metrics"]
                    session.model_version.save(update_fields=["metrics"])
                
            elif event["type"] == "complete":
                session.progress = 100.0
                session.status = "completed"
                session.logs += event.get("logs") + "\n"
                session.save(update_fields=["progress", "status", "logs"])

                if session.model_version:
                    session.model_version.status = "trained"
                    session.model_version.metrics = event["metrics"]
                    session.model_version.save(update_fields=["metrics", "status"])

            elif event["type"] == "error":
                session.status = "failed"
                session.error_message = event.get("error_message", "")
                session.save(update_fields=["status", "error_message"])

            update_core_session(
                session_id=session.session_id,
                progress=session.progress,
                logs=event.get("logs"),
                status=session.status,
                metrics=event.get("metrics")
            )

            update_core_model(
                version_id=session.model_version.model_version_id,
                status=session.model_version.status,
                metrics=session.model_version.metrics,
            )


        except TrainingSession.DoesNotExist:
            print(f"Training session {session_id} not found.")
        except Exception as e:
            print(f"[TrainingCallback] Failed to update session: {e}")
    
    return callback

if __name__ == "__main__":
    model_path = get_model_weights(model_version_id=4)
    print(model_path)