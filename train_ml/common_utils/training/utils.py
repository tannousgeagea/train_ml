import os
import requests
from ml_models.models import Model, ModelVersion, get_model_artifact_path
from training.models import TrainingSession
from django.conf import settings

API_URL = "http://cvisionops.want:29085"
SAVE_DIR = settings.MEDIA_ROOT

def get_model_weights(model_version_id):
    try:

        model_version = ModelVersion.objects.filter(model_version_id=model_version_id).first()
        if model_version and model_version.checkpoint:
            return model_version.checkpoint.path
            
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


def training_session_callback_factory(session_id: int):
    def callback(event: dict):
        try:
            session = TrainingSession.objects.get(id=session_id)

            if event["type"] == "epoch_end":
                session.progress = event.get("progress") or session.progress
                session.metrics = event.get("metrics") or session.metrics
                session.logs += event.get("logs") + "\n"
                session.save(update_fields=["progress", "metrics", "logs"])

                if session.model_version:
                    session.model_version.metrics = event["metrics"]
                    session.model_version.save(update_fields=["metrics"])
                
            elif event["type"] == "complete":
                session.progress = 100.0
                session.status = "completed"
                session.metrics = event.get("metrics") or session.metrics
                session.logs += event.get("logs") + "\n"
                session.save(update_fields=["progress", "status", "metrics"])

            elif event["type"] == "error":
                session.status = "failed"
                session.error_message = event.get("error_message", "")
                session.save(update_fields=["status", "error_message"])

        except TrainingSession.DoesNotExist:
            print(f"Training session {session_id} not found.")
        except Exception as e:
            print(f"[TrainingCallback] Failed to update session: {e}")
    
    return callback