
import os
import time
import django
django.setup()
from django.conf import settings
from celery import shared_task
from django.utils import timezone
from common_utils.logger.core import TrainingLogger
from ml_models.models import ModelVersion, get_model_artifact_path
from common_utils.ml_models.core import get_trainer
from common_utils.datasets.utils import prepare_dataset
from common_utils.training.utils import get_model_weights, training_session_callback_factory, upload_model_artifact, update_core_session


def logging_utils(logger, msg, session):
    logger.info(msg)
    session.logs += msg + "\n"
    session.save(update_fields=["logs"])

    update_core_session(
        session_id=session.session_id,
        logs=msg,
        progress=None,
        status=None,
        metrics=None,
        error_message=None
    )

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1}, ignore_result=True,
             name='train_model:execute')
def train_model(self, version_id:str, base_model_version:int=None):
    try:
        version = ModelVersion.objects.get(id=version_id)
        session = version.training_session

        # Start training
        session.status = "running"
        session.started_at = timezone.now()
        session.save()

        log_file = get_model_artifact_path(version, f"{version.model.name}_V{version.version}.log")
        log_path = os.path.join(settings.MEDIA_ROOT, log_file)
        logger = TrainingLogger(log_file=log_path)
        version.logs = log_file
        version.save(update_fields=["logs"])


        logger.info("Building model architecture...")
        if base_model_version:
            weights = get_model_weights(base_model_version, version)
        else:
            weights = "yolo11n.pt"

        callback = training_session_callback_factory(session_id=session.id)
        trainer_cls = get_trainer(framework_name=version.model.framework.name)


        msg = f"Loading Dataset {version.dataset_version} ..."
        logging_utils(logger, msg=msg, session=session)
        data = prepare_dataset(
            version_id=version.dataset_version.dataset_id,
            save_dir="",
            annotation_format="yolo",
        )
        msg = f"Loading dataset Done !"
        logging_utils(logger, msg=msg, session=session)

        logging_utils(logger, f"Loading trainer for model framework {version.model.framework.name} ...", session)
        trainer = trainer_cls(
            dataset=data,
            weights=weights,
            config=version.config,
            logger=logger,
            on_event=callback
        )
        logging_utils(logger, "Loading trainer Done !", session)

        logging_utils(logger, "Starting training loop...", session)
        results = trainer.train()
        model_filename = get_model_artifact_path(version, f"{version.model.name}_V{version.version}.pt")
        model_path = trainer.save(
            os.path.join(
                settings.MEDIA_ROOT, model_filename
            )
        )

        version.checkpoint = model_filename
        version.status = "trained"
        version.save()

        upload_model_artifact(
            core_api_url=os.getenv("CORE_API_URL"),
            version_id=version.model_version_id,
            file_path=model_path,
            artifact_type="checkpoint",
        )

        session.status = "completed"
        session.completed_at = timezone.now()
        session.save(update_fields=["status", "completed_at"])

    except Exception as e:
        if "version" in locals():
            version.status = "failed"
            version.save()
        if "session" in locals():
            session.status = "failed"
            session.error_message = str(e)
            session.completed_at = timezone.now()
            session.save()
        raise