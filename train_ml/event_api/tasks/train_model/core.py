
import time
import django
django.setup()
from celery import shared_task
from django.utils import timezone
from common_utils.logger.core import TrainingLogger
from training.models import ModelVersion
from common_utils.ml_models.core import get_trainer
from common_utils.datasets.utils import prepare_dataset
from common_utils.training.utils import get_model_weights, training_session_callback_factory

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5}, ignore_result=True,
             name='train_model:execute')
def train_model(self, version_id:str, base_model_version:int=None):
    try:
        version = ModelVersion.objects.get(id=version_id)
        session = version.training_session

        # Start training
        session.status = "running"
        session.started_at = timezone.now()
        session.save()

        log_path = f"/media/logs/yolo_{version.id}.log"
        logger = TrainingLogger(log_file=log_path)

        if base_model_version:
            weights = get_model_weights(base_model_version)
        else:
            weights = "yolo11n.pt"

        callback = training_session_callback_factory(session_id=session.id)
        trainer_cls = get_trainer(framework_name=version.model.framework.name)

        data = prepare_dataset(
            version_id=version.dataset_version.dataset_id,
            save_dir="",
            annotation_format="yolo",
        )

        trainer = trainer_cls(
            dataset=data,
            weights=weights,
            config=version.config,
            logger=logger,
            on_event=callback
        )

        results = trainer.train()
        version.status = "trained"
        version.save()

        session.status = "completed"
        session.completed_at = timezone.now()
        session.save()

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