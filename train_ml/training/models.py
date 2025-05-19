from django.db import models
from ml_models.models import ModelVersion

# Create your models here.
class TrainingSession(models.Model):
    model_version = models.OneToOneField(
        ModelVersion, 
        on_delete=models.CASCADE, 
        related_name="training_session"
    )

    session_id = models.IntegerField() 
    config = models.JSONField(blank=True, null=True)
    metrics = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[("pending", "Pending"), ("running", "Running"), ("completed", "Completed"), ("failed", "Failed")], default="pending")
    progress = models.FloatField(default=0.0)
    log_path = models.CharField(max_length=1024, blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "training_session"
        verbose_name_plural = "Training Sessions"

    def __str__(self):
        return f"Training for v{self.model_version} ({self.status})"
