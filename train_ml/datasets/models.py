from django.db import models
from projects.models import Project

class Dataset(models.Model):
    """
    Represents a dataset used for model training and evaluation.
    Stores metadata, source information, and links to the related project.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255, unique=True)
    version = models.PositiveIntegerField() 
    version_file = models.FileField(upload_to="versions/", null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    meta_info = models.JSONField(blank=True, null=True, help_text="Any additional metadata")
    dataset_id = models.PositiveIntegerField()
    
    class Meta:
        db_table = "ml_dataset"
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} (v{self.version})"
