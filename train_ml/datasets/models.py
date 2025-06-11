from django.db import models
from projects.models import Project

def get_version_file(instance, filename):
    return f"dataset/{instance.project.name}/{instance.name}/{instance.version}/{filename}"

class Dataset(models.Model):
    """
    Represents a dataset used for model training and evaluation.
    Stores metadata, source information, and links to the related project.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255)
    version = models.PositiveIntegerField() 
    version_file = models.FileField(upload_to=get_version_file, null=True, blank=True)
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
        unique_together = ("project", "name")

    def __str__(self) -> str:
        return f"{self.name} (v{self.version})"
