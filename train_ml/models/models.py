from django.db import models
from django.core.validators import FileExtensionValidator

def get_model_path(instance, filename):
    return f"{instance.model.name}/{instance.version}/{filename}"

class ModelTask(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "model_task"
        verbose_name_plural = "Model Tasks"
        
    def __str__(self):
        return f"Task: {self.name}"
    
class ModelFramework(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "model_framework"
        verbose_name_plural = "Model Frameworks"
        
    def __str__(self):
        return f"Framework: {self.name}"

class Model(models.Model):
    name = models.CharField(max_length=255, unique=True)
    task = models.ForeignKey(ModelTask, on_delete=models.RESTRICT)
    framework = models.ForeignKey(ModelFramework, on_delete=models.RESTRICT)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    project_id = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "ml_model"
        verbose_name_plural = "ML Models"

    def __str__(self):
        return f"{self.name} (Task: {self.task}, Framework: {self.framework})"


class ModelVersion(models.Model):
    model = models.ForeignKey('Model', on_delete=models.RESTRICT, related_name='versions')
    version = models.CharField(max_length=50)
    checkpoint = models.FileField(
        upload_to=get_model_path,
        validators=[FileExtensionValidator(allowed_extensions=['pt', 'pth', 'h5', 'onnx'])],
        max_length=1024, 
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "model_verison"
        verbose_name_plural = "Model Versions"
        unique_together = ('model', 'version')

    def __str__(self):
        return f"{self.model.name} (Version: {self.version})"
