from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import TrainingSession


@admin.register(TrainingSession)
class TrainingSessionAdmin(ModelAdmin):
    list_display = (
        "id",
        "model_version",
        "status",
        "progress",
        "started_at",
        "completed_at",
    )
    list_filter = ("status", "started_at", "completed_at")
    search_fields = ("model_version__model__name",)