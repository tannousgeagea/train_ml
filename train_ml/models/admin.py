from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import (
    Model,
    ModelVersion,
    ModelTask,
    ModelFramework,
)
# Register your models here.

@admin.register(ModelTask)
class ModelTaskAdmin(ModelAdmin):
    list_display = ("name", "description", "created_at")

@admin.register(ModelFramework)
class ModelFrameworkAdmin(ModelAdmin):
    list_display = ("name", "description", "created_at")

@admin.register(Model)
class ModelAdmin(ModelAdmin):
    list_display = ("id", "name", "task", "framework")
    
@admin.register(ModelVersion)
class ModelVersionAdmin(ModelAdmin):
    list_display = ("model", "version")
    list_filter = ("model", )
    
    