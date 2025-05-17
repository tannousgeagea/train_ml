from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import Dataset
# Register your models here.


@admin.register(Dataset)
class DatasetAdmin(ModelAdmin):
    list_display = ("id", "project", "name", "version",)
    list_filters = ("project", )