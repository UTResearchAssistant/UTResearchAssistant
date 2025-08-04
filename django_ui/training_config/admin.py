from django.contrib import admin
from .models import (
    HardwareProfile, BaseModelProfile, DatasetProfile, AdapterConfiguration,
    TrainingConfiguration, TrainingExperiment, ConfigurationComparison
)


@admin.register(HardwareProfile)
class HardwareProfileAdmin(admin.ModelAdmin):
    list_display = ['name', 'device_type', 'memory_gb', 'fp16_tflops', 'cost_per_hour', 'is_active']
    list_filter = ['device_type', 'is_active']
    search_fields = ['name', 'device_type']
    ordering = ['name']


@admin.register(BaseModelProfile)
class BaseModelProfileAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'parameter_count', 'supports_lora', 'is_active']
    list_filter = ['model_type', 'supports_lora', 'is_active']
    search_fields = ['name', 'hf_model_name']
    ordering = ['name']


@admin.register(DatasetProfile)
class DatasetProfileAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset_type', 'num_examples', 'total_tokens', 'size_gb', 'is_active']
    list_filter = ['dataset_type', 'is_active']
    search_fields = ['name', 'hf_dataset_name']
    ordering = ['name']


@admin.register(AdapterConfiguration)
class AdapterConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'adapter_type', 'lora_rank', 'lora_alpha', 'is_active']
    list_filter = ['adapter_type', 'is_active']
    search_fields = ['name']
    ordering = ['name']


@admin.register(TrainingConfiguration)
class TrainingConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'dataset', 'base_model', 'adapter', 'status', 'estimated_cost_usd', 'created_at']
    list_filter = ['status', 'optimizer', 'precision', 'created_at']
    search_fields = ['name', 'user__username']
    ordering = ['-created_at']
    readonly_fields = ['estimated_time_hours', 'estimated_cost_usd', 'estimated_gpu_hours', 'estimated_total_flops']


@admin.register(TrainingExperiment)
class TrainingExperimentAdmin(admin.ModelAdmin):
    list_display = ['configuration', 'started_at', 'completed_at', 'best_val_loss', 'total_steps']
    list_filter = ['started_at', 'completed_at']
    search_fields = ['configuration__name']
    ordering = ['-created_at']


@admin.register(ConfigurationComparison)
class ConfigurationComparisonAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at']
    search_fields = ['name', 'user__username']
    ordering = ['-created_at']
