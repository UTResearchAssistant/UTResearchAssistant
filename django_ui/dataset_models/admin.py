from django.contrib import admin
from .models import ResearchField, Dataset, MLModel, SearchHistory


@admin.register(ResearchField)
class ResearchFieldAdmin(admin.ModelAdmin):
    list_display = ['name', 'parent_field', 'created_at']
    list_filter = ['parent_field', 'created_at']
    search_fields = ['name', 'description']
    ordering = ['name']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset_type', 'license_type', 'citation_count', 'download_count', 'is_active']
    list_filter = ['dataset_type', 'license_type', 'is_active', 'publication_year']
    search_fields = ['name', 'description', 'tags', 'keywords']
    filter_horizontal = ['research_fields']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-citation_count', '-download_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'dataset_type', 'research_fields')
        }),
        ('URLs', {
            'fields': ('source_url', 'download_url', 'paper_url', 'github_url')
        }),
        ('Metadata', {
            'fields': ('size_mb', 'num_samples', 'num_classes', 'license_type', 'metadata')
        }),
        ('Tags & Keywords', {
            'fields': ('tags', 'keywords')
        }),
        ('Metrics', {
            'fields': ('download_count', 'citation_count', 'star_count')
        }),
        ('Publishing', {
            'fields': ('authors', 'publication_year')
        }),
        ('Status', {
            'fields': ('is_active', 'id', 'created_at', 'updated_at')
        }),
    )


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'framework', 'accuracy', 'citation_count', 'is_active']
    list_filter = ['model_type', 'framework', 'pretrained', 'is_active', 'publication_year']
    search_fields = ['name', 'description', 'architecture', 'tags', 'keywords']
    filter_horizontal = ['research_fields', 'training_datasets']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-citation_count', '-download_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'model_type', 'framework', 'research_fields')
        }),
        ('Architecture', {
            'fields': ('architecture', 'num_parameters', 'model_size_mb', 'pretrained')
        }),
        ('URLs', {
            'fields': ('model_url', 'paper_url', 'github_url', 'huggingface_url')
        }),
        ('Performance', {
            'fields': ('accuracy', 'f1_score', 'precision', 'recall', 'inference_time_ms')
        }),
        ('Training', {
            'fields': ('training_datasets', 'requirements', 'metadata')
        }),
        ('Tags & Keywords', {
            'fields': ('tags', 'keywords')
        }),
        ('Metrics', {
            'fields': ('download_count', 'citation_count', 'star_count')
        }),
        ('Publishing', {
            'fields': ('authors', 'publication_year')
        }),
        ('Status', {
            'fields': ('is_active', 'id', 'created_at', 'updated_at')
        }),
    )


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['search_type', 'query', 'research_field', 'results_count', 'created_at']
    list_filter = ['search_type', 'research_field', 'created_at']
    search_fields = ['query']
    readonly_fields = ['id', 'created_at']
    ordering = ['-created_at']
