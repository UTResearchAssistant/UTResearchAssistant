from django.contrib import admin
from .models import SelfExplanation, ExplanationFeedback

@admin.register(SelfExplanation)
class SelfExplanationAdmin(admin.ModelAdmin):
    list_display = ('training_configuration', 'created_at', 'updated_at')
    list_filter = ('created_at',)
    search_fields = ('training_configuration__name', 'explanation_text')
    readonly_fields = ('id', 'created_at', 'updated_at')
    raw_id_fields = ('training_configuration',)

@admin.register(ExplanationFeedback)
class ExplanationFeedbackAdmin(admin.ModelAdmin):
    list_display = ('self_explanation', 'user', 'is_helpful', 'created_at')
    list_filter = ('is_helpful',)
    search_fields = ('self_explanation__training_configuration__name', 'feedback_text')
    readonly_fields = ('id', 'created_at')
    raw_id_fields = ('self_explanation', 'user')
