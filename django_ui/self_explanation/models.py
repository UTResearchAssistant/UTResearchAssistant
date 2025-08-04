from django.db import models
import uuid

from training_config.models import TrainingConfiguration, DatasetProfile, BaseModelProfile

class SelfExplanation(models.Model):
    """Model for storing self-explanations of training configurations."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_configuration = models.OneToOneField(
        TrainingConfiguration, 
        on_delete=models.CASCADE, 
        related_name='self_explanation'
    )
    
    # The generated explanation
    explanation_text = models.TextField(
        help_text="The full, detailed self-explanation text."
    )
    
    # Structured components of the explanation
    key_findings = models.JSONField(
        default=list, 
        help_text="Bulleted list of key insights or predictions."
    )
    
    # Confidence and risk assessment
    confidence_score = models.FloatField(
        default=0.0,
        help_text="Confidence in the explanation's predictions (0.0 to 1.0)."
    )
    risk_assessment = models.TextField(
        blank=True,
        help_text="Potential risks or limitations of the configuration."
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Self-explanation for {self.training_configuration.name}"

class ExplanationFeedback(models.Model):
    """Model for storing user feedback on self-explanations."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    self_explanation = models.ForeignKey(
        SelfExplanation, 
        on_delete=models.CASCADE, 
        related_name='feedback'
    )
    user = models.ForeignKey(
        'auth.User', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    # Feedback content
    is_helpful = models.BooleanField(default=True)
    feedback_text = models.TextField(blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Feedback for {self.self_explanation.id} by {self.user.username if self.user else 'Anonymous'}"


class DatasetExplanation(models.Model):
    """Model for storing self-explanations of datasets."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset_profile = models.OneToOneField(
        DatasetProfile,
        on_delete=models.CASCADE,
        related_name='explanation'
    )
    explanation_text = models.TextField(
        help_text="The full, detailed self-explanation text for the dataset."
    )
    key_findings = models.JSONField(
        default=list,
        help_text="Bulleted list of key insights about the dataset."
    )
    confidence_score = models.FloatField(
        default=0.0,
        help_text="Confidence in the explanation's analysis (0.0 to 1.0)."
    )
    potential_biases = models.TextField(
        blank=True,
        help_text="Potential biases or limitations of the dataset."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Dataset Explanations"

    def __str__(self):
        return f"Explanation for {self.dataset_profile.name}"

