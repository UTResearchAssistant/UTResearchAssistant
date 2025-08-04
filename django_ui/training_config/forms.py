from django import forms
from .models import TrainingConfiguration, HyperparameterSearch

class TrainingConfigurationForm(forms.ModelForm):
    class Meta:
        model = TrainingConfiguration
        fields = [
            'name', 'dataset', 'base_model', 'adapter', 'hardware',
            'learning_rate', 'batch_size', 'epochs', 'gradient_accumulation_steps',
            'weight_decay', 'warmup_steps', 'max_grad_norm', 'optimizer',
            'precision', 'save_steps', 'eval_steps', 'early_stopping_patience', 'notes'
        ]
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

class HyperparameterSearchForm(forms.ModelForm):
    class Meta:
        model = HyperparameterSearch
        fields = [
            'name', 'base_configuration', 'search_space', 'n_trials', 
            'direction', 'objective_metric'
        ]
        widgets = {
            'search_space': forms.Textarea(attrs={'rows': 10, 'placeholder': 'Enter JSON for search space'}),
        }
