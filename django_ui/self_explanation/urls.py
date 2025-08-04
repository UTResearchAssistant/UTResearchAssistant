from django.urls import path
from .views import (
    GenerateExplanationView, 
    ExplanationDetailView,
    GenerateDatasetExplanationView,
    DatasetExplanationDetailView
)

app_name = 'self_explanation'

urlpatterns = [
    path('generate/', GenerateExplanationView.as_view(), name='generate_explanation'),
    path('<uuid:explanation_id>/', ExplanationDetailView.as_view(), name='explanation_detail'),
    path('dataset/generate/', GenerateDatasetExplanationView.as_view(), name='generate_dataset_explanation'),
    path('dataset/<uuid:explanation_id>/', DatasetExplanationDetailView.as_view(), name='dataset_explanation_detail'),
]
