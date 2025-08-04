from django.urls import path
from . import views

app_name = 'dataset_models'

urlpatterns = [
    # Search views
    path('datasets/', views.dataset_search, name='dataset_search'),
    path('models/', views.model_search, name='model_search'),
    path('search/', views.combined_search, name='combined_search'),
    
    # Detail views
    path('datasets/<uuid:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('models/<uuid:model_id>/', views.model_detail, name='model_detail'),
    
    # Dataset creation
    path('create-dataset/', views.dataset_creation_wizard, name='dataset_creation_wizard'),
    
    # API endpoints
    path('api/suggestions/', views.api_search_suggestions, name='api_suggestions'),
    path('api/generate-analysis/', views.api_generate_dataset_analysis, name='api_generate_analysis'),
    path('api/dataset-suggestions/', views.api_generate_dataset_suggestions, name='api_dataset_suggestions'),
]
