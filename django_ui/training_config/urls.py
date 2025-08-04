from django.urls import path
from . import views

app_name = 'training_config'

urlpatterns = [
    # Main dashboard
    path('', views.training_dashboard, name='dashboard'),
    
    # Configuration management
    path('builder/', views.configuration_builder, name='configuration_builder'),
    path('configurations/', views.configuration_list, name='configuration_list'),
    path('configurations/<uuid:config_id>/', views.configuration_detail, name='configuration_detail'),
    
    # Estimation and comparison
    path('estimate/', views.estimate_configuration, name='estimate_configuration'),
    path('compare/', views.configuration_comparison, name='configuration_comparison'),
    path('compare/<uuid:comparison_id>/', views.comparison_detail, name='comparison_detail'),
    
    # Local training execution
    path('start/<uuid:config_id>/', views.start_training, name='start_training'),
    path('experiments/<uuid:experiment_id>/', views.experiment_detail, name='experiment_detail'),
    
    # Cloud training
    path('cloud/', views.cloud_training_dashboard, name='cloud_dashboard'),
    path('cloud/start/<uuid:config_id>/', views.start_cloud_training, name='start_cloud_training'),
    path('cloud/jobs/<uuid:job_id>/', views.cloud_job_detail, name='cloud_job_detail'),
    path('cloud/monitor/', views.live_training_monitor, name='live_monitor'),
    
    # Live monitoring and APIs
    path('api/metrics/<uuid:job_id>/', views.training_metrics_api, name='training_metrics_api'),
    path('api/stream/<uuid:job_id>/', views.live_metrics_stream, name='live_metrics_stream'),
    
    # Webhook for cloud providers
    path('webhook/<uuid:job_id>/', views.webhook_training_update, name='webhook_training_update'),
    
    # Notebook generation
    path('generate/colab/<uuid:config_id>/', views.generate_colab_notebook, name='generate_colab_notebook'),
    
    # Profile management
    path('hardware/', views.hardware_profiles, name='hardware_profiles'),
    path('models/', views.model_profiles, name='model_profiles'),
    path('datasets/', views.dataset_profiles, name='dataset_profiles'),
]
