from django.urls import path
from . import views

app_name = 'video_analysis'

urlpatterns = [
    path('', views.list_analyses, name='list'),
    path('analyzer/', views.video_analyzer, name='analyzer'),
    path('analyze/', views.analyze_video, name='analyze'),
    path('status/<uuid:analysis_id>/', views.analysis_status, name='status'),
    path('results/<uuid:analysis_id>/', views.analysis_results, name='results'),
    path('delete/<uuid:analysis_id>/', views.delete_analysis, name='delete'),
]
