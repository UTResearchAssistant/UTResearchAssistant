from django.urls import path
from . import views

app_name = 'writing_assistance'

urlpatterns = [
    path('assist/', views.assist_writing, name='assist_writing'),
    path('status/<int:session_id>/', views.assistance_status, name='assistance_status'),
    path('analytics/', views.writing_analytics, name='writing_analytics'),
    path('list/', views.list_sessions, name='list_sessions'),
]
