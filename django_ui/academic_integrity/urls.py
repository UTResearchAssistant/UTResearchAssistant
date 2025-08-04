from django.urls import path
from . import views

app_name = 'academic_integrity'

urlpatterns = [
    path('check/', views.check_integrity, name='check_integrity'),
    path('status/<int:check_id>/', views.check_status, name='check_status'),
    path('report/<int:check_id>/', views.integrity_report, name='integrity_report'),
    path('list/', views.list_checks, name='list_checks'),
]
