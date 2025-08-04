from django.urls import path
from . import views

app_name = 'alerts'

urlpatterns = [
    path('list/', views.list_alerts, name='list_alerts'),
    path('create/', views.create_alert, name='create_alert'),
    path('alert/<int:alert_id>/read/', views.mark_read, name='mark_read'),
    path('alert/<int:alert_id>/delete/', views.delete_alert, name='delete_alert'),
    path('settings/', views.alert_settings, name='alert_settings'),
    path('settings/update/', views.update_settings, name='update_settings'),
]
