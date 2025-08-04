from django.urls import path
from . import views

app_name = 'collaboration'

urlpatterns = [
    path('projects/', views.list_projects, name='list_projects'),
    path('project/create/', views.create_project, name='create_project'),
    path('project/<int:project_id>/invite/', views.invite_member, name='invite_member'),
    path('project/<int:project_id>/activity/', views.project_activity, name='project_activity'),
    path('project/<int:project_id>/leave/', views.leave_project, name='leave_project'),
]
