from django.urls import path
from . import views

app_name = 'podcast_generation'

urlpatterns = [
    path('', views.list_podcasts, name='list'),
    path('generate/', views.generate_podcast, name='generate'),
    path('generator/', views.podcast_generator, name='generator'),
    path('status/<uuid:podcast_id>/', views.podcast_status, name='status'),
    path('detail/<uuid:podcast_id>/', views.podcast_detail, name='detail'),
    path('download/<uuid:podcast_id>/', views.download_podcast, name='download'),
    path('delete/<uuid:podcast_id>/', views.delete_podcast, name='delete'),
    path('like/<uuid:podcast_id>/', views.like_podcast, name='like'),
]
