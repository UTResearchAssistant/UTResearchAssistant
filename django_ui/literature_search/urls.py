"""URLs for Literature Search Service."""

from django.urls import path
from . import views

app_name = 'literature_search'

urlpatterns = [
    path('search/', views.search_literature, name='search_literature'),
    path('history/', views.search_history, name='search_history'),
]
