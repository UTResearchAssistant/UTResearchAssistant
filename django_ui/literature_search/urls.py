"""URLs for Literature Search Service."""

from django.urls import path
from . import views

app_name = 'literature_search'

urlpatterns = [
    path('', views.search_literature, name='search'),
    path('search/', views.perform_search, name='perform_search'),
    path('history/', views.search_history, name='history'),
    path('bookmarks/', views.bookmarks, name='bookmarks'),
    path('bookmark/', views.bookmark_paper, name='bookmark_paper'),
    path('paper/<uuid:paper_id>/', views.paper_detail, name='paper_detail'),
    path('clear-history/', views.clear_search_history, name='clear_history'),
]
