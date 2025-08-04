from django.urls import path
from . import views

app_name = 'citation_management'

urlpatterns = [
    path('', views.list_citations, name='list'),
    path('list/', views.list_citations, name='list_citations'),
    path('add/', views.add_citation, name='add_citation'),
    path('delete/<uuid:citation_id>/', views.delete_citation, name='delete_citation'),
    path('format/', views.format_citations, name='format_citations'),
    path('import/', views.import_citations, name='import_citations'),
    path('export/', views.export_citations, name='export_citations'),
    path('bibliography/', views.bibliography_generator, name='bibliography'),
]
