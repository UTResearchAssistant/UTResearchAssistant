"""URL configuration for the Django UI project."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core.urls")),
    path("api/literature/", include("literature_search.urls")),
    path("api/podcast/", include("podcast_generation.urls")),
    path("api/video/", include("video_analysis.urls")),
    path("api/writing/", include("writing_assistance.urls")),
    path("api/integrity/", include("academic_integrity.urls")),
    path("api/citation/", include("citation_management.urls")),
    path("api/collaboration/", include("collaboration.urls")),
    path("api/alerts/", include("alerts.urls")),
]
