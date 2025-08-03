"""Enhanced URL routes for the core app."""

from django.urls import path, include
from . import views

# Import enhanced API views
from api.enhanced_research import (
    SyncEnhancedLiteratureSearchView,
    SyncPaperAnalysisView,
    SyncPaperComparisonView,
    SyncSmartRecommendationView,
    SyncResearchTrendAnalysisView,
    SyncResearchGapAnalysisView,
    health_check
)

urlpatterns = [
    # Main pages
    path("", views.home, name="home"),
    # Authentication
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    
    # Enhanced features
    path("search/", views.enhanced_search, name="enhanced_search"),
    path("podcasts/", views.podcast_generator, name="podcast_generator"),
    path(
        "podcasts/download/<uuid:podcast_id>/",
        views.download_podcast,
        name="download_podcast",
    ),
    path("videos/", views.video_analyzer, name="video_analyzer"),
    path("alerts/", views.research_alerts, name="research_alerts"),
    path("coordinator/", views.research_coordinator_view, name="research_coordinator"),
    path("writing/", views.writing_assistant, name="writing_assistant"),
    path("collaboration/", views.collaboration, name="collaboration"),
    path("preferences/", views.user_preferences, name="user_preferences"),
    
    # Advanced Research Features (NEW)
    path("analysis/", views.paper_analysis_view, name="paper_analysis"),
    path("trends/", views.trend_analysis_view, name="trend_analysis"),
    path("gaps/", views.research_gaps_view, name="research_gaps"),
    path("recommendations/", views.recommendations_view, name="recommendations"),
    path("comparison/", views.paper_comparison_view, name="paper_comparison"),
    
    # Paper management
    path("papers/<uuid:paper_id>/", views.paper_detail, name="paper_detail"),
    
    # Enhanced API endpoints (NEW)
    path("api/v2/search/literature/", SyncEnhancedLiteratureSearchView.as_view(), name="api_enhanced_search"),
    path("api/v2/papers/analyze/", SyncPaperAnalysisView.as_view(), name="api_paper_analysis"),
    path("api/v2/papers/compare/", SyncPaperComparisonView.as_view(), name="api_paper_comparison"),
    path("api/v2/recommendations/", SyncSmartRecommendationView.as_view(), name="api_recommendations"),
    path("api/v2/trends/<str:field>/", SyncResearchTrendAnalysisView.as_view(), name="api_trend_analysis"),
    path("api/v2/gaps/<str:field>/", SyncResearchGapAnalysisView.as_view(), name="api_research_gaps"),
    path("api/v2/health/", health_check, name="api_health"),
    
    # Legacy API endpoints (maintained for compatibility)
    path("api/search/", views.api_search, name="api_search"),
    
    # Legacy views (for backward compatibility)
    path("dataset/", views.dataset_view, name="dataset"),
    path("training/", views.training_view, name="training"),
    path("research/", views.research_view, name="research"),
    path("prompt/", views.prompt_view, name="prompt"),
]
