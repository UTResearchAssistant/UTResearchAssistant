"""FastAPI application entry point.

This module creates an instance of the FastAPI application and includes
routers defined in the ``api/v1`` and ``api/v2`` packages.  When run with an ASGI
server such as Uvicorn or Hypercorn, it exposes HTTP endpoints for
searching, summarizing, and advanced research assistance features.

The app uses dependency injection via FastAPI's built‑in support, and
can be extended by adding new routers in ``backend/app/api/v1`` or ``backend/app/api/v2``.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import summarization, search, health


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        An instance of a FastAPI application with included routers.
    """
    app = FastAPI(
        title="AI Research Assistant",
        version="2.0.0",
        description="Comprehensive AI-powered research assistant with advanced features including podcasts, video analysis, multilingual search, and academic integrity checking.",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include v1 API routers (existing functionality)
    app.include_router(health.router)
    app.include_router(summarization.router, prefix="/api/v1", tags=["summarisation"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])

    # dataset management endpoints
    from .api.v1 import (
        dataset,
        training,
        research,
    )  # local import to avoid circular deps

    app.include_router(dataset.router, prefix="/api/v1", tags=["dataset"])
    app.include_router(training.router, prefix="/api/v1", tags=["training"])
    app.include_router(research.router, prefix="/api/v1", tags=["research"])

    # Include v2 API routers (enhanced functionality)
    try:
        from .api.v2 import enhanced_routes

        app.include_router(
            enhanced_routes.router, prefix="/api/v2", tags=["enhanced-features"]
        )
        print("✅ Enhanced features (v2) loaded successfully")
    except ImportError as e:
        print(f"⚠️  Enhanced features (v2) not available: {e}")

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "AI Research Assistant",
            "version": "2.0.0",
            "description": "Comprehensive research assistant with advanced features",
            "features": {
                "v1": {
                    "search": "/api/v1/search",
                    "summarization": "/api/v1/summarization",
                    "datasets": "/api/v1/dataset",
                    "training": "/api/v1/training",
                    "research": "/api/v1/research",
                },
                "v2": {
                    "enhanced_search": "/api/v2/search/enhanced",
                    "multilingual_search": "/api/v2/search/multilingual",
                    "podcast_generation": "/api/v2/podcast/generate",
                    "video_analysis": "/api/v2/video/analyze",
                    "research_alerts": "/api/v2/alerts",
                    "writing_assistance": "/api/v2/writing/analyze",
                    "integrity_check": "/api/v2/integrity/check",
                    "collaboration": "/api/v2/collaboration",
                    "analytics": "/api/v2/analytics",
                },
            },
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Instantiate the app when this module is imported by an ASGI server.
app = create_app()
