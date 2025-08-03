"""Enhanced API router for advanced research features."""

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
)
from fastapi.responses import FileResponse
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import logging

from ..services.enhanced_literature_service import EnhancedLiteratureSearchService
from ..services.podcast_service import PodcastGenerationService
from ..services.video_analysis_service import VideoAnalysisService
from ..core.enhanced_config import enhanced_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests/responses


class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    sources: Optional[List[str]] = Field(
        default=None, description="Data sources to search"
    )
    max_results: Optional[int] = Field(
        default=50, description="Maximum number of results"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Search filters"
    )
    languages: Optional[List[str]] = Field(
        default=None, description="Languages to search in"
    )


class PodcastRequest(BaseModel):
    paper_ids: List[str] = Field(..., description="List of paper IDs")
    style: Optional[str] = Field(default="conversational", description="Podcast style")
    duration_minutes: Optional[int] = Field(default=15, description="Podcast duration")
    episode_type: Optional[str] = Field(default="summary", description="Episode type")
    voice_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Voice configuration"
    )


class VideoAnalysisRequest(BaseModel):
    video_url: str = Field(..., description="URL of the video to analyze")
    analysis_type: Optional[str] = Field(
        default="comprehensive", description="Type of analysis"
    )


class ResearchAlertRequest(BaseModel):
    alert_type: str = Field(..., description="Type of alert")
    query: str = Field(..., description="Search query for alert")
    frequency: Optional[str] = Field(default="weekly", description="Alert frequency")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Alert filters")
    notification_method: Optional[str] = Field(
        default="email", description="Notification method"
    )


class WritingAssistanceRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    assistance_type: str = Field(..., description="Type of assistance needed")
    context: Optional[str] = Field(default="academic", description="Writing context")


class IntegrityCheckRequest(BaseModel):
    text: str = Field(..., description="Text to check")
    check_type: str = Field(..., description="Type of check")
    title: Optional[str] = Field(default=None, description="Document title")


# Enhanced Literature Search Endpoints


@router.post("/search/enhanced")
async def enhanced_literature_search(request: EnhancedSearchRequest):
    """Enhanced literature search with multilingual support."""
    try:
        async with EnhancedLiteratureSearchService(enhanced_settings) as search_service:
            results = await search_service.unified_search(
                query=request.query,
                sources=request.sources,
                max_results=request.max_results,
                filters=request.filters,
            )

            return {
                "success": True,
                "query": request.query,
                "total_results": len(results),
                "results": results,
                "sources_searched": request.sources
                or ["arxiv", "semantic_scholar", "pubmed", "google_scholar"],
                "timestamp": datetime.utcnow(),
            }
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/multilingual")
async def multilingual_search(request: EnhancedSearchRequest):
    """Search for papers in multiple languages."""
    try:
        if not request.languages:
            raise HTTPException(
                status_code=400,
                detail="Languages must be specified for multilingual search",
            )

        async with EnhancedLiteratureSearchService(enhanced_settings) as search_service:
            results = await search_service.search_multilingual(
                query=request.query,
                languages=request.languages,
                max_results=request.max_results,
            )

            return {
                "success": True,
                "query": request.query,
                "languages": request.languages,
                "total_results": len(results),
                "results": results,
                "timestamp": datetime.utcnow(),
            }
    except Exception as e:
        logger.error(f"Multilingual search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/paper/{paper_id}")
async def get_paper_details(paper_id: str, source: str):
    """Get detailed information about a specific paper."""
    try:
        async with EnhancedLiteratureSearchService(enhanced_settings) as search_service:
            paper_details = await search_service.get_paper_details(paper_id, source)

            if not paper_details:
                raise HTTPException(status_code=404, detail="Paper not found")

            return {
                "success": True,
                "paper": paper_details,
                "timestamp": datetime.utcnow(),
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching paper details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Podcast Generation Endpoints


@router.post("/podcast/generate")
async def generate_podcast(request: PodcastRequest, background_tasks: BackgroundTasks):
    """Generate a podcast episode from research papers."""
    try:
        if not enhanced_settings.enable_podcasts:
            raise HTTPException(
                status_code=403, detail="Podcast generation is disabled"
            )

        podcast_service = PodcastGenerationService(enhanced_settings)

        # For demo purposes, create mock papers
        mock_papers = [
            {
                "id": paper_id,
                "title": f"Research Paper {paper_id}",
                "abstract": "This is a mock paper abstract for demonstration purposes.",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "topics": ["Machine Learning", "AI"],
            }
            for paper_id in request.paper_ids
        ]

        if request.episode_type == "interview":
            episode_data = await podcast_service.generate_interview_style_podcast(
                papers=mock_papers, duration_minutes=request.duration_minutes
            )
        elif request.episode_type == "debate":
            episode_data = await podcast_service.generate_debate_podcast(
                papers=mock_papers, controversial_topic="AI Ethics in Research"
            )
        else:
            episode_data = await podcast_service.generate_paper_summary_podcast(
                papers=mock_papers,
                style=request.style,
                duration_minutes=request.duration_minutes,
                voice_config=request.voice_config,
            )

        return {
            "success": True,
            "episode": episode_data,
            "timestamp": datetime.utcnow(),
        }
    except Exception as e:
        logger.error(f"Podcast generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/podcast/{episode_id}")
async def get_podcast_episode(episode_id: str):
    """Get podcast episode details."""
    try:
        # This would fetch from database in production
        return {
            "success": True,
            "episode": {
                "id": episode_id,
                "title": "Research Insights Episode",
                "status": "generated",
                "audio_url": f"/api/v2/podcast/{episode_id}/audio",
            },
        }
    except Exception as e:
        logger.error(f"Error fetching podcast episode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/podcast/{episode_id}/audio")
async def get_podcast_audio(episode_id: str):
    """Download podcast audio file."""
    try:
        # This would return the actual audio file in production
        audio_path = f"./storage/podcasts/episode_{episode_id}.mp3"

        # For demo, return a placeholder response
        return {
            "message": "Audio file would be served here",
            "episode_id": episode_id,
            "download_url": f"/download/podcast/{episode_id}.mp3",
        }
    except Exception as e:
        logger.error(f"Error serving podcast audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video Analysis Endpoints


@router.post("/video/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a research video."""
    try:
        if not enhanced_settings.enable_video_analysis:
            raise HTTPException(status_code=403, detail="Video analysis is disabled")

        video_service = VideoAnalysisService(enhanced_settings)

        if request.analysis_type == "lecture":
            analysis = await video_service.analyze_lecture_video(request.video_url)
        elif request.analysis_type == "presentation":
            analysis = await video_service.analyze_conference_presentation(
                request.video_url
            )
        else:
            analysis = await video_service.analyze_research_video(
                request.video_url, request.analysis_type
            )

        return {"success": True, "analysis": analysis, "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/generate-podcast")
async def generate_video_podcast(request: VideoAnalysisRequest):
    """Generate a podcast from a video analysis."""
    try:
        video_service = VideoAnalysisService(enhanced_settings)

        # First analyze the video
        video_analysis = await video_service.analyze_research_video(request.video_url)

        # Then generate podcast
        podcast_data = await video_service.generate_video_summary_podcast(
            video_analysis
        )

        return {
            "success": True,
            "video_analysis": video_analysis,
            "podcast": podcast_data,
            "timestamp": datetime.utcnow(),
        }
    except Exception as e:
        logger.error(f"Video podcast generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Research Alerts Endpoints


@router.post("/alerts/create")
async def create_research_alert(request: ResearchAlertRequest):
    """Create a new research alert."""
    try:
        # This would save to database in production
        alert_data = {
            "id": f"alert_{datetime.now().timestamp()}",
            "alert_type": request.alert_type,
            "query": request.query,
            "frequency": request.frequency,
            "filters": request.filters,
            "notification_method": request.notification_method,
            "active": True,
            "created_at": datetime.utcnow(),
        }

        return {
            "success": True,
            "alert": alert_data,
            "message": "Research alert created successfully",
        }
    except Exception as e:
        logger.error(f"Error creating research alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_research_alerts(user_id: Optional[str] = None):
    """Get all research alerts for a user."""
    try:
        # This would fetch from database in production
        alerts = [
            {
                "id": "alert_1",
                "alert_type": "keyword",
                "query": "machine learning",
                "frequency": "weekly",
                "active": True,
                "created_at": datetime.utcnow(),
            }
        ]

        return {"success": True, "alerts": alerts, "total": len(alerts)}
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/{alert_id}")
async def delete_research_alert(alert_id: str):
    """Delete a research alert."""
    try:
        # This would delete from database in production
        return {"success": True, "message": f"Alert {alert_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Writing Assistance Endpoints


@router.post("/writing/analyze")
async def analyze_writing(request: WritingAssistanceRequest):
    """Analyze and improve academic writing."""
    try:
        # Mock writing analysis
        analysis = {
            "original_text": request.text,
            "improved_text": f"[IMPROVED] {request.text}",
            "suggestions": [
                {"type": "grammar", "message": "Consider using active voice"},
                {"type": "style", "message": "Simplify complex sentences"},
                {"type": "clarity", "message": "Define technical terms"},
            ],
            "metrics": {
                "readability_score": 7.5,
                "grade_level": "Graduate",
                "word_count": len(request.text.split()),
                "sentence_count": len(request.text.split(".")),
            },
            "assistance_type": request.assistance_type,
            "analyzed_at": datetime.utcnow(),
        }

        return {"success": True, "analysis": analysis}
    except Exception as e:
        logger.error(f"Writing analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/writing/generate-citations")
async def generate_citations(papers: List[Dict], style: str = "APA"):
    """Generate citations for papers in specified style."""
    try:
        citations = []

        for paper in papers:
            if style.upper() == "APA":
                citation = f"{', '.join(paper.get('authors', ['Unknown']))} ({paper.get('year', 'n.d.')}). {paper.get('title', 'Untitled')}. {paper.get('journal', 'Journal')}"
            elif style.upper() == "MLA":
                citation = f"{paper.get('authors', ['Unknown'])[0]}. \"{paper.get('title', 'Untitled')}.\" {paper.get('journal', 'Journal')}, {paper.get('year', 'n.d.')}"
            else:
                citation = f"{paper.get('title', 'Untitled')} - {', '.join(paper.get('authors', ['Unknown']))}"

            citations.append(
                {"paper_id": paper.get("id"), "citation": citation, "style": style}
            )

        return {"success": True, "citations": citations, "style": style}
    except Exception as e:
        logger.error(f"Citation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Academic Integrity Endpoints


@router.post("/integrity/check")
async def check_academic_integrity(request: IntegrityCheckRequest):
    """Check document for plagiarism and AI detection."""
    try:
        if not enhanced_settings.enable_plagiarism_check:
            raise HTTPException(
                status_code=403, detail="Integrity checking is disabled"
            )

        # Mock integrity check
        check_result = {
            "document_title": request.title,
            "plagiarism_score": 15.2,  # Percentage
            "ai_detection_score": 23.5,  # Percentage
            "similarity_matches": [
                {
                    "source": "Academic Paper X",
                    "similarity": 8.5,
                    "url": "https://example.com/paper1",
                }
            ],
            "ai_detection_details": {
                "confidence": 0.75,
                "model_predictions": ["GPT-like patterns detected"],
            },
            "check_type": request.check_type,
            "status": "completed",
            "checked_at": datetime.utcnow(),
        }

        return {"success": True, "integrity_check": check_result}
    except Exception as e:
        logger.error(f"Integrity check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Research Collaboration Endpoints


@router.post("/collaboration/find-researchers")
async def find_potential_collaborators(
    research_interests: List[str], max_results: int = 10
):
    """Find researchers with similar interests for collaboration."""
    try:
        # Mock collaborator search
        collaborators = [
            {
                "id": f"researcher_{i}",
                "name": f"Dr. Researcher {i}",
                "affiliation": f"University {i}",
                "h_index": 25 + i,
                "research_interests": research_interests[:2]
                + [f"Additional Interest {i}"],
                "match_score": 0.85 - (i * 0.05),
                "recent_papers": [f"Recent Paper {i}.1", f"Recent Paper {i}.2"],
            }
            for i in range(1, min(max_results + 1, 6))
        ]

        return {
            "success": True,
            "collaborators": collaborators,
            "search_interests": research_interests,
            "total_found": len(collaborators),
        }
    except Exception as e:
        logger.error(f"Collaborator search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaboration/request")
async def send_collaboration_request(
    requester_id: str,
    target_researcher_id: str,
    message: str,
    collaboration_type: str = "research",
):
    """Send a collaboration request to a researcher."""
    try:
        request_data = {
            "id": f"collab_req_{datetime.now().timestamp()}",
            "requester_id": requester_id,
            "target_researcher_id": target_researcher_id,
            "message": message,
            "collaboration_type": collaboration_type,
            "status": "pending",
            "created_at": datetime.utcnow(),
        }

        return {
            "success": True,
            "collaboration_request": request_data,
            "message": "Collaboration request sent successfully",
        }
    except Exception as e:
        logger.error(f"Collaboration request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Research Project Management Endpoints


@router.post("/projects/create")
async def create_research_project(
    user_id: str, title: str, description: str, research_questions: List[str]
):
    """Create a new research project."""
    try:
        project_data = {
            "id": f"project_{datetime.now().timestamp()}",
            "user_id": user_id,
            "title": title,
            "description": description,
            "research_questions": research_questions,
            "status": "planning",
            "created_at": datetime.utcnow(),
            "related_papers": [],
            "collaborators": [],
        }

        return {
            "success": True,
            "project": project_data,
            "message": "Research project created successfully",
        }
    except Exception as e:
        logger.error(f"Project creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def get_research_projects(user_id: Optional[str] = None):
    """Get research projects for a user."""
    try:
        # Mock projects
        projects = [
            {
                "id": "project_1",
                "title": "AI in Healthcare Research",
                "description": "Investigating applications of AI in medical diagnosis",
                "status": "active",
                "created_at": datetime.utcnow(),
                "progress": 65,
            }
        ]

        return {"success": True, "projects": projects, "total": len(projects)}
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and Analytics Endpoints


@router.get("/analytics/usage")
async def get_usage_analytics():
    """Get system usage analytics."""
    try:
        analytics = {
            "total_searches": 1250,
            "podcasts_generated": 45,
            "videos_analyzed": 23,
            "alerts_active": 78,
            "integrity_checks": 156,
            "popular_topics": [
                {"topic": "Machine Learning", "count": 234},
                {"topic": "AI Ethics", "count": 187},
                {"topic": "Deep Learning", "count": 165},
            ],
            "recent_activity": [
                {"action": "search", "timestamp": datetime.utcnow(), "user": "user123"},
                {
                    "action": "podcast_generated",
                    "timestamp": datetime.utcnow(),
                    "user": "user456",
                },
            ],
        }

        return {
            "success": True,
            "analytics": analytics,
            "generated_at": datetime.utcnow(),
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check with service status."""
    try:
        service_status = {
            "literature_search": "healthy",
            "podcast_generation": (
                "healthy" if enhanced_settings.enable_podcasts else "disabled"
            ),
            "video_analysis": (
                "healthy" if enhanced_settings.enable_video_analysis else "disabled"
            ),
            "multilingual_support": (
                "healthy" if enhanced_settings.enable_multilingual else "disabled"
            ),
            "integrity_checking": (
                "healthy" if enhanced_settings.enable_plagiarism_check else "disabled"
            ),
        }

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "services": service_status,
            "features_enabled": {
                "podcasts": enhanced_settings.enable_podcasts,
                "video_analysis": enhanced_settings.enable_video_analysis,
                "multilingual": enhanced_settings.enable_multilingual,
                "citation_networks": enhanced_settings.enable_citation_networks,
                "ai_detection": enhanced_settings.enable_ai_detection,
            },
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
