"""Enhanced Views for the Django Research Assistant.

This module contains all the views for the enhanced research assistant including
literature search, podcast generation, video analysis, and collaboration features.
"""

import json
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse, JsonResponse, FileResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.cache import never_cache

from .forms import CustomUserCreationForm, CustomLoginForm
from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from .models import (
    Paper,
    PodcastEpisode,
    VideoAnalysis,
    ResearchAlert,
    CollaborationRequest,
    WritingAssistance,
    IntegrityCheck,
    SearchHistory,
    UserPreferences,
    PaperBookmark,
    ResearchProject,
)
# TODO: Fix these service imports once they are properly set up
# from .services import (
#     EnhancedLiteratureSearchService,
#     PodcastGenerationService,
#     VideoAnalysisService,
#     WritingAssistanceService,
# )

# Temporary placeholder classes until services are properly set up
class EnhancedLiteratureSearchService:
    async def unified_search(self, query, sources, filters=None, limit=50):
        return {"papers": [], "total": 0}

class PodcastGenerationService:
    async def generate_interview_style_podcast(self, content, duration):
        return {"success": False, "error": "Service not available"}
    
    async def generate_paper_summary_podcast(self, content, style, duration, voice):
        return {"success": False, "error": "Service not available"}

class VideoAnalysisService:
    async def analyze_research_video(self, url, analysis_type):
        return {"error": "Service not available"}

class WritingAssistanceService:
    async def assist_writing(self, content, task_type, tone):
        return {"error": "Service not available"}

# Import Django-integrated services
# TODO: Fix these imports once the services are properly set up
# from django_ui.services.django_integration import django_service_integrator
# from django_ui.services.literature_search_service import literature_search_service
# from django_ui.services.paper_analysis_service import paper_analysis_service
# from django_ui.services.llama_text_processor import llama_text_processor
from agents.research_coordinator import ResearchCoordinator

logger = logging.getLogger(__name__)


def home(request: HttpRequest) -> HttpResponse:
    """Enhanced home page with dashboard features."""
    context = {
        "recent_searches": [],
        "bookmarked_papers": [],
        "active_projects": [],
        "recent_podcasts": [],
        "pending_alerts": 0,
    }

    if request.user.is_authenticated:
        # Get user's recent activity
        context["recent_searches"] = SearchHistory.objects.filter(
            user=request.user
        ).order_by("-created_at")[:5]

        context["bookmarked_papers"] = (
            PaperBookmark.objects.filter(user=request.user)
            .select_related("paper")
            .order_by("-created_at")[:5]
        )

        # Temporarily handle missing columns gracefully
        try:
            context["active_projects"] = (
                ResearchProject.objects.filter(
                    Q(owner=request.user) | Q(collaborators=request.user)
                )
                .filter(status__in=["planning", "active"])
                .order_by("-updated_at")[:3]
            )
        except Exception as e:
            # Handle missing database columns gracefully
            context["active_projects"] = []

        context["recent_podcasts"] = PodcastEpisode.objects.filter(
            creator=request.user
        ).order_by("-created_at")[:3]

        context["pending_alerts"] = ResearchAlert.objects.filter(
            user=request.user, is_active=True
        ).count()

    return render(request, "core/home.html", context)


# Authentication Views
@csrf_protect
@never_cache
def register_view(request: HttpRequest) -> HttpResponse:
    """User registration view."""
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            messages.success(
                request, f"Account created for {username}! You can now log in."
            )
            return redirect("login")
    else:
        form = CustomUserCreationForm()

    return render(request, "core/register.html", {"form": form})


@csrf_protect
@never_cache
def login_view(request: HttpRequest) -> HttpResponse:
    """Custom login view."""
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = CustomLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            remember_me = form.cleaned_data.get("remember_me", False)

            # Try to authenticate with username or email
            user = authenticate(request, username=username, password=password)
            if not user:
                # Try with email
                try:
                    user_obj = User.objects.get(email=username)
                    user = authenticate(
                        request, username=user_obj.username, password=password
                    )
                except User.DoesNotExist:
                    pass

            if user is not None:
                login(request, user)
                if not remember_me:
                    request.session.set_expiry(0)  # Session expires when browser closes
                messages.success(
                    request, f"Welcome back, {user.first_name or user.username}!"
                )
                next_url = request.GET.get("next", "home")
                return redirect(next_url)
            else:
                messages.error(request, "Invalid username/email or password.")
    else:
        form = CustomLoginForm()

    return render(request, "core/login.html", {"form": form})


@login_required
def logout_view(request: HttpRequest) -> HttpResponse:
    """User logout view."""
    user_name = request.user.first_name or request.user.username
    logout(request)
    messages.success(
        request, f"You have been logged out successfully. See you later, {user_name}!"
    )
    return redirect("home")


@login_required
def enhanced_search(request: HttpRequest) -> HttpResponse:
    """Enhanced literature search with multiple sources."""
    results = []
    query = ""
    total_results = 0

    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        sources = request.POST.getlist("sources")
        filters = {}

        # Parse filters
        if request.POST.get("start_date"):
            filters["start_date"] = request.POST.get("start_date")
        if request.POST.get("end_date"):
            filters["end_date"] = request.POST.get("end_date")
        if request.POST.get("min_citations"):
            filters["min_citations"] = request.POST.get("min_citations")
        if request.POST.get("open_access_only"):
            filters["open_access_only"] = True

        if query:
            # Perform search
            search_service = EnhancedLiteratureSearchService()
            try:
                # Run async function in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                search_results = loop.run_until_complete(
                    search_service.unified_search(query, sources, filters, limit=50)
                )
                loop.close()

                results = search_results.get("papers", [])
                total_results = search_results.get("total", 0)

                # Save search history
                SearchHistory.objects.create(
                    user=request.user,
                    query=query,
                    search_type="unified",
                    sources=json.dumps(sources),
                    filters=json.dumps(filters),
                    results_count=total_results,
                )

                messages.success(
                    request, f"Found {total_results} papers matching your query."
                )

            except Exception as e:
                logger.error(f"Search error: {e}")
                messages.error(request, "Search failed. Please try again.")

    # Get available sources
    available_sources = [
        {
            "id": "arxiv",
            "name": "arXiv",
            "description": "Physics, Mathematics, Computer Science",
        },
        {
            "id": "semantic_scholar",
            "name": "Semantic Scholar",
            "description": "Multi-disciplinary",
        },
        {"id": "crossref", "name": "CrossRef", "description": "Academic publications"},
    ]

    context = {
        "query": query,
        "results": results,
        "total_results": total_results,
        "available_sources": available_sources,
        "recent_searches": SearchHistory.objects.filter(user=request.user).order_by(
            "-created_at"
        )[:5],
    }

    return render(request, "core/enhanced_search.html", context)


@login_required
def podcast_generator(request: HttpRequest) -> HttpResponse:
    """Generate podcasts from research papers."""
    podcasts = PodcastEpisode.objects.filter(creator=request.user).order_by(
        "-created_at"
    )

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "generate":
            paper_content = request.POST.get("paper_content", "").strip()
            title = request.POST.get("title", "").strip()
            style = request.POST.get("style", "summary")
            duration = int(request.POST.get("duration", 300))
            voice = request.POST.get("voice", "alloy")

            if paper_content and title:
                try:
                    # Generate podcast
                    podcast_service = PodcastGenerationService()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    if style == "interview":
                        result = loop.run_until_complete(
                            podcast_service.generate_interview_style_podcast(
                                paper_content, duration
                            )
                        )
                    else:
                        result = loop.run_until_complete(
                            podcast_service.generate_paper_summary_podcast(
                                paper_content, style, duration, voice
                            )
                        )

                    loop.close()

                    if result.get("success"):
                        # Save podcast episode
                        episode = PodcastEpisode.objects.create(
                            title=title,
                            description=f"AI-generated {style} podcast",
                            creator=request.user,
                            style=style,
                            duration_seconds=duration,
                            transcript=result["metadata"].get("script", ""),
                            voice_model=voice,
                            generation_time=result["metadata"].get(
                                "generation_time", 0
                            ),
                            file_size=result["metadata"].get("file_size", 0),
                        )

                        # Save audio file
                        if result.get("audio_data"):
                            audio_file = ContentFile(result["audio_data"])
                            episode.audio_file.save(
                                f"podcast_{episode.id}.mp3", audio_file, save=True
                            )

                        messages.success(
                            request, f"Podcast '{title}' generated successfully!"
                        )
                        return redirect("podcast_generator")
                    else:
                        messages.error(
                            request, "Podcast generation failed. Please try again."
                        )

                except Exception as e:
                    logger.error(f"Podcast generation error: {e}")
                    messages.error(
                        request, "An error occurred during podcast generation."
                    )
            else:
                messages.error(request, "Please provide both title and paper content.")

    # Pagination
    paginator = Paginator(podcasts, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "podcasts": page_obj,
        "podcast_styles": [
            ("summary", "Summary Style"),
            ("interview", "Interview Style"),
            ("debate", "Debate Style"),
            ("educational", "Educational Style"),
        ],
        "voice_options": [
            ("alloy", "Alloy"),
            ("echo", "Echo"),
            ("fable", "Fable"),
            ("onyx", "Onyx"),
            ("nova", "Nova"),
            ("shimmer", "Shimmer"),
        ],
    }

    return render(request, "core/podcast_generator.html", context)


@login_required
def video_analyzer(request: HttpRequest) -> HttpResponse:
    """Analyze research videos."""
    analyses = VideoAnalysis.objects.filter(creator=request.user).order_by(
        "-created_at"
    )

    if request.method == "POST":
        video_url = request.POST.get("video_url", "").strip()
        title = request.POST.get("title", "").strip()
        video_type = request.POST.get("video_type", "lecture")
        analysis_type = request.POST.get("analysis_type", "comprehensive")

        if video_url and title:
            try:
                # Create analysis record
                analysis = VideoAnalysis.objects.create(
                    title=title,
                    video_url=video_url,
                    creator=request.user,
                    video_type=video_type,
                    processing_status="processing",
                )

                # Analyze video
                video_service = VideoAnalysisService()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    video_service.analyze_research_video(video_url, analysis_type)
                )

                loop.close()

                # Update analysis with results
                analysis.transcript = result.get("transcript", "")
                analysis.summary = result.get("summary", "")
                analysis.key_concepts = json.dumps(result.get("key_concepts", []))
                analysis.timeline = json.dumps(result.get("timeline", []))
                analysis.topics = json.dumps(result.get("topics", []))
                analysis.sentiment_analysis = json.dumps(
                    result.get("sentiment_analysis", {})
                )
                analysis.processing_time = result.get("processing_time", 0)
                analysis.processing_status = (
                    "completed" if not result.get("error") else "failed"
                )
                analysis.error_message = result.get("error", "")
                analysis.save()

                if analysis.processing_status == "completed":
                    messages.success(
                        request, f"Video analysis for '{title}' completed successfully!"
                    )
                else:
                    messages.error(
                        request, f"Video analysis failed: {analysis.error_message}"
                    )

                return redirect("video_analyzer")

            except Exception as e:
                logger.error(f"Video analysis error: {e}")
                messages.error(request, "An error occurred during video analysis.")
        else:
            messages.error(request, "Please provide both title and video URL.")

    # Pagination
    paginator = Paginator(analyses, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "analyses": page_obj,
        "video_types": [
            ("lecture", "Academic Lecture"),
            ("conference", "Conference Presentation"),
            ("seminar", "Research Seminar"),
            ("interview", "Research Interview"),
            ("tutorial", "Educational Tutorial"),
        ],
    }

    return render(request, "core/video_analyzer.html", context)


@login_required
def research_alerts(request: HttpRequest) -> HttpResponse:
    """Manage research alerts."""
    alerts = ResearchAlert.objects.filter(user=request.user).order_by("-created_at")

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "create":
            title = request.POST.get("title", "").strip()
            alert_type = request.POST.get("alert_type", "keyword")
            keywords = request.POST.get("keywords", "").strip()
            frequency = request.POST.get("frequency", "weekly")

            if title:
                alert = ResearchAlert.objects.create(
                    user=request.user,
                    title=title,
                    alert_type=alert_type,
                    keywords=json.dumps(keywords.split(",") if keywords else []),
                    frequency=frequency,
                )
                messages.success(request, f"Alert '{title}' created successfully!")
            else:
                messages.error(request, "Please provide an alert title.")

        elif action == "toggle":
            alert_id = request.POST.get("alert_id")
            try:
                alert = get_object_or_404(ResearchAlert, id=alert_id, user=request.user)
                alert.is_active = not alert.is_active
                alert.save()
                status = "activated" if alert.is_active else "deactivated"
                messages.success(request, f"Alert '{alert.title}' {status}.")
            except Exception as e:
                messages.error(request, "Failed to update alert.")

        elif action == "delete":
            alert_id = request.POST.get("alert_id")
            try:
                alert = get_object_or_404(ResearchAlert, id=alert_id, user=request.user)
                alert_title = alert.title
                alert.delete()
                messages.success(request, f"Alert '{alert_title}' deleted.")
            except Exception as e:
                messages.error(request, "Failed to delete alert.")

        return redirect("research_alerts")

    context = {
        "alerts": alerts,
        "alert_types": [
            ("keyword", "Keyword Alert"),
            ("author", "Author Alert"),
            ("journal", "Journal Alert"),
            ("citation", "Citation Alert"),
            ("conference", "Conference Alert"),
        ],
        "frequencies": [
            ("daily", "Daily"),
            ("weekly", "Weekly"),
            ("monthly", "Monthly"),
            ("immediate", "Immediate"),
        ],
    }

    return render(request, "core/research_alerts.html", context)


@login_required
def writing_assistant(request: HttpRequest) -> HttpResponse:
    """AI-powered writing assistance."""
    assistance_sessions = WritingAssistance.objects.filter(user=request.user).order_by(
        "-created_at"
    )

    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        task_type = request.POST.get("task_type", "literature_review")
        content = request.POST.get("content", "").strip()
        tone = request.POST.get("tone", "academic")

        if title and content:
            try:
                # Get writing assistance
                writing_service = WritingAssistanceService()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    writing_service.assist_writing(content, task_type, tone)
                )

                loop.close()

                # Save assistance session
                session = WritingAssistance.objects.create(
                    user=request.user,
                    title=title,
                    task_type=task_type,
                    content=content,
                    improved_content=result.get("improved_content", ""),
                    suggestions=json.dumps(result.get("suggestions", [])),
                    style_improvements=json.dumps(result.get("style_improvements", [])),
                    readability_score=result.get("readability_score", 0.0),
                    word_count=result.get("word_count", 0),
                    tone=tone,
                )

                messages.success(
                    request, f"Writing assistance for '{title}' completed!"
                )
                return redirect("writing_assistant")

            except Exception as e:
                logger.error(f"Writing assistance error: {e}")
                messages.error(request, "An error occurred during writing assistance.")
        else:
            messages.error(request, "Please provide both title and content.")

    # Pagination
    paginator = Paginator(assistance_sessions, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "sessions": page_obj,
        "task_types": [
            ("literature_review", "Literature Review"),
            ("abstract", "Abstract"),
            ("introduction", "Introduction"),
            ("methodology", "Methodology"),
            ("results", "Results"),
            ("discussion", "Discussion"),
            ("conclusion", "Conclusion"),
            ("citation", "Citation Formatting"),
        ],
        "tones": [
            ("academic", "Academic"),
            ("formal", "Formal"),
            ("technical", "Technical"),
            ("accessible", "Accessible"),
        ],
    }

    return render(request, "core/writing_assistant.html", context)


@login_required
def collaboration(request: HttpRequest) -> HttpResponse:
    """Research collaboration features."""
    sent_requests = CollaborationRequest.objects.filter(
        requester=request.user
    ).order_by("-created_at")

    received_requests = CollaborationRequest.objects.filter(
        recipient=request.user
    ).order_by("-created_at")

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "send_request":
            recipient_email = request.POST.get("recipient_email", "").strip()
            title = request.POST.get("title", "").strip()
            description = request.POST.get("description", "").strip()
            collaboration_type = request.POST.get("collaboration_type", "research")

            if recipient_email and title and description:
                try:
                    recipient = User.objects.get(email=recipient_email)
                    if recipient != request.user:
                        CollaborationRequest.objects.create(
                            requester=request.user,
                            recipient=recipient,
                            title=title,
                            description=description,
                            collaboration_type=collaboration_type,
                        )
                        messages.success(
                            request, f"Collaboration request sent to {recipient.email}!"
                        )
                    else:
                        messages.error(
                            request, "You cannot send a request to yourself."
                        )
                except User.DoesNotExist:
                    messages.error(request, "User with this email does not exist.")
            else:
                messages.error(request, "Please fill in all required fields.")

        elif action == "respond":
            request_id = request.POST.get("request_id")
            response = request.POST.get("response")  # accept or decline
            response_message = request.POST.get("response_message", "")

            try:
                collab_request = get_object_or_404(
                    CollaborationRequest,
                    id=request_id,
                    recipient=request.user,
                    status="pending",
                )

                collab_request.status = (
                    "accepted" if response == "accept" else "declined"
                )
                collab_request.response_message = response_message
                collab_request.responded_at = datetime.now()
                collab_request.save()

                action_text = "accepted" if response == "accept" else "declined"
                messages.success(request, f"Collaboration request {action_text}.")

            except Exception as e:
                messages.error(request, "Failed to respond to collaboration request.")

        return redirect("collaboration")

    context = {
        "sent_requests": sent_requests,
        "received_requests": received_requests,
        "collaboration_types": [
            ("research", "Research Project"),
            ("review", "Paper Review"),
            ("coauthoring", "Co-authoring"),
            ("mentoring", "Mentoring"),
            ("data_sharing", "Data Sharing"),
        ],
    }

    return render(request, "core/collaboration.html", context)


@login_required
def user_preferences(request: HttpRequest) -> HttpResponse:
    """User preferences and settings."""
    preferences, created = UserPreferences.objects.get_or_create(user=request.user)

    if request.method == "POST":
        # Update preferences
        preferences.max_search_results = int(request.POST.get("max_search_results", 20))
        preferences.preferred_languages = json.dumps(
            request.POST.getlist("preferred_languages")
        )
        preferences.email_notifications = bool(request.POST.get("email_notifications"))
        preferences.web_notifications = bool(request.POST.get("web_notifications"))
        preferences.notification_frequency = request.POST.get(
            "notification_frequency", "weekly"
        )
        preferences.preferred_podcast_style = request.POST.get(
            "preferred_podcast_style", "summary"
        )
        preferences.preferred_voice = request.POST.get("preferred_voice", "alloy")
        preferences.theme = request.POST.get("theme", "light")
        preferences.items_per_page = int(request.POST.get("items_per_page", 25))

        preferences.save()
        messages.success(request, "Preferences updated successfully!")
        return redirect("user_preferences")

    context = {
        "preferences": preferences,
        "available_languages": [
            ("en", "English"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("pt", "Portuguese"),
            ("zh", "Chinese"),
            ("ja", "Japanese"),
        ],
    }

    return render(request, "core/user_preferences.html", context)


@login_required
def paper_detail(request: HttpRequest, paper_id: str) -> HttpResponse:
    """Display paper details with actions."""
    paper = get_object_or_404(Paper, id=paper_id)
    is_bookmarked = PaperBookmark.objects.filter(
        user=request.user, paper=paper
    ).exists()

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "bookmark":
            bookmark, created = PaperBookmark.objects.get_or_create(
                user=request.user,
                paper=paper,
                defaults={"notes": request.POST.get("notes", "")},
            )
            if created:
                messages.success(request, "Paper bookmarked!")
            else:
                messages.info(request, "Paper already bookmarked.")

        elif action == "remove_bookmark":
            PaperBookmark.objects.filter(user=request.user, paper=paper).delete()
            messages.success(request, "Bookmark removed.")

        return redirect("paper_detail", paper_id=paper_id)

    context = {
        "paper": paper,
        "is_bookmarked": is_bookmarked,
    }

    return render(request, "core/paper_detail.html", context)


# API Views for AJAX requests
@login_required
@csrf_exempt
def api_search(request: HttpRequest) -> JsonResponse:
    """API endpoint for search functionality."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "")
        sources = data.get("sources", ["arxiv"])

        if not query:
            return JsonResponse({"error": "Query is required"}, status=400)

        search_service = EnhancedLiteratureSearchService()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = loop.run_until_complete(
            search_service.unified_search(query, sources, limit=20)
        )

        loop.close()

        return JsonResponse(results)

    except Exception as e:
        logger.error(f"API search error: {e}")
        return JsonResponse({"error": "Search failed"}, status=500)


@login_required
def download_podcast(request: HttpRequest, podcast_id: str) -> HttpResponse:
    """Download podcast audio file."""
    podcast = get_object_or_404(PodcastEpisode, id=podcast_id, creator=request.user)

    if podcast.audio_file:
        # Increment play count
        podcast.play_count += 1
        podcast.save()

        return FileResponse(
            podcast.audio_file,
            as_attachment=True,
            filename=f"{podcast.title}.mp3",
        )
    else:
        messages.error(request, "Audio file not found.")
        return redirect("podcast_generator")


# ========================================
# NEW ADVANCED FEATURES FROM NOTEBOOK
# ========================================

@login_required
def paper_analysis_view(request: HttpRequest) -> HttpResponse:
    """Advanced AI-powered paper analysis."""
    analysis_results = []
    
    if request.method == "POST":
        paper_data = {
            "title": request.POST.get("title", "").strip(),
            "abstract": request.POST.get("abstract", "").strip(),
            "authors": [a.strip() for a in request.POST.get("authors", "").split(",") if a.strip()],
            "publication_date": request.POST.get("publication_date", ""),
            "journal": request.POST.get("journal", "").strip(),
            "keywords": [k.strip() for k in request.POST.get("keywords", "").split(",") if k.strip()],
        }
        
        if paper_data["title"] and paper_data["abstract"]:
            try:
                # Import the service
                from backend.app.services.paper_analysis_service import paper_analysis_service
                
                # Run analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                analysis_results = loop.run_until_complete(
                    paper_analysis_service.analyze_paper(paper_data)
                )
                loop.close()
                
                messages.success(request, "Paper analysis completed successfully!")
                
            except Exception as e:
                logger.error(f"Paper analysis error: {e}")
                messages.error(request, f"Analysis failed: {str(e)}")
        else:
            messages.error(request, "Please provide at least title and abstract.")
    
    context = {
        "analysis_results": analysis_results,
        "research_fields": [
            "Computer Science", "Machine Learning", "Natural Language Processing",
            "Computer Vision", "Biomedical", "Physics", "Mathematics", 
            "Social Sciences", "Engineering", "Environmental"
        ]
    }
    
    return render(request, "core/paper_analysis.html", context)


@login_required
def trend_analysis_view(request: HttpRequest) -> HttpResponse:
    """Research trend analysis and visualization."""
    trend_results = {}
    
    if request.method == "POST":
        field = request.POST.get("field", "").strip()
        time_range = request.POST.get("time_range", "5_years")
        
        if field:
            try:
                # Import the enhanced research API view for trend analysis
                from backend.app.api.v1.enhanced_research import ResearchTrendAnalysisView
                
                # Create a mock request object for the API call
                class MockRequest:
                    def __init__(self, method="GET"):
                        self.method = method
                        self.GET = {"time_range": time_range}
                
                api_view = ResearchTrendAnalysisView()
                
                # Run trend analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    api_view._analyze_field_trends(field, time_range)
                )
                loop.close()
                
                trend_results = response
                messages.success(request, f"Trend analysis for '{field}' completed!")
                
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                messages.error(request, f"Trend analysis failed: {str(e)}")
        else:
            messages.error(request, "Please provide a research field.")
    
    context = {
        "trend_results": trend_results,
        "popular_fields": [
            "Machine Learning", "Artificial Intelligence", "Deep Learning",
            "Natural Language Processing", "Computer Vision", "Robotics",
            "Quantum Computing", "Biomedical Engineering", "Climate Science",
            "Neuroscience", "Blockchain", "Cybersecurity"
        ],
        "time_ranges": [
            ("1_year", "Last 1 Year"),
            ("3_years", "Last 3 Years"), 
            ("5_years", "Last 5 Years"),
            ("10_years", "Last 10 Years")
        ]
    }
    
    return render(request, "core/trend_analysis.html", context)


@login_required
def research_gaps_view(request: HttpRequest) -> HttpResponse:
    """Research gap identification and analysis."""
    gap_results = {}
    
    if request.method == "POST":
        field = request.POST.get("field", "").strip()
        depth = request.POST.get("depth", "comprehensive")
        
        if field:
            try:
                # Import the enhanced research API view for gap analysis
                from backend.app.api.v1.enhanced_research import ResearchGapAnalysisView
                
                # Create a mock request object for the API call
                class MockRequest:
                    def __init__(self, method="GET"):
                        self.method = method
                        self.GET = {"depth": depth}
                
                api_view = ResearchGapAnalysisView()
                
                # Run gap analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                gaps = loop.run_until_complete(
                    api_view._find_research_gaps(field, depth)
                )
                loop.close()
                
                gap_results = {
                    "field": field,
                    "gaps": gaps,
                    "gap_count": len(gaps)
                }
                
                messages.success(request, f"Research gap analysis for '{field}' completed!")
                
            except Exception as e:
                logger.error(f"Research gap analysis error: {e}")
                messages.error(request, f"Gap analysis failed: {str(e)}")
        else:
            messages.error(request, "Please provide a research field.")
    
    context = {
        "gap_results": gap_results,
        "analysis_depths": [
            ("quick", "Quick Analysis"),
            ("comprehensive", "Comprehensive Analysis"),
            ("deep", "Deep Analysis")
        ],
        "popular_fields": [
            "Machine Learning", "Artificial Intelligence", "Deep Learning",
            "Natural Language Processing", "Computer Vision", "Robotics",
            "Quantum Computing", "Biomedical Engineering", "Climate Science",
            "Neuroscience", "Blockchain", "Cybersecurity"
        ]
    }
    
    return render(request, "core/research_gaps.html", context)


@login_required
def recommendations_view(request: HttpRequest) -> HttpResponse:
    """AI-powered personalized research recommendations."""
    recommendations = {}
    
    if request.method == "POST":
        interests = [i.strip() for i in request.POST.get("interests", "").split(",") if i.strip()]
        context_text = request.POST.get("context", "").strip()
        recommendation_type = request.POST.get("type", "papers")
        
        if interests:
            try:
                # Import the enhanced research API view for recommendations
                from backend.app.api.v1.enhanced_research import SmartRecommendationView
                
                api_view = SmartRecommendationView()
                
                # Run recommendations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                recommendations = loop.run_until_complete(
                    api_view._generate_recommendations(interests, context_text, recommendation_type)
                )
                loop.close()
                
                messages.success(request, f"Generated {len(recommendations)} {recommendation_type} recommendations!")
                
            except Exception as e:
                logger.error(f"Recommendations error: {e}")
                messages.error(request, f"Recommendations failed: {str(e)}")
        else:
            messages.error(request, "Please provide your research interests.")
    
    context = {
        "recommendations": recommendations,
        "recommendation_types": [
            ("papers", "Research Papers"),
            ("topics", "Research Topics"),
            ("researchers", "Researchers")
        ],
        "sample_interests": [
            "Machine Learning, Neural Networks, Computer Vision",
            "Climate Science, Environmental Modeling, Sustainability",
            "Biomedical Engineering, Medical Imaging, Drug Discovery",
            "Natural Language Processing, Text Mining, Sentiment Analysis"
        ]
    }
    
    return render(request, "core/recommendations.html", context)


@login_required 
def paper_comparison_view(request: HttpRequest) -> HttpResponse:
    """Compare multiple research papers for similarities and differences."""
    comparison_results = {}
    
    if request.method == "POST":
        # Get paper data from form
        papers_data = []
        
        # Extract papers from form (up to 5 papers)
        for i in range(1, 6):  # Support up to 5 papers
            title = request.POST.get(f"paper_{i}_title", "").strip()
            abstract = request.POST.get(f"paper_{i}_abstract", "").strip()
            authors = request.POST.get(f"paper_{i}_authors", "").strip()
            
            if title and abstract:
                paper = {
                    "title": title,
                    "abstract": abstract,
                    "authors": [a.strip() for a in authors.split(",") if a.strip()],
                    "publication_date": request.POST.get(f"paper_{i}_date", ""),
                    "journal": request.POST.get(f"paper_{i}_journal", "").strip()
                }
                papers_data.append(paper)
        
        if len(papers_data) >= 2:
            try:
                # Import the paper analysis service
                from backend.app.services.paper_analysis_service import paper_analysis_service
                
                # Run comparison
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                comparison_results = loop.run_until_complete(
                    paper_analysis_service.compare_papers(papers_data)
                )
                loop.close()
                
                messages.success(request, f"Comparison of {len(papers_data)} papers completed!")
                
            except Exception as e:
                logger.error(f"Paper comparison error: {e}")
                messages.error(request, f"Comparison failed: {str(e)}")
        else:
            messages.error(request, "Please provide at least 2 papers for comparison.")
    
    context = {
        "comparison_results": comparison_results,
        "max_papers": 5
    }
    
    return render(request, "core/paper_comparison.html", context)


@login_required
def research_coordinator_view(request: HttpRequest) -> HttpResponse:
    """Manage and execute multi-agent research projects."""
    # In a real application, the coordinator should be a singleton
    # or managed more carefully.
    coordinator = ResearchCoordinator()
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "create_project":
            title = request.POST.get("title", "").strip()
            description = request.POST.get("description", "").strip()
            keywords = request.POST.get("keywords", "").strip()
            objectives = request.POST.get("objectives", "").strip()

            if title and description and keywords:
                # Create Django DB record
                project_db = ResearchProject.objects.create(
                    title=title,
                    description=description,
                    keywords=keywords,
                    objectives=objectives,
                    owner=request.user,
                    status='pending'
                )
                messages.success(request, f"Project '{title}' created. You can now execute it.")
                return redirect('research_coordinator')

            else:
                messages.error(request, "Please fill all required fields.")

        elif action == "execute_project":
            project_id = request.POST.get("project_id")
            project_db = get_object_or_404(ResearchProject, id=project_id, owner=request.user)
            
            project_db.status = 'processing'
            project_db.save()
            
            messages.info(request, f"Executing project '{project_db.title}'. This may take a few moments.")
            
            try:
                # In a real app, this would be a background task (e.g., with Celery).
                # For this demo, we run it synchronously.
                project_data = {
                    'title': project_db.title,
                    'description': project_db.description,
                    'keywords': [k.strip() for k in project_db.keywords.split(',')],
                    'objectives': [o.strip() for o in project_db.objectives.splitlines()]
                }

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                project_in_coordinator = loop.run_until_complete(
                    coordinator.create_research_project(project_data)
                )
                
                results = loop.run_until_complete(
                    coordinator.execute_project(project_in_coordinator.id)
                )
                
                loop.close()

                project_db.results = json.dumps(results, indent=2, default=str)
                project_db.status = 'completed'
                project_db.save()
                
                messages.success(request, f"Project '{project_db.title}' executed successfully.")

            except Exception as e:
                logger.error(f"Error executing project {project_id}: {e}", exc_info=True)
                project_db.status = 'failed'
                project_db.results = json.dumps({'error': str(e)})
                project_db.save()
                messages.error(request, f"Failed to execute project: {e}")

            return redirect('research_coordinator')
            
        elif action == "view_results":
            project_id = request.POST.get("project_id")
            project_db = get_object_or_404(ResearchProject, id=project_id, owner=request.user)
            
            try:
                results = json.loads(project_db.results)
            except (json.JSONDecodeError, TypeError):
                results = {"error": "Results are not available or are in an invalid format."}

            return JsonResponse(results)

    projects = ResearchProject.objects.filter(owner=request.user).order_by("-created_at")
    
    # Pagination
    paginator = Paginator(projects, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        'projects': page_obj,
    }
    
    return render(request, "core/research_coordinator.html", context)


# Legacy views for backward compatibility
def dataset_view(request: HttpRequest) -> HttpResponse:
    """Legacy dataset view - redirects to enhanced search."""
    return redirect("enhanced_search")


def training_view(request: HttpRequest) -> HttpResponse:
    """Legacy training view - redirects to user preferences."""
    return redirect("user_preferences")


def research_view(request: HttpRequest) -> HttpResponse:
    """Legacy research view - redirects to enhanced search."""
    return redirect("enhanced_search")


def prompt_view(request: HttpRequest) -> HttpResponse:
    """Legacy prompt view - redirects to writing assistant."""
    return redirect("writing_assistant")
