"""Enhanced Podcast Generation Views."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from django.contrib import messages
from django.conf import settings
import os

from core.models import Paper, PodcastEpisode

logger = logging.getLogger(__name__)


@login_required
def list_podcasts(request):
    """List user's podcasts."""
    podcasts = PodcastEpisode.objects.filter(
        creator=request.user
    ).select_related('paper').order_by('-created_at')
    
    # Apply filters
    search_query = request.GET.get('search', '')
    style_filter = request.GET.get('style', '')
    
    if search_query:
        podcasts = podcasts.filter(
            Q(title__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(paper__title__icontains=search_query)
        )
    
    if style_filter:
        podcasts = podcasts.filter(style=style_filter)
    
    paginator = Paginator(podcasts, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    if request.headers.get('Accept') == 'application/json':
        podcasts_data = []
        for podcast in page_obj:
            authors_list = []
            try:
                authors_list = json.loads(podcast.paper.authors) if podcast.paper.authors else []
            except (json.JSONDecodeError, TypeError):
                authors_list = [podcast.paper.authors] if podcast.paper.authors else []
            
            podcasts_data.append({
                'id': str(podcast.id),
                'title': podcast.title,
                'description': podcast.description,
                'style': podcast.style,
                'duration_seconds': podcast.duration_seconds,
                'created_at': podcast.created_at.isoformat(),
                'play_count': podcast.play_count,
                'like_count': podcast.like_count,
                'paper': {
                    'id': str(podcast.paper.id),
                    'title': podcast.paper.title,
                    'authors': authors_list
                },
                'has_audio': bool(podcast.audio_file),
                'audio_url': podcast.audio_file.url if podcast.audio_file else None
            })
        
        return JsonResponse({
            'success': True,
            'podcasts': podcasts_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'page_number': page_obj.number,
            'total_pages': paginator.num_pages,
            'total_count': paginator.count
        })
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'style_filter': style_filter,
        'available_styles': [
            ('summary', 'Summary Style'),
            ('interview', 'Interview Style'),
            ('debate', 'Debate Style'),
            ('educational', 'Educational Style')
        ],
        'total_podcasts': paginator.count
    }
    
    return render(request, 'podcast_generation/list.html', context)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def generate_podcast(request):
    """Generate a podcast from research papers."""
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST
        
        paper_id = data.get('paper_id')
        title = data.get('title', '')
        style = data.get('style', 'summary')
        voice_model = data.get('voice_model', 'alloy')
        language = data.get('language', 'en')
        custom_prompt = data.get('custom_prompt', '')
        
        if not paper_id:
            return JsonResponse({
                'success': False,
                'error': 'Paper ID is required'
            }, status=400)
        
        paper = get_object_or_404(Paper, id=paper_id)
        
        # Generate title if not provided
        if not title:
            title = f"Research Summary: {paper.title[:50]}..."
        
        # Create podcast episode record
        podcast = PodcastEpisode.objects.create(
            title=title,
            description=f"AI-generated podcast discussing: {paper.title}",
            paper=paper,
            creator=request.user,
            style=style,
            voice_model=voice_model,
            language=language,
            duration_seconds=0  # Will be updated after generation
        )
        
        # Generate script based on paper content
        script = generate_podcast_script(paper, style, custom_prompt)
        podcast.script = script
        
        # Generate transcript (for now, same as script)
        podcast.transcript = script
        
        # Mock audio generation (in real implementation, you'd use TTS)
        # For now, we'll just estimate duration based on script length
        estimated_duration = len(script.split()) * 0.4  # ~0.4 seconds per word
        podcast.duration_seconds = int(estimated_duration)
        
        podcast.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Podcast generation started',
            'podcast_id': str(podcast.id),
            'estimated_duration': int(estimated_duration)
        })
        
    except Paper.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Paper not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Podcast generation error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during generation'
        }, status=500)


def podcast_status(request, podcast_id):
    """Get podcast generation status."""
    try:
        podcast = get_object_or_404(PodcastEpisode, id=podcast_id)
        
        # Check if user has access
        if request.user.is_authenticated and podcast.creator != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        return JsonResponse({
            'success': True,
            'podcast_id': str(podcast.id),
            'status': 'completed',  # For now, always completed
            'progress': 100,
            'title': podcast.title,
            'duration_seconds': podcast.duration_seconds,
            'created_at': podcast.created_at.isoformat(),
            'has_audio': bool(podcast.audio_file)
        })
        
    except PodcastEpisode.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Podcast not found'
        }, status=404)


def podcast_detail(request, podcast_id):
    """Get detailed podcast information."""
    try:
        podcast = get_object_or_404(PodcastEpisode, id=podcast_id)
        
        # Check if user has access
        if request.user.is_authenticated and podcast.creator != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        authors_list = []
        try:
            authors_list = json.loads(podcast.paper.authors) if podcast.paper.authors else []
        except (json.JSONDecodeError, TypeError):
            authors_list = [podcast.paper.authors] if podcast.paper.authors else []
        
        podcast_data = {
            'id': str(podcast.id),
            'title': podcast.title,
            'description': podcast.description,
            'style': podcast.style,
            'voice_model': podcast.voice_model,
            'language': podcast.language,
            'duration_seconds': podcast.duration_seconds,
            'created_at': podcast.created_at.isoformat(),
            'play_count': podcast.play_count,
            'like_count': podcast.like_count,
            'share_count': podcast.share_count,
            'script': podcast.script,
            'transcript': podcast.transcript,
            'paper': {
                'id': str(podcast.paper.id),
                'title': podcast.paper.title,
                'authors': authors_list,
                'abstract': podcast.paper.abstract,
                'publication_date': podcast.paper.publication_date.isoformat() if podcast.paper.publication_date else None
            },
            'has_audio': bool(podcast.audio_file),
            'audio_url': podcast.audio_file.url if podcast.audio_file else None
        }
        
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': True,
                'podcast': podcast_data
            })
        
        return render(request, 'podcast_generation/detail.html', {
            'podcast': podcast_data
        })
        
    except PodcastEpisode.DoesNotExist:
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': False,
                'error': 'Podcast not found'
            }, status=404)
        
        messages.error(request, 'Podcast not found')
        return redirect('podcast_generation:list')


def download_podcast(request, podcast_id):
    """Download generated podcast audio file."""
    try:
        podcast = get_object_or_404(PodcastEpisode, id=podcast_id)
        
        # Check if user has access
        if request.user.is_authenticated and podcast.creator != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        if not podcast.audio_file:
            return JsonResponse({
                'success': False,
                'error': 'Audio file not available'
            }, status=404)
        
        # Increment play count
        podcast.play_count += 1
        podcast.save(update_fields=['play_count'])
        
        return FileResponse(
            podcast.audio_file,
            as_attachment=True,
            filename=f"podcast_{podcast.id}.mp3"
        )
        
    except PodcastEpisode.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Podcast not found'
        }, status=404)


@login_required
@require_http_methods(["DELETE"])
def delete_podcast(request, podcast_id):
    """Delete a podcast."""
    try:
        podcast = get_object_or_404(
            PodcastEpisode,
            id=podcast_id,
            creator=request.user
        )
        
        # Delete audio file if exists
        if podcast.audio_file:
            try:
                os.remove(podcast.audio_file.path)
            except OSError:
                pass
        
        podcast.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Podcast "{podcast.title}" deleted'
        })
        
    except PodcastEpisode.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Podcast not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Delete podcast error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def like_podcast(request, podcast_id):
    """Like/unlike a podcast."""
    try:
        podcast = get_object_or_404(PodcastEpisode, id=podcast_id)
        action = json.loads(request.body).get('action', 'like')
        
        if action == 'like':
            podcast.like_count += 1
        elif action == 'unlike' and podcast.like_count > 0:
            podcast.like_count -= 1
        
        podcast.save(update_fields=['like_count'])
        
        return JsonResponse({
            'success': True,
            'like_count': podcast.like_count,
            'action': action
        })
        
    except PodcastEpisode.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Podcast not found'
        }, status=404)


@login_required
def podcast_generator(request):
    """Podcast generation interface."""
    # Get user's papers for podcast generation
    from core.models import PaperBookmark
    
    bookmarked_papers = PaperBookmark.objects.filter(
        user=request.user
    ).select_related('paper').order_by('-created_at')[:20]
    
    # Get recent papers
    recent_papers = Paper.objects.order_by('-created_at')[:10]
    
    context = {
        'bookmarked_papers': bookmarked_papers,
        'recent_papers': recent_papers,
        'available_styles': [
            ('summary', 'Summary Style'),
            ('interview', 'Interview Style'),
            ('debate', 'Debate Style'),
            ('educational', 'Educational Style')
        ],
        'voice_models': [
            ('alloy', 'Alloy'),
            ('echo', 'Echo'),
            ('fable', 'Fable'),
            ('onyx', 'Onyx'),
            ('nova', 'Nova'),
            ('shimmer', 'Shimmer')
        ]
    }
    
    return render(request, 'podcast_generation/generator.html', context)


def generate_podcast_script(paper, style, custom_prompt=''):
    """Generate podcast script based on paper content and style."""
    authors_list = []
    try:
        authors_list = json.loads(paper.authors) if paper.authors else []
    except (json.JSONDecodeError, TypeError):
        authors_list = [paper.authors] if paper.authors else []
    
    authors_str = ', '.join(authors_list) if authors_list else 'Unknown authors'
    
    if style == 'summary':
        script = f"""
Welcome to Research Insights, where we break down the latest academic papers.

Today we're discussing '{paper.title}' by {authors_str}.

{f"Published in {paper.journal}" if paper.journal else "This research paper"} explores fascinating insights in the field.

Let me walk you through the key findings:

{paper.abstract}

{f"The authors conclude that this research {paper.summary}" if paper.summary else "This work contributes significantly to our understanding of the topic."}

{"" if not custom_prompt else f"Additionally, {custom_prompt}"}

That's our summary for today. Thank you for listening to Research Insights.
        """.strip()
    
    elif style == 'interview':
        script = f"""
Host: Welcome to Academic Conversations. Today I'm joined by the research team behind '{paper.title}'. 

Could you tell us about your research?

Researcher: Certainly. Our work, published {f"in {paper.journal}" if paper.journal else "recently"}, investigates {paper.title.lower()}.

Host: What motivated this research?

Researcher: {paper.abstract[:200]}...

Host: What are the key findings?

Researcher: {paper.summary if paper.summary else "Our research reveals significant insights that advance the field."}

Host: {"" if not custom_prompt else f"{custom_prompt}"}

Host: Thank you for sharing your insights with us today.

Researcher: Thank you for having me.
        """.strip()
    
    elif style == 'debate':
        script = f"""
Moderator: Welcome to Research Debates. Today we're discussing '{paper.title}' by {authors_str}.

Advocate: This research represents a significant breakthrough. {paper.abstract[:150]}...

Critic: While the work has merit, we must consider the limitations...

Advocate: The methodology is sound, and the findings clearly show...

Critic: However, the sample size and scope raise questions about generalizability...

Moderator: {"" if not custom_prompt else f"{custom_prompt}"}

Moderator: Both perspectives highlight the complexity of this research area.
        """.strip()
    
    else:  # educational
        script = f"""
Welcome to Learning Labs, where we make research accessible.

Today's lesson: Understanding '{paper.title}'

First, let's define our terms and context...

{paper.abstract}

Key concept 1: The methodology used in this research...
Key concept 2: The implications of these findings...
Key concept 3: How this connects to broader research...

{f"Summary: {paper.summary}" if paper.summary else "In summary, this research advances our understanding significantly."}

{"" if not custom_prompt else f"Additional insight: {custom_prompt}"}

That concludes today's lesson. Keep learning!
        """.strip()
    
    return script
