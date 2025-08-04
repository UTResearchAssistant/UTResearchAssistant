"""Enhanced Video Analysis Views."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import re
from urllib.parse import urlparse

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from django.contrib import messages

from core.models import VideoAnalysis

logger = logging.getLogger(__name__)


@login_required
def list_analyses(request):
    """List user's video analyses."""
    analyses = VideoAnalysis.objects.filter(
        creator=request.user
    ).order_by('-created_at')
    
    # Apply filters
    search_query = request.GET.get('search', '')
    status_filter = request.GET.get('status', '')
    video_type_filter = request.GET.get('video_type', '')
    
    if search_query:
        analyses = analyses.filter(
            Q(title__icontains=search_query) |
            Q(video_url__icontains=search_query) |
            Q(summary__icontains=search_query)
        )
    
    if status_filter:
        analyses = analyses.filter(processing_status=status_filter)
    
    if video_type_filter:
        analyses = analyses.filter(video_type=video_type_filter)
    
    paginator = Paginator(analyses, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    if request.headers.get('Accept') == 'application/json':
        analyses_data = []
        for analysis in page_obj:
            key_concepts = []
            topics = []
            try:
                key_concepts = json.loads(analysis.key_concepts) if analysis.key_concepts else []
                topics = json.loads(analysis.topics) if analysis.topics else []
            except (json.JSONDecodeError, TypeError):
                pass
            
            analyses_data.append({
                'id': str(analysis.id),
                'title': analysis.title,
                'video_url': analysis.video_url,
                'video_type': analysis.video_type,
                'language': analysis.language,
                'duration_seconds': analysis.duration_seconds,
                'processing_status': analysis.processing_status,
                'created_at': analysis.created_at.isoformat(),
                'summary': analysis.summary,
                'key_concepts': key_concepts,
                'topics': topics,
                'has_transcript': bool(analysis.transcript)
            })
        
        return JsonResponse({
            'success': True,
            'analyses': analyses_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'page_number': page_obj.number,
            'total_pages': paginator.num_pages,
            'total_count': paginator.count
        })
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'status_filter': status_filter,
        'video_type_filter': video_type_filter,
        'available_statuses': [
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        'available_types': [
            ('lecture', 'Academic Lecture'),
            ('conference', 'Conference Presentation'),
            ('seminar', 'Research Seminar'),
            ('interview', 'Research Interview'),
            ('tutorial', 'Educational Tutorial')
        ],
        'total_analyses': paginator.count
    }
    
    return render(request, 'video_analysis/list.html', context)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def analyze_video(request):
    """Analyze a video for research content."""
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST
        
        video_url = data.get('video_url', '').strip()
        title = data.get('title', '')
        video_type = data.get('video_type', 'lecture')
        language = data.get('language', 'en')
        
        if not video_url:
            return JsonResponse({
                'success': False,
                'error': 'Video URL is required'
            }, status=400)
        
        # Validate URL
        if not is_valid_video_url(video_url):
            return JsonResponse({
                'success': False,
                'error': 'Invalid video URL. Please provide a valid YouTube, Vimeo, or other video URL.'
            }, status=400)
        
        # Generate title if not provided
        if not title:
            title = f"Video Analysis - {extract_video_title(video_url)}"
        
        # Create video analysis record
        analysis = VideoAnalysis.objects.create(
            title=title,
            video_url=video_url,
            creator=request.user,
            video_type=video_type,
            language=language,
            processing_status='processing'
        )
        
        # Start analysis process (mock implementation)
        process_video_analysis(analysis)
        
        return JsonResponse({
            'success': True,
            'message': 'Video analysis started',
            'analysis_id': str(analysis.id)
        })
        
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during analysis'
        }, status=500)


def analysis_status(request, analysis_id):
    """Get video analysis status."""
    try:
        analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
        
        # Check if user has access
        if request.user.is_authenticated and analysis.creator != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        progress = 100 if analysis.processing_status == 'completed' else (
            50 if analysis.processing_status == 'processing' else 0
        )
        
        return JsonResponse({
            'success': True,
            'analysis_id': str(analysis.id),
            'status': analysis.processing_status,
            'progress': progress,
            'title': analysis.title,
            'created_at': analysis.created_at.isoformat(),
            'error_message': analysis.error_message
        })
        
    except VideoAnalysis.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Analysis not found'
        }, status=404)


def analysis_results(request, analysis_id):
    """Get video analysis results."""
    try:
        analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
        
        # Check if user has access
        if request.user.is_authenticated and analysis.creator != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Access denied'
            }, status=403)
        
        if analysis.processing_status != 'completed':
            return JsonResponse({
                'success': False,
                'error': 'Analysis not completed yet'
            }, status=400)
        
        # Parse JSON fields
        key_concepts = []
        timeline = []
        topics = []
        sentiment_analysis = {}
        
        try:
            key_concepts = json.loads(analysis.key_concepts) if analysis.key_concepts else []
            timeline = json.loads(analysis.timeline) if analysis.timeline else []
            topics = json.loads(analysis.topics) if analysis.topics else []
            sentiment_analysis = json.loads(analysis.sentiment_analysis) if analysis.sentiment_analysis else {}
        except (json.JSONDecodeError, TypeError):
            pass
        
        results_data = {
            'id': str(analysis.id),
            'title': analysis.title,
            'video_url': analysis.video_url,
            'video_type': analysis.video_type,
            'language': analysis.language,
            'duration_seconds': analysis.duration_seconds,
            'transcript': analysis.transcript,
            'summary': analysis.summary,
            'key_concepts': key_concepts,
            'timeline': timeline,
            'topics': topics,
            'sentiment_analysis': sentiment_analysis,
            'processing_time': analysis.processing_time,
            'created_at': analysis.created_at.isoformat()
        }
        
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': True,
                'analysis': results_data
            })
        
        return render(request, 'video_analysis/results.html', {
            'analysis': results_data
        })
        
    except VideoAnalysis.DoesNotExist:
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': False,
                'error': 'Analysis not found'
            }, status=404)
        
        messages.error(request, 'Analysis not found')
        return redirect('video_analysis:list')


@login_required
@require_http_methods(["DELETE"])
def delete_analysis(request, analysis_id):
    """Delete a video analysis."""
    try:
        analysis = get_object_or_404(
            VideoAnalysis,
            id=analysis_id,
            creator=request.user
        )
        
        title = analysis.title
        analysis.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Analysis "{title}" deleted'
        })
        
    except VideoAnalysis.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Analysis not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Delete analysis error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)


@login_required
def video_analyzer(request):
    """Video analysis interface."""
    context = {
        'available_types': [
            ('lecture', 'Academic Lecture'),
            ('conference', 'Conference Presentation'),
            ('seminar', 'Research Seminar'),
            ('interview', 'Research Interview'),
            ('tutorial', 'Educational Tutorial')
        ],
        'supported_languages': [
            ('en', 'English'),
            ('es', 'Spanish'),
            ('fr', 'French'),
            ('de', 'German'),
            ('it', 'Italian'),
            ('pt', 'Portuguese'),
            ('ja', 'Japanese'),
            ('ko', 'Korean'),
            ('zh', 'Chinese')
        ]
    }
    
    return render(request, 'video_analysis/analyzer.html', context)


def is_valid_video_url(url):
    """Validate if the URL is a supported video platform."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Support common video platforms
        supported_domains = [
            'youtube.com', 'www.youtube.com', 'youtu.be',
            'vimeo.com', 'www.vimeo.com',
            'dailymotion.com', 'www.dailymotion.com',
            'twitch.tv', 'www.twitch.tv',
            'wistia.com', 'www.wistia.com'
        ]
        
        return any(domain == supported or domain.endswith('.' + supported) 
                  for supported in supported_domains)
    except:
        return False


def extract_video_title(url):
    """Extract a basic title from the video URL."""
    try:
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            return "YouTube Video"
        elif 'vimeo.com' in parsed.netloc:
            return "Vimeo Video"
        elif 'dailymotion.com' in parsed.netloc:
            return "Dailymotion Video"
        else:
            return f"Video from {parsed.netloc}"
    except:
        return "Video Analysis"


def process_video_analysis(analysis):
    """Process video analysis (mock implementation)."""
    try:
        # Mock analysis results
        mock_transcript = """
        Welcome to today's lecture on machine learning fundamentals. 
        In this session, we'll explore the key concepts that form the foundation 
        of modern artificial intelligence systems.
        
        First, let's discuss supervised learning, which involves training algorithms 
        on labeled datasets to make predictions on new, unseen data.
        
        Next, we'll examine unsupervised learning, where algorithms find patterns 
        in data without explicit labels or target variables.
        
        Finally, we'll cover reinforcement learning, where agents learn through 
        interaction with an environment and feedback in the form of rewards.
        """
        
        mock_key_concepts = [
            "Machine Learning",
            "Supervised Learning",
            "Unsupervised Learning", 
            "Reinforcement Learning",
            "Artificial Intelligence",
            "Pattern Recognition",
            "Data Science"
        ]
        
        mock_topics = [
            {"topic": "Introduction", "start_time": 0, "end_time": 60},
            {"topic": "Supervised Learning", "start_time": 60, "end_time": 180},
            {"topic": "Unsupervised Learning", "start_time": 180, "end_time": 300},
            {"topic": "Reinforcement Learning", "start_time": 300, "end_time": 420}
        ]
        
        mock_timeline = [
            {"timestamp": 0, "event": "Lecture begins"},
            {"timestamp": 30, "event": "Introduction to ML concepts"},
            {"timestamp": 90, "event": "Supervised learning explanation"},
            {"timestamp": 210, "event": "Unsupervised learning examples"},
            {"timestamp": 330, "event": "Reinforcement learning overview"},
            {"timestamp": 420, "event": "Conclusion and Q&A"}
        ]
        
        mock_sentiment = {
            "overall_sentiment": "positive",
            "confidence": 0.85,
            "emotional_arc": [
                {"timestamp": 0, "sentiment": "neutral", "confidence": 0.7},
                {"timestamp": 100, "sentiment": "positive", "confidence": 0.8},
                {"timestamp": 200, "sentiment": "positive", "confidence": 0.9},
                {"timestamp": 300, "sentiment": "positive", "confidence": 0.85}
            ]
        }
        
        # Update analysis with mock results
        analysis.transcript = mock_transcript
        analysis.summary = "This educational video provides a comprehensive introduction to machine learning fundamentals, covering supervised learning, unsupervised learning, and reinforcement learning concepts."
        analysis.key_concepts = json.dumps(mock_key_concepts)
        analysis.topics = json.dumps(mock_topics)
        analysis.timeline = json.dumps(mock_timeline)
        analysis.sentiment_analysis = json.dumps(mock_sentiment)
        analysis.duration_seconds = 420  # 7 minutes
        analysis.processing_status = 'completed'
        analysis.processing_time = 2.5  # 2.5 seconds processing time
        
        analysis.save()
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        analysis.processing_status = 'failed'
        analysis.error_message = str(e)
        analysis.save()
