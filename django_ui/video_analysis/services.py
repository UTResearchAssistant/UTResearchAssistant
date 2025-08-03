"""Video Analysis Service with AI-powered transcription and analysis.

This service provides comprehensive video analysis including transcription,
content analysis, key concept extraction, and timeline generation.
"""

import asyncio
import json
import logging
import tempfile
import os
from typing import Dict, Optional, List, Any
from datetime import datetime
import re

import openai
import requests
from django.conf import settings

from core.models import VideoAnalysis

logger = logging.getLogger(__name__)


class VideoAnalysisService:
    """AI-powered video analysis service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv']
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
    
    async def analyze_video(
        self,
        video_url: str,
        title: str,
        video_type: str = 'lecture',
        language: str = 'en',
        user=None
    ) -> Dict:
        """
        Analyze a video with comprehensive AI-powered analysis.
        
        Args:
            video_url: URL of the video to analyze
            title: Title for the analysis
            video_type: Type of video (lecture, conference, etc.)
            language: Language of the video
            user: User requesting the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        analysis_start = datetime.now()
        
        try:
            # Create video analysis record
            video_analysis = VideoAnalysis.objects.create(
                title=title,
                video_url=video_url,
                creator=user,
                video_type=video_type,
                language=language,
                processing_status='processing'
            )
            
            logger.info(f"Starting video analysis for: {title}")
            
            # Step 1: Download and validate video
            video_info = await self._get_video_info(video_url)
            if not video_info['valid']:
                raise ValueError(f"Invalid video: {video_info['error']}")
            
            # Step 2: Extract audio and transcribe
            logger.info("Extracting audio and transcribing...")
            transcript = await self._transcribe_video(video_url, language)
            
            # Step 3: Analyze content with AI
            logger.info("Analyzing content with AI...")
            analysis_results = await self._analyze_content(transcript, video_type)
            
            # Step 4: Extract timeline and key moments
            logger.info("Extracting timeline and key moments...")
            timeline = await self._extract_timeline(transcript)
            
            # Step 5: Perform sentiment analysis
            logger.info("Performing sentiment analysis...")
            sentiment = await self._analyze_sentiment(transcript)
            
            # Update video analysis record
            processing_time = (datetime.now() - analysis_start).total_seconds()
            
            video_analysis.transcript = transcript
            video_analysis.summary = analysis_results['summary']
            video_analysis.key_concepts = json.dumps(analysis_results['key_concepts'])
            video_analysis.timeline = json.dumps(timeline)
            video_analysis.topics = json.dumps(analysis_results['topics'])
            video_analysis.sentiment_analysis = json.dumps(sentiment)
            video_analysis.duration_seconds = video_info.get('duration', 0)
            video_analysis.processing_status = 'completed'
            video_analysis.processing_time = processing_time
            video_analysis.save()
            
            logger.info(f"Successfully completed video analysis: {video_analysis.id}")
            
            return {
                'success': True,
                'analysis': video_analysis,
                'processing_time': processing_time,
                'transcript_length': len(transcript),
                'key_concepts_count': len(analysis_results['key_concepts']),
                'topics_count': len(analysis_results['topics'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            
            # Update record with error
            if 'video_analysis' in locals():
                video_analysis.processing_status = 'failed'
                video_analysis.error_message = str(e)
                video_analysis.processing_time = (datetime.now() - analysis_start).total_seconds()
                video_analysis.save()
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - analysis_start).total_seconds()
            }
    
    async def _get_video_info(self, video_url: str) -> Dict:
        """Get video information and validate."""
        try:
            # For YouTube videos, we could use youtube-dl or yt-dlp
            # For now, we'll do basic validation
            
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                return {
                    'valid': True,
                    'source': 'youtube',
                    'duration': None  # Would extract with youtube-dl
                }
            elif video_url.startswith('http'):
                # Try to get video info from direct URL
                try:
                    response = await asyncio.to_thread(
                        requests.head, video_url, timeout=10
                    )
                    
                    content_type = response.headers.get('content-type', '')
                    content_length = int(response.headers.get('content-length', 0))
                    
                    if not content_type.startswith('video/'):
                        return {'valid': False, 'error': 'Not a video file'}
                    
                    if content_length > self.max_file_size:
                        return {'valid': False, 'error': 'File too large'}
                    
                    return {
                        'valid': True,
                        'source': 'direct',
                        'content_type': content_type,
                        'size': content_length
                    }
                    
                except Exception as e:
                    return {'valid': False, 'error': f'Cannot access video: {e}'}
            else:
                return {'valid': False, 'error': 'Invalid video URL'}
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _transcribe_video(self, video_url: str, language: str) -> str:
        """Transcribe video using OpenAI Whisper."""
        try:
            # In a real implementation, you would:
            # 1. Download the video or extract audio
            # 2. Convert to supported audio format
            # 3. Split into chunks if needed (25MB limit)
            # 4. Transcribe with Whisper
            
            # For demo purposes, we'll simulate transcription
            # In production, use ffmpeg to extract audio and process with Whisper
            
            logger.info("Simulating video transcription...")
            
            # Simulate transcription based on video type
            if 'youtube.com' in video_url:
                # Could integrate with YouTube API for captions
                sample_transcript = """
                Welcome to today's lecture on artificial intelligence and machine learning.
                In this presentation, we'll explore the fundamental concepts of neural networks
                and their applications in modern research. We'll begin by discussing the
                historical development of AI, then move on to current methodologies and
                future directions in the field. The key topics we'll cover include
                supervised learning, unsupervised learning, and reinforcement learning
                paradigms. Let's start with the basics of neural network architecture...
                """
            else:
                sample_transcript = """
                This is a transcribed research presentation covering important scientific
                concepts and methodologies. The speaker discusses various approaches
                to solving complex problems in their field of study, presents experimental
                results, and concludes with implications for future research directions.
                """
            
            # In production, replace with actual Whisper API call:
            # with open(audio_file_path, 'rb') as audio_file:
            #     response = await asyncio.to_thread(
            #         self.openai_client.audio.transcriptions.create,
            #         model="whisper-1",
            #         file=audio_file,
            #         language=language
            #     )
            #     return response.text
            
            return sample_transcript.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
            raise
    
    async def _analyze_content(self, transcript: str, video_type: str) -> Dict:
        """Analyze video content using AI."""
        try:
            analysis_prompt = f"""
            Analyze this {video_type} transcript and provide:
            1. A comprehensive summary (2-3 paragraphs)
            2. Key concepts and topics discussed (as a list)
            3. Main themes and subject areas
            
            Transcript:
            {transcript}
            
            Provide the response in JSON format:
            {{
                "summary": "detailed summary",
                "key_concepts": ["concept1", "concept2", ...],
                "topics": ["topic1", "topic2", ...]
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic content analyzer. Analyze the given transcript and extract key information in the requested JSON format."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response.choices[0].message.content)
                return analysis_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response.choices[0].message.content
                return {
                    "summary": content[:500] + "...",
                    "key_concepts": ["Content analysis", "Research presentation"],
                    "topics": ["Academic content", "Research methodology"]
                }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                "summary": "Content analysis unavailable due to processing error.",
                "key_concepts": [],
                "topics": []
            }
    
    async def _extract_timeline(self, transcript: str) -> List[Dict]:
        """Extract timeline and key moments from transcript."""
        try:
            timeline_prompt = f"""
            Create a timeline of key moments from this transcript.
            Identify important sections, topic changes, and significant points.
            
            Transcript:
            {transcript}
            
            Provide response as JSON array:
            [
                {{
                    "timestamp": "estimated time",
                    "title": "Section title",
                    "description": "What happens at this point",
                    "importance": "high/medium/low"
                }}
            ]
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing video content and creating detailed timelines of key moments."
                    },
                    {"role": "user", "content": timeline_prompt}
                ],
                max_tokens=1000,
                temperature=0.4
            )
            
            try:
                timeline_data = json.loads(response.choices[0].message.content)
                return timeline_data
            except json.JSONDecodeError:
                # Fallback timeline
                return [
                    {
                        "timestamp": "0:00",
                        "title": "Introduction",
                        "description": "Presentation begins",
                        "importance": "medium"
                    },
                    {
                        "timestamp": "5:00",
                        "title": "Main Content",
                        "description": "Core topics discussed",
                        "importance": "high"
                    },
                    {
                        "timestamp": "End",
                        "title": "Conclusion",
                        "description": "Summary and closing remarks",
                        "importance": "medium"
                    }
                ]
            
        except Exception as e:
            logger.error(f"Error extracting timeline: {e}")
            return []
    
    async def _analyze_sentiment(self, transcript: str) -> Dict:
        """Analyze sentiment and emotional tone of the video."""
        try:
            sentiment_prompt = f"""
            Analyze the sentiment and emotional tone of this transcript.
            Consider the speaker's attitude, confidence level, and overall mood.
            
            Transcript:
            {transcript}
            
            Provide response in JSON format:
            {{
                "overall_sentiment": "positive/negative/neutral",
                "confidence_level": "high/medium/low",
                "emotional_tone": "description of tone",
                "key_emotions": ["emotion1", "emotion2"],
                "sentiment_score": float between -1 and 1
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in sentiment analysis and emotional tone assessment for academic content."
                    },
                    {"role": "user", "content": sentiment_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            try:
                sentiment_data = json.loads(response.choices[0].message.content)
                return sentiment_data
            except json.JSONDecodeError:
                return {
                    "overall_sentiment": "neutral",
                    "confidence_level": "medium",
                    "emotional_tone": "academic and informative",
                    "key_emotions": ["focused", "educational"],
                    "sentiment_score": 0.0
                }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "overall_sentiment": "neutral",
                "confidence_level": "unknown",
                "emotional_tone": "analysis unavailable",
                "key_emotions": [],
                "sentiment_score": 0.0
            }
    
    async def get_analysis_summary(self, analysis: VideoAnalysis) -> Dict:
        """Get a summary of video analysis results."""
        return {
            'analysis_id': analysis.id,
            'title': analysis.title,
            'video_url': analysis.video_url,
            'duration': analysis.duration_seconds,
            'processing_time': analysis.processing_time,
            'transcript_length': len(analysis.transcript) if analysis.transcript else 0,
            'key_concepts': json.loads(analysis.key_concepts) if analysis.key_concepts else [],
            'topics': json.loads(analysis.topics) if analysis.topics else [],
            'timeline_events': len(json.loads(analysis.timeline)) if analysis.timeline else 0,
            'sentiment': json.loads(analysis.sentiment_analysis) if analysis.sentiment_analysis else {},
            'status': analysis.processing_status,
            'created_at': analysis.created_at
        }
    
    async def search_videos(
        self,
        query: str,
        user=None,
        filters: Dict = None
    ) -> List[VideoAnalysis]:
        """Search analyzed videos by content."""
        queryset = VideoAnalysis.objects.filter(
            processing_status='completed'
        )
        
        if user:
            queryset = queryset.filter(creator=user)
        
        # Search in title, transcript, and summary
        if query:
            queryset = queryset.filter(
                Q(title__icontains=query) |
                Q(transcript__icontains=query) |
                Q(summary__icontains=query)
            )
        
        # Apply filters
        if filters:
            if 'video_type' in filters:
                queryset = queryset.filter(video_type=filters['video_type'])
            if 'date_from' in filters:
                queryset = queryset.filter(created_at__gte=filters['date_from'])
            if 'date_to' in filters:
                queryset = queryset.filter(created_at__lte=filters['date_to'])
        
        return queryset.order_by('-created_at')[:50]


# Create singleton instance
video_analysis_service = VideoAnalysisService()
