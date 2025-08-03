"""Podcast Generation Service with AI-powered text-to-speech.

This service generates podcasts from research papers using OpenAI's TTS API
and advanced AI for script generation.
"""

import asyncio
import json
import logging
import tempfile
import os
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

import openai
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from core.models import Paper, PodcastEpisode

logger = logging.getLogger(__name__)


class PodcastGenerationService:
    """AI-powered podcast generation service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.voice_models = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        self.podcast_styles = {
            'summary': self._generate_summary_script,
            'interview': self._generate_interview_script,
            'debate': self._generate_debate_script,
            'educational': self._generate_educational_script,
        }
    
    async def generate_podcast(
        self,
        paper: Paper,
        style: str = 'summary',
        voice: str = 'alloy',
        language: str = 'en',
        user=None
    ) -> Dict:
        """
        Generate a podcast from a research paper.
        
        Args:
            paper: Paper object to generate podcast from
            style: Podcast style (summary, interview, debate, educational)
            voice: Voice model to use
            language: Language for the podcast
            user: User creating the podcast
            
        Returns:
            Dictionary containing podcast generation results
        """
        generation_start = datetime.now()
        
        try:
            # Generate script based on style
            logger.info(f"Generating {style} script for paper: {paper.title}")
            script = await self._generate_script(paper, style, language)
            
            # Generate audio from script
            logger.info(f"Generating audio with voice: {voice}")
            audio_data, duration = await self._generate_audio(script, voice)
            
            # Save audio file
            audio_filename = f"podcast_{paper.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            audio_path = default_storage.save(
                f"podcasts/{audio_filename}",
                ContentFile(audio_data)
            )
            
            # Create podcast episode
            generation_time = (datetime.now() - generation_start).total_seconds()
            
            episode = PodcastEpisode.objects.create(
                title=f"Podcast: {paper.title[:150]}",
                description=f"AI-generated {style} podcast of the research paper",
                paper=paper,
                creator=user,
                style=style,
                duration_seconds=duration,
                audio_file=audio_path,
                script=script,
                voice_model=voice,
                language=language,
                generation_time=generation_time,
                file_size=len(audio_data)
            )
            
            logger.info(f"Successfully generated podcast episode: {episode.id}")
            
            return {
                'success': True,
                'episode': episode,
                'duration': duration,
                'file_size': len(audio_data),
                'generation_time': generation_time,
                'audio_url': default_storage.url(audio_path)
            }
            
        except Exception as e:
            logger.error(f"Error generating podcast: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': (datetime.now() - generation_start).total_seconds()
            }
    
    async def _generate_script(self, paper: Paper, style: str, language: str) -> str:
        """Generate podcast script based on paper and style."""
        if style not in self.podcast_styles:
            raise ValueError(f"Unsupported podcast style: {style}")
        
        return await self.podcast_styles[style](paper, language)
    
    async def _generate_summary_script(self, paper: Paper, language: str) -> str:
        """Generate a summary-style podcast script."""
        prompt = f"""
        Create an engaging podcast script that summarizes this research paper.
        The script should be informative, accessible, and approximately 3-5 minutes when spoken.
        
        Paper Title: {paper.title}
        Abstract: {paper.abstract}
        Authors: {', '.join(json.loads(paper.authors) if paper.authors else [])}
        
        Guidelines:
        - Start with a catchy introduction
        - Explain the research problem and motivation
        - Describe the methodology briefly
        - Highlight key findings and implications
        - End with significance and future directions
        - Use conversational tone suitable for audio
        - Include natural pauses and transitions
        - Language: {language}
        
        Format as a natural speech script with speaker cues.
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert science communicator creating engaging podcast scripts that make research accessible to a broad audience."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary script: {e}")
            raise
    
    async def _generate_interview_script(self, paper: Paper, language: str) -> str:
        """Generate an interview-style podcast script."""
        prompt = f"""
        Create an engaging interview-style podcast script about this research paper.
        Format it as a conversation between a host and the lead researcher.
        The script should be approximately 5-7 minutes when spoken.
        
        Paper Title: {paper.title}
        Abstract: {paper.abstract}
        Authors: {', '.join(json.loads(paper.authors) if paper.authors else [])}
        
        Guidelines:
        - Host asks insightful questions
        - Researcher provides detailed but accessible answers
        - Include natural conversation flow with interruptions and clarifications
        - Cover motivation, methodology, findings, and implications
        - Use conversational tone
        - Language: {language}
        
        Format:
        HOST: [question or comment]
        RESEARCHER: [response]
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating engaging interview-style podcast scripts that explore research in depth through natural conversation."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating interview script: {e}")
            raise
    
    async def _generate_debate_script(self, paper: Paper, language: str) -> str:
        """Generate a debate-style podcast script."""
        prompt = f"""
        Create an engaging debate-style podcast script about this research paper.
        Format it as a discussion between two experts with different perspectives.
        The script should be approximately 6-8 minutes when spoken.
        
        Paper Title: {paper.title}
        Abstract: {paper.abstract}
        
        Guidelines:
        - Expert A supports/advocates for the research
        - Expert B provides constructive criticism and alternative viewpoints
        - Include back-and-forth discussion with evidence-based arguments
        - Cover strengths, limitations, methodology concerns, and implications
        - Maintain respectful but engaging debate tone
        - Language: {language}
        
        Format:
        EXPERT A: [supportive argument]
        EXPERT B: [critical or alternative perspective]
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating balanced debate-style podcast scripts that explore research from multiple academic perspectives."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating debate script: {e}")
            raise
    
    async def _generate_educational_script(self, paper: Paper, language: str) -> str:
        """Generate an educational-style podcast script."""
        prompt = f"""
        Create an educational podcast script that teaches the key concepts from this research paper.
        The script should be structured like a mini-lecture for students.
        Approximately 4-6 minutes when spoken.
        
        Paper Title: {paper.title}
        Abstract: {paper.abstract}
        
        Guidelines:
        - Start with learning objectives
        - Explain background concepts and terminology
        - Walk through the research step-by-step
        - Include examples and analogies to aid understanding
        - Summarize key takeaways
        - End with discussion questions
        - Use clear, pedagogical tone
        - Language: {language}
        
        Format as a structured educational presentation.
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educator creating clear, structured podcast scripts that teach complex research concepts effectively."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2200,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating educational script: {e}")
            raise
    
    async def _generate_audio(self, script: str, voice: str) -> tuple[bytes, int]:
        """Generate audio from script using OpenAI TTS."""
        try:
            # Clean script for TTS (remove speaker labels for single voice)
            cleaned_script = self._clean_script_for_tts(script)
            
            # Generate audio using OpenAI TTS
            response = await asyncio.to_thread(
                self.openai_client.audio.speech.create,
                model="tts-1-hd",
                voice=voice,
                input=cleaned_script,
                response_format="mp3"
            )
            
            audio_data = response.content
            
            # Estimate duration (rough calculation: ~150 words per minute)
            word_count = len(cleaned_script.split())
            estimated_duration = int((word_count / 150) * 60)
            
            return audio_data, estimated_duration
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
    
    def _clean_script_for_tts(self, script: str) -> str:
        """Clean script for text-to-speech conversion."""
        # Remove speaker labels and format for single voice
        lines = script.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove speaker labels (HOST:, RESEARCHER:, etc.)
            if ':' in line and len(line.split(':')[0].split()) <= 2:
                # Likely a speaker label
                content = ':'.join(line.split(':')[1:]).strip()
                if content:
                    cleaned_lines.append(content)
            else:
                cleaned_lines.append(line)
        
        # Join with natural pauses
        cleaned_script = '. '.join(cleaned_lines)
        
        # Clean up formatting
        cleaned_script = cleaned_script.replace('..', '.')
        cleaned_script = cleaned_script.replace('  ', ' ')
        
        return cleaned_script
    
    async def generate_transcript(self, audio_file_path: str) -> str:
        """Generate transcript from audio file using OpenAI Whisper."""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = await asyncio.to_thread(
                    self.openai_client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating transcript: {e}")
            raise
    
    async def get_podcast_analytics(self, episode: PodcastEpisode) -> Dict:
        """Get analytics for a podcast episode."""
        return {
            'episode_id': episode.id,
            'title': episode.title,
            'play_count': episode.play_count,
            'like_count': episode.like_count,
            'share_count': episode.share_count,
            'duration': episode.duration_seconds,
            'file_size': episode.file_size,
            'generation_time': episode.generation_time,
            'created_at': episode.created_at,
            'engagement_rate': (episode.like_count + episode.share_count) / max(episode.play_count, 1)
        }


# Create singleton instance
podcast_generation_service = PodcastGenerationService()
