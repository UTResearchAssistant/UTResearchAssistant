"""Enhanced Django services for the research assistant.

This module contains all the business logic services for the enhanced features
including literature search, podcast generation, video analysis, and more.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import requests
import openai
from django.conf import settings
from django.core.files.base import ContentFile
from django.contrib.auth.models import User

# Academic APIs
import arxiv
from scholarly import scholarly

logger = logging.getLogger(__name__)


class EnhancedLiteratureSearchService:
    """Service for enhanced literature search across multiple sources."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def unified_search(
        self,
        query: str,
        sources: List[str] = None,
        filters: Dict = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Perform unified search across multiple academic sources."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        results = {"papers": [], "total": 0, "sources": sources}

        for source in sources:
            try:
                if source == "arxiv":
                    source_results = await self._search_arxiv(
                        query, limit // len(sources)
                    )
                elif source == "semantic_scholar":
                    source_results = await self._search_semantic_scholar(
                        query, limit // len(sources)
                    )
                elif source == "crossref":
                    source_results = await self._search_crossref(
                        query, limit // len(sources)
                    )
                else:
                    continue

                results["papers"].extend(source_results)

            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
                continue

        # Remove duplicates and sort by relevance
        results["papers"] = self._deduplicate_papers(results["papers"])
        results["total"] = len(results["papers"])

        # Apply filters if provided
        if filters:
            results["papers"] = self._apply_filters(results["papers"], filters)

        return results

    async def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv for papers."""
        papers = []
        try:
            search = arxiv.Search(
                query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance
            )

            for result in search.results():
                paper = {
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [author.name for author in result.authors],
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "publication_date": result.published.date().isoformat(),
                    "pdf_url": result.pdf_url,
                    "external_url": result.entry_id,
                    "source": "arxiv",
                    "subject_categories": result.categories,
                }
                papers.append(paper)

        except Exception as e:
            logger.error(f"ArXiv search error: {e}")

        return papers

    async def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar for papers."""
        papers = []
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,citationCount,url,openAccessPdf",
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    paper = {
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "authors": [
                            author.get("name", "") for author in item.get("authors", [])
                        ],
                        "publication_date": (
                            f"{item.get('year', '')}-01-01"
                            if item.get("year")
                            else None
                        ),
                        "citation_count": item.get("citationCount", 0),
                        "external_url": item.get("url", ""),
                        "pdf_url": (
                            item.get("openAccessPdf", {}).get("url", "")
                            if item.get("openAccessPdf")
                            else ""
                        ),
                        "source": "semantic_scholar",
                        "is_open_access": bool(item.get("openAccessPdf")),
                    }
                    papers.append(paper)

        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")

        return papers

    async def _search_crossref(self, query: str, limit: int) -> List[Dict]:
        """Search CrossRef for papers."""
        papers = []
        try:
            url = "https://api.crossref.org/works"
            params = {
                "query": query,
                "rows": limit,
                "select": "title,abstract,author,published-print,DOI,URL,container-title",
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("message", {}).get("items", []):
                    paper = {
                        "title": " ".join(item.get("title", [])),
                        "abstract": item.get("abstract", ""),
                        "authors": [
                            f"{author.get('given', '')} {author.get('family', '')}"
                            for author in item.get("author", [])
                        ],
                        "doi": item.get("DOI", ""),
                        "publication_date": self._parse_crossref_date(
                            item.get("published-print")
                        ),
                        "journal": " ".join(item.get("container-title", [])),
                        "external_url": item.get("URL", ""),
                        "source": "crossref",
                    }
                    papers.append(paper)

        except Exception as e:
            logger.error(f"CrossRef search error: {e}")

        return papers

    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            title_key = paper.get("title", "").lower().strip()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)

        return unique_papers

    def _apply_filters(self, papers: List[Dict], filters: Dict) -> List[Dict]:
        """Apply search filters to papers."""
        filtered_papers = papers

        if filters.get("start_date"):
            start_date = datetime.fromisoformat(filters["start_date"]).date()
            filtered_papers = [
                p
                for p in filtered_papers
                if p.get("publication_date")
                and datetime.fromisoformat(p["publication_date"]).date() >= start_date
            ]

        if filters.get("end_date"):
            end_date = datetime.fromisoformat(filters["end_date"]).date()
            filtered_papers = [
                p
                for p in filtered_papers
                if p.get("publication_date")
                and datetime.fromisoformat(p["publication_date"]).date() <= end_date
            ]

        if filters.get("min_citations"):
            min_citations = int(filters["min_citations"])
            filtered_papers = [
                p
                for p in filtered_papers
                if p.get("citation_count", 0) >= min_citations
            ]

        if filters.get("open_access_only"):
            filtered_papers = [
                p for p in filtered_papers if p.get("is_open_access", False)
            ]

        return filtered_papers

    def _parse_crossref_date(self, date_obj: Dict) -> Optional[str]:
        """Parse CrossRef date object to ISO string."""
        if not date_obj or "date-parts" not in date_obj:
            return None

        date_parts = date_obj["date-parts"][0]
        if len(date_parts) >= 3:
            return f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
        elif len(date_parts) >= 2:
            return f"{date_parts[0]}-{date_parts[1]:02d}-01"
        elif len(date_parts) >= 1:
            return f"{date_parts[0]}-01-01"

        return None


class PodcastGenerationService:
    """Service for generating podcasts from research papers."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_paper_summary_podcast(
        self,
        paper_content: str,
        style: str = "conversational",
        duration: int = 300,
        voice: str = "alloy",
    ) -> Dict[str, Any]:
        """Generate a summary-style podcast from a research paper."""

        # Generate script
        script = await self._generate_summary_script(paper_content, duration, style)

        # Generate audio
        audio_data = await self._text_to_speech(script, voice)

        # Calculate metadata
        metadata = {
            "style": style,
            "duration_seconds": duration,
            "voice_model": voice,
            "word_count": len(script.split()),
            "script": script,
            "file_size": len(audio_data) if audio_data else 0,
        }

        return {
            "audio_data": audio_data,
            "metadata": metadata,
            "success": audio_data is not None,
        }

    async def generate_interview_style_podcast(
        self, paper_content: str, duration: int = 600, voices: List[str] = None
    ) -> Dict[str, Any]:
        """Generate an interview-style podcast with multiple voices."""

        if voices is None:
            voices = ["alloy", "echo"]

        # Generate interview script
        script = await self._generate_interview_script(paper_content, duration)

        # Split script into segments by speaker
        segments = self._parse_interview_script(script)

        # Generate audio for each segment
        audio_segments = []
        for segment in segments:
            voice = voices[segment["speaker_index"] % len(voices)]
            audio_data = await self._text_to_speech(segment["text"], voice)
            if audio_data:
                audio_segments.append(audio_data)

        # Combine audio segments (simplified - in production, use audio processing)
        combined_audio = b"".join(audio_segments) if audio_segments else None

        metadata = {
            "style": "interview",
            "duration_seconds": duration,
            "voices": voices,
            "segments": len(segments),
            "script": script,
            "file_size": len(combined_audio) if combined_audio else 0,
        }

        return {
            "audio_data": combined_audio,
            "metadata": metadata,
            "success": combined_audio is not None,
        }

    async def _generate_summary_script(
        self, paper_content: str, duration: int, style: str
    ) -> str:
        """Generate a script for summary-style podcast."""

        # Estimate words per minute based on style
        wpm = 150 if style == "conversational" else 130
        target_words = (duration // 60) * wpm

        prompt = f"""
        Create a {style} podcast script from this research paper. 
        Target length: approximately {target_words} words ({duration // 60} minutes).
        
        Style guidelines:
        - {style} tone throughout
        - Clear explanations for general audience
        - Highlight key findings and implications
        - Include smooth transitions
        - End with actionable insights
        
        Paper content:
        {paper_content[:4000]}
        
        Generate only the script text, no stage directions.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=target_words + 200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Script generation error: {e}")
            return "Script generation failed. Please try again."

    async def _generate_interview_script(
        self, paper_content: str, duration: int
    ) -> str:
        """Generate an interview-style script."""

        target_words = (duration // 60) * 140  # Slightly slower for dialogue

        prompt = f"""
        Create an interview-style podcast script about this research paper.
        Format: Host interviewing the lead researcher.
        Target length: approximately {target_words} words ({duration // 60} minutes).
        
        Structure:
        - Introduction by host
        - Q&A format with natural dialogue
        - Host asks clarifying questions
        - Researcher explains concepts clearly
        - Conclusion with key takeaways
        
        Use format:
        Host: [dialogue]
        Researcher: [dialogue]
        
        Paper content:
        {paper_content[:4000]}
        
        Generate only the dialogue, no stage directions.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=target_words + 200,
                temperature=0.8,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Interview script generation error: {e}")
            return "Host: Welcome to our research podcast.\nResearcher: Thank you for having me."

    def _parse_interview_script(self, script: str) -> List[Dict]:
        """Parse interview script into segments by speaker."""
        segments = []
        lines = script.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("Host:"):
                segments.append(
                    {"speaker": "Host", "speaker_index": 0, "text": line[5:].strip()}
                )
            elif line.startswith("Researcher:"):
                segments.append(
                    {
                        "speaker": "Researcher",
                        "speaker_index": 1,
                        "text": line[11:].strip(),
                    }
                )

        return segments

    async def _text_to_speech(self, text: str, voice: str) -> Optional[bytes]:
        """Convert text to speech using OpenAI's TTS."""
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1", voice=voice, input=text, response_format="mp3"
            )
            return response.content
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return None


class VideoAnalysisService:
    """Service for analyzing research videos."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def analyze_research_video(
        self, video_url: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze a research video comprehensively."""

        results = {
            "video_url": video_url,
            "analysis_type": analysis_type,
            "transcript": "",
            "summary": "",
            "key_concepts": [],
            "timeline": [],
            "topics": [],
            "sentiment_analysis": {},
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: Extract audio and transcribe
            transcript = await self._transcribe_video(video_url)
            results["transcript"] = transcript

            if not transcript:
                results["error"] = "Failed to transcribe video"
                return results

            # Step 2: Generate summary
            results["summary"] = await self._generate_video_summary(transcript)

            # Step 3: Extract key concepts
            results["key_concepts"] = await self._extract_key_concepts(transcript)

            # Step 4: Create timeline
            results["timeline"] = await self._create_timeline(transcript)

            # Step 5: Topic modeling
            results["topics"] = await self._extract_topics(transcript)

            # Step 6: Sentiment analysis
            if analysis_type == "comprehensive":
                results["sentiment_analysis"] = await self._analyze_sentiment(
                    transcript
                )

        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            results["error"] = str(e)

        results["processing_time"] = (datetime.now() - start_time).total_seconds()
        return results

    async def _transcribe_video(self, video_url: str) -> str:
        """Transcribe video audio using Whisper."""
        try:
            # In a real implementation, you would:
            # 1. Download the video
            # 2. Extract audio
            # 3. Use OpenAI Whisper to transcribe

            # For now, return a placeholder
            # This would be replaced with actual Whisper API call
            return "Video transcription would be implemented here using OpenAI Whisper API."

        except Exception as e:
            logger.error(f"Video transcription error: {e}")
            return ""

    async def _generate_video_summary(self, transcript: str) -> str:
        """Generate a summary of the video content."""

        prompt = f"""
        Analyze this video transcript and provide a comprehensive summary.
        Focus on:
        - Main research topics discussed
        - Key findings and conclusions
        - Methodologies mentioned
        - Important insights and implications
        
        Transcript:
        {transcript[:3000]}
        
        Provide a clear, structured summary.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Summary generation failed."

    async def _extract_key_concepts(self, transcript: str) -> List[Dict]:
        """Extract key concepts from the transcript."""

        prompt = f"""
        Extract the top 10 key concepts from this research video transcript.
        For each concept, provide:
        - concept: the main term or idea
        - definition: brief explanation
        - importance: why it's significant
        - timestamp: approximate time mentioned (if determinable)
        
        Return as JSON array.
        
        Transcript:
        {transcript[:3000]}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )

            # Parse JSON response
            concepts_text = response.choices[0].message.content.strip()
            try:
                return json.loads(concepts_text)
            except json.JSONDecodeError:
                # Fallback to basic parsing
                return [
                    {
                        "concept": "Concept extraction",
                        "definition": "Failed to parse",
                        "importance": "Low",
                    }
                ]

        except Exception as e:
            logger.error(f"Concept extraction error: {e}")
            return []

    async def _create_timeline(self, transcript: str) -> List[Dict]:
        """Create a timeline of topics discussed in the video."""

        # This would analyze the transcript to create timestamps
        # For now, return a basic structure
        return [
            {
                "time": "00:00",
                "topic": "Introduction",
                "description": "Overview of research",
            },
            {
                "time": "05:00",
                "topic": "Methodology",
                "description": "Research methods discussed",
            },
            {
                "time": "15:00",
                "topic": "Results",
                "description": "Key findings presented",
            },
            {
                "time": "25:00",
                "topic": "Discussion",
                "description": "Implications and future work",
            },
            {
                "time": "30:00",
                "topic": "Conclusion",
                "description": "Summary and takeaways",
            },
        ]

    async def _extract_topics(self, transcript: str) -> List[Dict]:
        """Extract main topics from the transcript."""

        prompt = f"""
        Identify the main research topics discussed in this transcript.
        For each topic, provide:
        - topic: main subject area
        - relevance: how central it is (1-10)
        - keywords: related terms
        
        Return as JSON array, maximum 5 topics.
        
        Transcript:
        {transcript[:2000]}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )

            topics_text = response.choices[0].message.content.strip()
            try:
                return json.loads(topics_text)
            except json.JSONDecodeError:
                return [
                    {
                        "topic": "Research Discussion",
                        "relevance": 8,
                        "keywords": ["research", "analysis"],
                    }
                ]

        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return []

    async def _analyze_sentiment(self, transcript: str) -> Dict:
        """Analyze sentiment throughout the video."""

        # Basic sentiment analysis
        return {
            "overall_sentiment": "positive",
            "confidence": 0.75,
            "emotional_tone": "academic",
            "engagement_level": "high",
        }


class WritingAssistanceService:
    """Service for academic writing assistance."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def assist_writing(
        self,
        content: str,
        task_type: str,
        tone: str = "academic",
        target_audience: str = "researchers",
    ) -> Dict[str, Any]:
        """Provide writing assistance for academic content."""

        result = {
            "original_content": content,
            "improved_content": "",
            "suggestions": [],
            "style_improvements": [],
            "citation_corrections": [],
            "readability_score": 0.0,
            "word_count": len(content.split()),
        }

        try:
            # Generate improved content
            result["improved_content"] = await self._improve_content(
                content, task_type, tone, target_audience
            )

            # Generate suggestions
            result["suggestions"] = await self._generate_suggestions(content, task_type)

            # Style improvements
            result["style_improvements"] = await self._suggest_style_improvements(
                content, tone
            )

            # Basic readability score (simplified)
            result["readability_score"] = self._calculate_readability_score(content)

        except Exception as e:
            logger.error(f"Writing assistance error: {e}")
            result["error"] = str(e)

        return result

    async def _improve_content(
        self, content: str, task_type: str, tone: str, audience: str
    ) -> str:
        """Improve the academic content."""

        prompt = f"""
        Improve this {task_type} content for {audience}.
        Tone: {tone}
        
        Requirements:
        - Maintain academic rigor
        - Improve clarity and flow
        - Enhance argument structure
        - Fix grammar and style issues
        - Keep the same general length
        
        Original content:
        {content}
        
        Provide the improved version:
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(content.split()) + 200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Content improvement error: {e}")
            return content

    async def _generate_suggestions(self, content: str, task_type: str) -> List[str]:
        """Generate writing suggestions."""

        prompt = f"""
        Analyze this {task_type} and provide 5 specific suggestions for improvement.
        Focus on:
        - Structure and organization
        - Argument clarity
        - Evidence and support
        - Academic tone
        - Transitions
        
        Content:
        {content[:2000]}
        
        Return as a list of actionable suggestions.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4,
            )

            suggestions_text = response.choices[0].message.content.strip()
            # Parse suggestions (simple split by lines/numbers)
            suggestions = [
                s.strip()
                for s in suggestions_text.split("\n")
                if s.strip() and len(s.strip()) > 10
            ]
            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            logger.error(f"Suggestions generation error: {e}")
            return [
                "Review content structure",
                "Check argument flow",
                "Verify citations",
            ]

    async def _suggest_style_improvements(self, content: str, tone: str) -> List[str]:
        """Suggest style improvements."""

        improvements = []

        # Basic style checks
        if len([s for s in content.split(".") if len(s.split()) > 30]) > 0:
            improvements.append(
                "Consider breaking down long sentences for better readability"
            )

        if content.count("very") + content.count("really") + content.count("quite") > 3:
            improvements.append(
                "Reduce use of qualifiers like 'very', 'really', 'quite' for stronger academic tone"
            )

        if (
            tone == "academic"
            and content.count("I think") + content.count("I believe") > 2
        ):
            improvements.append(
                "Replace personal opinions with evidence-based statements"
            )

        return improvements

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a basic readability score."""

        words = content.split()
        sentences = content.split(".")

        if len(sentences) == 0:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Simple readability approximation
        if avg_words_per_sentence < 15:
            return 8.5  # Easy to read
        elif avg_words_per_sentence < 25:
            return 7.0  # Moderate
        else:
            return 5.5  # Complex
