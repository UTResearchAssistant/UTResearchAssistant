"""Video analysis service for research content."""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import re
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class VideoAnalysisService:
    """Service for analyzing research videos and extracting insights."""

    def __init__(self, config=None):
        self.config = config
        self.storage_path = (
            Path(config.storage_path if config else "./storage") / "videos"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        self.openai_client = None
        if config and config.openai_api_key:
            try:
                import openai

                self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
            except ImportError:
                logger.warning("OpenAI package not available")

        self.whisper_available = False
        try:
            import whisper

            self.whisper_model = whisper.load_model("base")
            self.whisper_available = True
        except ImportError:
            logger.warning("Whisper not available for local transcription")

    async def analyze_research_video(
        self, video_url: str, analysis_type: str = "comprehensive"
    ) -> Dict:
        """Analyze a research video and extract insights."""

        try:
            # Download video metadata
            video_info = await self._get_video_metadata(video_url)

            # Extract audio and transcribe
            transcript = await self._transcribe_video(video_url)

            # Analyze transcript
            analysis = await self._analyze_transcript(transcript, analysis_type)

            # Extract research-specific insights
            research_insights = await self._extract_research_insights(
                transcript, video_info
            )

            # Find mentioned papers
            mentioned_papers = await self._find_mentioned_papers(transcript)

            result = {
                "video_url": video_url,
                "title": video_info.get("title"),
                "description": video_info.get("description"),
                "duration": video_info.get("duration"),
                "channel": video_info.get("channel"),
                "upload_date": video_info.get("upload_date"),
                "transcript": transcript,
                "summary": analysis.get("summary"),
                "key_points": analysis.get("key_points", []),
                "topics": analysis.get("topics", []),
                "research_insights": research_insights,
                "mentioned_papers": mentioned_papers,
                "confidence_score": analysis.get("confidence_score", 0.8),
                "language": analysis.get("language", "en"),
                "analysis_type": analysis_type,
                "analyzed_at": datetime.utcnow(),
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing video {video_url}: {e}")
            raise

    async def analyze_lecture_video(self, video_url: str) -> Dict:
        """Specialized analysis for academic lecture videos."""

        analysis = await self.analyze_research_video(video_url, "lecture")

        # Extract lecture-specific elements
        lecture_structure = await self._extract_lecture_structure(
            analysis["transcript"]
        )
        slides_content = await self._extract_slides_references(analysis["transcript"])
        learning_objectives = await self._identify_learning_objectives(
            analysis["transcript"]
        )

        analysis.update(
            {
                "lecture_structure": lecture_structure,
                "slides_content": slides_content,
                "learning_objectives": learning_objectives,
                "video_type": "lecture",
            }
        )

        return analysis

    async def analyze_conference_presentation(self, video_url: str) -> Dict:
        """Specialized analysis for conference presentation videos."""

        analysis = await self.analyze_research_video(video_url, "presentation")

        # Extract presentation-specific elements
        methodology = await self._extract_methodology(analysis["transcript"])
        results_section = await self._extract_results(analysis["transcript"])
        future_work = await self._extract_future_work(analysis["transcript"])
        questions_answers = await self._extract_qa_section(analysis["transcript"])

        analysis.update(
            {
                "methodology": methodology,
                "results": results_section,
                "future_work": future_work,
                "qa_section": questions_answers,
                "video_type": "conference_presentation",
            }
        )

        return analysis

    async def generate_video_summary_podcast(self, video_analysis: Dict) -> Dict:
        """Generate a podcast episode summarizing a research video."""

        # Import podcast service
        from .podcast_service import PodcastGenerationService

        podcast_service = PodcastGenerationService(self.config)

        # Create pseudo-paper from video analysis
        video_paper = {
            "title": video_analysis.get("title", "Research Video"),
            "abstract": video_analysis.get("summary", ""),
            "authors": [video_analysis.get("channel", "Unknown")],
            "source": "video",
            "url": video_analysis.get("video_url"),
            "topics": video_analysis.get("topics", []),
            "key_points": video_analysis.get("key_points", []),
        }

        # Generate podcast
        podcast_data = await podcast_service.generate_paper_summary_podcast(
            [video_paper], style="conversational", duration_minutes=10
        )

        podcast_data["source_video"] = video_analysis["video_url"]
        podcast_data["video_title"] = video_analysis.get("title")

        return podcast_data

    async def _get_video_metadata(self, video_url: str) -> Dict:
        """Extract video metadata."""

        try:
            # Try to use yt-dlp for YouTube videos
            if "youtube.com" in video_url or "youtu.be" in video_url:
                return await self._get_youtube_metadata(video_url)
            else:
                # For other video sources, extract basic info
                return {
                    "title": "Unknown Video",
                    "description": "",
                    "duration": 0,
                    "channel": "Unknown",
                    "upload_date": None,
                }
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {}

    async def _get_youtube_metadata(self, video_url: str) -> Dict:
        """Extract YouTube video metadata using yt-dlp."""

        try:
            # This would require yt-dlp installation
            # For now, return mock data
            return {
                "title": "Research Video",
                "description": "Academic research presentation",
                "duration": 1800,  # 30 minutes
                "channel": "Research Channel",
                "upload_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error extracting YouTube metadata: {e}")
            return {}

    async def _transcribe_video(self, video_url: str) -> str:
        """Transcribe video audio to text."""

        try:
            # Try OpenAI Whisper API first
            if self.openai_client:
                return await self._transcribe_with_openai(video_url)

            # Fallback to local Whisper
            elif self.whisper_available:
                return await self._transcribe_with_local_whisper(video_url)

            else:
                logger.warning("No transcription service available")
                return "Transcription not available"

        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
            return "Transcription failed"

    async def _transcribe_with_openai(self, video_url: str) -> str:
        """Transcribe using OpenAI Whisper API."""

        try:
            # This would require downloading audio first
            # For now, return placeholder
            return "Transcription would be performed using OpenAI Whisper API"
        except Exception as e:
            logger.error(f"Error with OpenAI transcription: {e}")
            return ""

    async def _transcribe_with_local_whisper(self, video_url: str) -> str:
        """Transcribe using local Whisper model."""

        try:
            # This would require audio extraction and processing
            # For now, return placeholder
            return "Transcription would be performed using local Whisper model"
        except Exception as e:
            logger.error(f"Error with local Whisper: {e}")
            return ""

    async def _analyze_transcript(self, transcript: str, analysis_type: str) -> Dict:
        """Analyze transcript content using LLM."""

        if not self.openai_client or not transcript:
            return self._basic_transcript_analysis(transcript)

        prompt = self._get_analysis_prompt(transcript, analysis_type)

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing academic content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )

            content = response.choices[0].message.content
            return self._parse_analysis_response(content)

        except Exception as e:
            logger.error(f"Error analyzing transcript with LLM: {e}")
            return self._basic_transcript_analysis(transcript)

    def _get_analysis_prompt(self, transcript: str, analysis_type: str) -> str:
        """Get analysis prompt based on type."""

        base_prompt = f"""
        Analyze this academic video transcript and provide:

        1. A concise summary (2-3 sentences)
        2. Key points (5-7 bullet points)
        3. Main topics/themes (3-5 topics)
        4. Confidence score (0-1) for transcript quality
        5. Language detected

        Transcript:
        {transcript[:3000]}...

        """

        if analysis_type == "lecture":
            base_prompt += """
            Additional analysis for lecture:
            - Learning objectives
            - Main concepts taught
            - Difficulty level
            """
        elif analysis_type == "presentation":
            base_prompt += """
            Additional analysis for research presentation:
            - Research methodology mentioned
            - Key findings/results
            - Future research directions
            """

        base_prompt += """
        Format response as JSON with keys: summary, key_points, topics, confidence_score, language
        """

        return base_prompt

    def _parse_analysis_response(self, content: str) -> Dict:
        """Parse LLM analysis response."""

        try:
            # Try to extract JSON from response
            import json

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback to basic parsing
        return {
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "key_points": ["Analysis available in summary"],
            "topics": ["General Research"],
            "confidence_score": 0.5,
            "language": "en",
        }

    def _basic_transcript_analysis(self, transcript: str) -> Dict:
        """Basic transcript analysis without LLM."""

        if not transcript:
            return {
                "summary": "No transcript available",
                "key_points": [],
                "topics": [],
                "confidence_score": 0.0,
                "language": "unknown",
            }

        # Simple keyword-based analysis
        words = transcript.lower().split()
        word_count = len(words)

        # Extract potential topics
        research_keywords = [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural network",
            "algorithm",
            "data science",
            "research",
            "experiment",
            "methodology",
            "results",
            "conclusion",
        ]

        topics = []
        for keyword in research_keywords:
            if keyword in transcript.lower():
                topics.append(keyword.title())

        # Generate simple summary
        sentences = transcript.split(".")
        summary = (
            ". ".join(sentences[:3]) + "."
            if len(sentences) >= 3
            else transcript[:200] + "..."
        )

        return {
            "summary": summary,
            "key_points": [
                f"Contains {word_count} words",
                f"Duration: estimated {word_count // 150} minutes",
            ],
            "topics": topics[:5],
            "confidence_score": 0.6,
            "language": "en",
        }

    async def _extract_research_insights(
        self, transcript: str, video_info: Dict
    ) -> Dict:
        """Extract research-specific insights from transcript."""

        insights = {
            "research_areas": [],
            "methodologies": [],
            "findings": [],
            "limitations": [],
            "future_directions": [],
        }

        if not transcript:
            return insights

        # Simple keyword-based extraction
        text_lower = transcript.lower()

        # Research areas
        area_keywords = {
            "machine learning": ["machine learning", "ml", "artificial intelligence"],
            "deep learning": ["deep learning", "neural network", "cnn", "rnn"],
            "natural language processing": ["nlp", "natural language", "text mining"],
            "computer vision": ["computer vision", "image recognition", "cv"],
            "data science": ["data science", "big data", "analytics"],
        }

        for area, keywords in area_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                insights["research_areas"].append(area)

        # Methodologies
        method_keywords = [
            "experiment",
            "survey",
            "case study",
            "analysis",
            "model",
            "algorithm",
            "framework",
            "approach",
            "technique",
        ]

        for keyword in method_keywords:
            if keyword in text_lower:
                insights["methodologies"].append(keyword)

        return insights

    async def _find_mentioned_papers(self, transcript: str) -> List[Dict]:
        """Find papers mentioned in the transcript."""

        mentioned = []

        if not transcript:
            return mentioned

        # Look for citation patterns
        citation_patterns = [
            r"([A-Z][a-z]+ et al\.?,? \d{4})",  # Author et al., 2023
            r"([A-Z][a-z]+ and [A-Z][a-z]+,? \d{4})",  # Author and Author, 2023
            r"([A-Z][a-z]+,? \d{4})",  # Author, 2023
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, transcript)
            for match in matches:
                mentioned.append(
                    {"citation": match, "type": "inferred", "confidence": 0.7}
                )

        # Look for paper titles (quoted text)
        title_pattern = r'"([^"]{20,100})"'
        title_matches = re.findall(title_pattern, transcript)

        for title in title_matches:
            if any(
                keyword in title.lower()
                for keyword in ["research", "study", "analysis", "investigation"]
            ):
                mentioned.append({"title": title, "type": "title", "confidence": 0.6})

        return mentioned[:10]  # Limit to top 10

    async def _extract_lecture_structure(self, transcript: str) -> Dict:
        """Extract lecture structure from transcript."""

        structure = {
            "introduction": "",
            "main_sections": [],
            "conclusion": "",
            "total_sections": 0,
        }

        if not transcript:
            return structure

        # Look for section indicators
        section_indicators = [
            "first",
            "second",
            "third",
            "next",
            "finally",
            "introduction",
            "conclusion",
            "summary",
            "part one",
            "part two",
            "chapter",
        ]

        sentences = transcript.split(".")
        sections = []

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower().strip()
            if any(indicator in sentence_lower for indicator in section_indicators):
                sections.append(
                    {
                        "position": i,
                        "content": sentence.strip(),
                        "type": "section_marker",
                    }
                )

        structure["main_sections"] = sections
        structure["total_sections"] = len(sections)

        return structure

    async def _extract_slides_references(self, transcript: str) -> List[str]:
        """Extract references to slides from transcript."""

        slide_refs = []

        if not transcript:
            return slide_refs

        # Look for slide references
        slide_patterns = [
            r"slide (\d+)",
            r"on this slide",
            r"next slide",
            r"previous slide",
            r"slide shows",
            r"as you can see",
        ]

        for pattern in slide_patterns:
            matches = re.findall(pattern, transcript.lower())
            slide_refs.extend(matches)

        return slide_refs[:20]  # Limit to 20 references

    async def _identify_learning_objectives(self, transcript: str) -> List[str]:
        """Identify learning objectives from lecture transcript."""

        objectives = []

        if not transcript:
            return objectives

        # Look for objective indicators
        objective_patterns = [
            r"you will learn (.*?)(?:\.|,|;)",
            r"objective is to (.*?)(?:\.|,|;)",
            r"goal is to (.*?)(?:\.|,|;)",
            r"by the end.*?you will (.*?)(?:\.|,|;)",
        ]

        for pattern in objective_patterns:
            matches = re.findall(pattern, transcript.lower())
            objectives.extend(matches)

        return objectives[:5]  # Limit to 5 objectives

    async def _extract_methodology(self, transcript: str) -> Dict:
        """Extract research methodology from presentation transcript."""

        methodology = {
            "approach": "",
            "data_collection": "",
            "analysis_methods": [],
            "tools_used": [],
        }

        if not transcript:
            return methodology

        # Look for methodology keywords
        method_keywords = {
            "quantitative": ["quantitative", "statistical", "numerical"],
            "qualitative": ["qualitative", "interview", "survey"],
            "experimental": ["experiment", "control group", "randomized"],
            "observational": ["observational", "case study", "longitudinal"],
        }

        text_lower = transcript.lower()

        for method_type, keywords in method_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                methodology["approach"] = method_type
                break

        # Extract tools
        tool_keywords = [
            "python",
            "r",
            "matlab",
            "spss",
            "tensorflow",
            "pytorch",
            "scikit-learn",
            "pandas",
            "numpy",
        ]

        for tool in tool_keywords:
            if tool in text_lower:
                methodology["tools_used"].append(tool)

        return methodology

    async def _extract_results(self, transcript: str) -> Dict:
        """Extract research results from presentation transcript."""

        results = {"key_findings": [], "statistics": [], "performance_metrics": []}

        if not transcript:
            return results

        # Look for result indicators
        result_patterns = [
            r"we found (.*?)(?:\.|,|;)",
            r"results show (.*?)(?:\.|,|;)",
            r"our findings (.*?)(?:\.|,|;)",
            r"accuracy of (\d+\.?\d*%?)",
            r"improvement of (\d+\.?\d*%?)",
        ]

        for pattern in result_patterns:
            matches = re.findall(pattern, transcript.lower())
            if matches:
                if "accuracy" in pattern or "improvement" in pattern:
                    results["performance_metrics"].extend(matches)
                else:
                    results["key_findings"].extend(matches)

        return results

    async def _extract_future_work(self, transcript: str) -> List[str]:
        """Extract future work directions from transcript."""

        future_work = []

        if not transcript:
            return future_work

        # Look for future work indicators
        future_patterns = [
            r"future work (.*?)(?:\.|,|;)",
            r"next steps (.*?)(?:\.|,|;)",
            r"we plan to (.*?)(?:\.|,|;)",
            r"future research (.*?)(?:\.|,|;)",
        ]

        for pattern in future_patterns:
            matches = re.findall(pattern, transcript.lower())
            future_work.extend(matches)

        return future_work[:5]  # Limit to 5 items

    async def _extract_qa_section(self, transcript: str) -> List[Dict]:
        """Extract Q&A section from transcript."""

        qa_pairs = []

        if not transcript:
            return qa_pairs

        # Look for Q&A patterns
        qa_patterns = [
            r"question: (.*?) answer: (.*?)(?:\n|$)",
            r"q: (.*?) a: (.*?)(?:\n|$)",
            r"question.*?(.*?)\?.*?answer.*?(.*?)(?:\.|$)",
        ]

        for pattern in qa_patterns:
            matches = re.findall(pattern, transcript.lower(), re.DOTALL)
            for match in matches:
                if len(match) == 2:
                    qa_pairs.append(
                        {"question": match[0].strip(), "answer": match[1].strip()}
                    )

        return qa_pairs[:10]  # Limit to 10 Q&A pairs
