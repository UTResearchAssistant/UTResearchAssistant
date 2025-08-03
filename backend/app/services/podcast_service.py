"""Podcast generation service for research papers."""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os
import hashlib
import openai
from pathlib import Path

logger = logging.getLogger(__name__)


class PodcastGenerationService:
    """Service for generating research podcasts from papers."""

    def __init__(self, config=None):
        self.config = config
        self.storage_path = (
            Path(config.storage_path if config else "./storage") / "podcasts"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client
        if config and config.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        else:
            self.openai_client = None
            logger.warning(
                "OpenAI API key not configured - podcast generation will be limited"
            )

    async def generate_paper_summary_podcast(
        self,
        papers: List[Dict],
        style: str = "conversational",
        duration_minutes: int = 15,
        voice_config: Dict = None,
    ) -> Dict:
        """Generate a podcast episode summarizing research papers."""

        if not papers:
            raise ValueError("No papers provided for podcast generation")

        # Generate podcast script
        script = await self._generate_podcast_script(papers, style, duration_minutes)

        # Generate audio
        audio_files = await self._generate_audio(script, voice_config or {})

        # Combine audio files
        final_audio_path = await self._combine_audio_files(audio_files)

        # Create episode metadata
        episode_data = {
            "title": self._generate_episode_title(papers),
            "description": self._generate_episode_description(papers),
            "script": script,
            "audio_url": str(final_audio_path),
            "duration": await self._get_audio_duration(final_audio_path),
            "paper_ids": [paper.get("id") for paper in papers if paper.get("id")],
            "speaker_voices": voice_config,
            "episode_type": "summary",
            "topics": self._extract_topics(papers),
            "generated_at": datetime.utcnow(),
            "style": style,
        }

        return episode_data

    async def generate_interview_style_podcast(
        self,
        papers: List[Dict],
        interviewer_questions: List[str] = None,
        duration_minutes: int = 20,
    ) -> Dict:
        """Generate an interview-style podcast about research papers."""

        # Generate interview questions if not provided
        if not interviewer_questions:
            interviewer_questions = await self._generate_interview_questions(papers)

        # Generate interview script
        script = await self._generate_interview_script(papers, interviewer_questions)

        # Configure voices (interviewer and interviewee)
        voice_config = {
            "interviewer": {"voice": "alloy", "style": "curious"},
            "expert": {"voice": "echo", "style": "knowledgeable"},
        }

        # Generate audio
        audio_files = await self._generate_audio(script, voice_config)
        final_audio_path = await self._combine_audio_files(audio_files)

        episode_data = {
            "title": f"Research Interview: {self._generate_episode_title(papers)}",
            "description": f"An in-depth interview discussing recent research in {self._get_main_topic(papers)}",
            "script": script,
            "audio_url": str(final_audio_path),
            "duration": await self._get_audio_duration(final_audio_path),
            "paper_ids": [paper.get("id") for paper in papers if paper.get("id")],
            "speaker_voices": voice_config,
            "episode_type": "interview",
            "topics": self._extract_topics(papers),
            "generated_at": datetime.utcnow(),
            "questions": interviewer_questions,
        }

        return episode_data

    async def generate_debate_podcast(
        self, papers: List[Dict], controversial_topic: str
    ) -> Dict:
        """Generate a debate-style podcast exploring different viewpoints."""

        # Analyze papers for different viewpoints
        viewpoints = await self._analyze_viewpoints(papers, controversial_topic)

        # Generate debate script
        script = await self._generate_debate_script(
            papers, viewpoints, controversial_topic
        )

        # Configure multiple voices for debate participants
        voice_config = {
            "moderator": {"voice": "alloy", "style": "neutral"},
            "advocate": {"voice": "echo", "style": "passionate"},
            "skeptic": {"voice": "fable", "style": "analytical"},
        }

        audio_files = await self._generate_audio(script, voice_config)
        final_audio_path = await self._combine_audio_files(audio_files)

        episode_data = {
            "title": f"Research Debate: {controversial_topic}",
            "description": f"A structured debate exploring different perspectives on {controversial_topic}",
            "script": script,
            "audio_url": str(final_audio_path),
            "duration": await self._get_audio_duration(final_audio_path),
            "paper_ids": [paper.get("id") for paper in papers if paper.get("id")],
            "speaker_voices": voice_config,
            "episode_type": "debate",
            "topics": [controversial_topic] + self._extract_topics(papers),
            "generated_at": datetime.utcnow(),
            "debate_topic": controversial_topic,
            "viewpoints": viewpoints,
        }

        return episode_data

    async def _generate_podcast_script(
        self, papers: List[Dict], style: str, duration_minutes: int
    ) -> str:
        """Generate podcast script using LLM."""

        if not self.openai_client:
            return self._generate_basic_script(papers)

        # Prepare papers summary for LLM
        papers_summary = self._prepare_papers_for_llm(papers)

        prompt = f"""
        Create a {duration_minutes}-minute podcast script in {style} style about the following research papers:

        {papers_summary}

        Requirements:
        - Engaging and accessible to general audience
        - Include key findings and implications
        - Add natural transitions and conversation flow
        - Include speaker cues in [SPEAKER: name] format
        - Make it exactly {duration_minutes} minutes when read aloud
        - Include interesting analogies and real-world applications

        Style guidelines for {style}:
        - conversational: Friendly, casual tone with personal anecdotes
        - academic: Professional, detailed explanations
        - narrative: Story-driven with compelling structure
        - educational: Clear explanations with examples

        Format the script with clear speaker tags and natural dialogue.
        """

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert podcast script writer specializing in research communication.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3000,
                temperature=0.7,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating script with OpenAI: {e}")
            return self._generate_basic_script(papers)

    async def _generate_interview_questions(self, papers: List[Dict]) -> List[str]:
        """Generate interview questions about the research papers."""

        if not self.openai_client:
            return self._get_default_interview_questions()

        papers_summary = self._prepare_papers_for_llm(papers)

        prompt = f"""
        Generate 8-10 insightful interview questions about these research papers:

        {papers_summary}

        The questions should:
        - Be suitable for a general audience
        - Explore implications and applications
        - Encourage discussion of methodology
        - Ask about future research directions
        - Include some provocative but respectful questions
        - Build upon each other logically

        Format as a numbered list.
        """

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert interviewer specializing in academic research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            questions_text = response.choices[0].message.content
            questions = [
                q.strip()
                for q in questions_text.split("\n")
                if q.strip() and q[0].isdigit()
            ]
            return questions
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            return self._get_default_interview_questions()

    async def _generate_interview_script(
        self, papers: List[Dict], questions: List[str]
    ) -> str:
        """Generate interview script with questions and responses."""

        if not self.openai_client:
            return self._generate_basic_interview_script(papers, questions)

        papers_summary = self._prepare_papers_for_llm(papers)
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        prompt = f"""
        Create a podcast interview script based on these research papers and questions:

        PAPERS:
        {papers_summary}

        QUESTIONS:
        {questions_text}

        Create a natural conversation between an INTERVIEWER and an EXPERT. Include:
        - Natural transitions between questions
        - Follow-up questions based on responses
        - Clear speaker tags [INTERVIEWER] and [EXPERT]
        - Engaging explanations accessible to general audience
        - About 15-20 minutes of content when read aloud

        Make it sound like a real conversation, not scripted Q&A.
        """

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating engaging interview content about research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.7,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating interview script: {e}")
            return self._generate_basic_interview_script(papers, questions)

    async def _analyze_viewpoints(self, papers: List[Dict], topic: str) -> Dict:
        """Analyze papers for different viewpoints on a controversial topic."""

        viewpoints = {"supporting": [], "opposing": [], "neutral": []}

        for paper in papers:
            # Simple keyword-based analysis (can be enhanced with ML)
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            # This is a simplified approach - in production, use sentiment analysis
            if any(
                word in text.lower()
                for word in ["benefit", "advantage", "improve", "effective"]
            ):
                viewpoints["supporting"].append(paper)
            elif any(
                word in text.lower()
                for word in ["risk", "danger", "concern", "limitation"]
            ):
                viewpoints["opposing"].append(paper)
            else:
                viewpoints["neutral"].append(paper)

        return viewpoints

    async def _generate_debate_script(
        self, papers: List[Dict], viewpoints: Dict, topic: str
    ) -> str:
        """Generate debate script with multiple perspectives."""

        if not self.openai_client:
            return self._generate_basic_debate_script(papers, topic)

        papers_summary = self._prepare_papers_for_llm(papers)

        prompt = f"""
        Create a structured debate podcast script about: {topic}

        Papers to reference:
        {papers_summary}

        Structure the debate with:
        - [MODERATOR] - introduces topic and manages discussion
        - [ADVOCATE] - argues for the positive aspects
        - [SKEPTIC] - raises concerns and counterpoints

        Include:
        - Opening statements from each side
        - Evidence from the research papers
        - Cross-examination/rebuttals
        - Closing statements
        - Moderator summary

        Keep it respectful but engaging, about 20 minutes when spoken.
        """

        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert debate moderator and content creator.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.7,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating debate script: {e}")
            return self._generate_basic_debate_script(papers, topic)

    async def _generate_audio(self, script: str, voice_config: Dict) -> List[str]:
        """Generate audio files from script using TTS."""

        if not self.openai_client:
            logger.warning("Cannot generate audio without OpenAI API key")
            return []

        audio_files = []

        # Parse script for different speakers
        segments = self._parse_script_segments(script)

        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "narrator")
            text = segment.get("text", "")

            if not text.strip():
                continue

            # Get voice for speaker
            voice = voice_config.get(speaker.lower(), {}).get("voice", "alloy")

            try:
                # Generate audio with OpenAI TTS
                response = await self.openai_client.audio.speech.acreate(
                    model="tts-1-hd", voice=voice, input=text
                )

                # Save audio file
                audio_filename = self.storage_path / f"segment_{i}_{speaker}.mp3"
                with open(audio_filename, "wb") as f:
                    f.write(response.content)

                audio_files.append(str(audio_filename))

            except Exception as e:
                logger.error(f"Error generating audio for segment {i}: {e}")

        return audio_files

    async def _combine_audio_files(self, audio_files: List[str]) -> str:
        """Combine multiple audio files into one."""

        if not audio_files:
            return ""

        if len(audio_files) == 1:
            return audio_files[0]

        # Generate unique filename
        combined_filename = (
            self.storage_path
            / f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        )

        try:
            # This is a placeholder - in production, use audio processing library like pydub
            # For now, just return the first file
            return audio_files[0]
        except Exception as e:
            logger.error(f"Error combining audio files: {e}")
            return audio_files[0] if audio_files else ""

    def _parse_script_segments(self, script: str) -> List[Dict]:
        """Parse script into segments by speaker."""

        segments = []
        current_speaker = "narrator"
        current_text = ""

        lines = script.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for speaker tags like [SPEAKER: name] or [NAME]
            if line.startswith("[") and "]:" in line:
                # Save previous segment
                if current_text.strip():
                    segments.append(
                        {"speaker": current_speaker, "text": current_text.strip()}
                    )

                # Extract new speaker
                current_speaker = line.split(":")[0][1:].strip().lower()
                current_text = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.startswith("[") and line.endswith("]"):
                # Save previous segment
                if current_text.strip():
                    segments.append(
                        {"speaker": current_speaker, "text": current_text.strip()}
                    )

                # Extract new speaker
                current_speaker = line[1:-1].strip().lower()
                current_text = ""
            else:
                current_text += " " + line

        # Add final segment
        if current_text.strip():
            segments.append({"speaker": current_speaker, "text": current_text.strip()})

        return segments

    def _prepare_papers_for_llm(self, papers: List[Dict]) -> str:
        """Prepare papers summary for LLM input."""

        summaries = []
        for i, paper in enumerate(papers, 1):
            summary = f"""
Paper {i}:
Title: {paper.get('title', 'N/A')}
Authors: {', '.join(paper.get('authors', [])[:3])}
Abstract: {paper.get('abstract', 'N/A')[:500]}...
Citations: {paper.get('citation_count', 0)}
Source: {paper.get('source', 'N/A')}
"""
            summaries.append(summary)

        return "\n".join(summaries)

    def _generate_basic_script(self, papers: List[Dict]) -> str:
        """Generate basic script without LLM."""

        script = "[NARRATOR]: Welcome to Research Insights, where we explore the latest developments in academic research.\n\n"

        for i, paper in enumerate(papers, 1):
            script += f"[NARRATOR]: Let's look at paper {i}: {paper.get('title', 'Untitled')}.\n"
            script += f"[NARRATOR]: This research by {', '.join(paper.get('authors', ['Unknown'])[:2])} "
            script += f"explores {paper.get('abstract', 'various topics')[:200]}...\n\n"

        script += (
            "[NARRATOR]: Thank you for listening to Research Insights. Stay curious!"
        )

        return script

    def _generate_episode_title(self, papers: List[Dict]) -> str:
        """Generate episode title from papers."""

        if len(papers) == 1:
            return f"Research Spotlight: {papers[0].get('title', 'Latest Research')[:50]}..."
        else:
            main_topic = self._get_main_topic(papers)
            return f"Research Roundup: Latest in {main_topic}"

    def _generate_episode_description(self, papers: List[Dict]) -> str:
        """Generate episode description."""

        description = f"In this episode, we explore {len(papers)} recent research paper"
        description += "s" if len(papers) > 1 else ""
        description += f" covering {self._get_main_topic(papers)}. "

        if papers:
            description += (
                f"Key topics include: {', '.join(self._extract_topics(papers)[:3])}."
            )

        return description

    def _extract_topics(self, papers: List[Dict]) -> List[str]:
        """Extract main topics from papers."""

        topics = set()

        for paper in papers:
            # Extract from categories if available
            if paper.get("categories"):
                topics.update(paper["categories"][:2])

            # Extract from keywords if available
            if paper.get("keywords"):
                topics.update(paper["keywords"][:3])

            # Simple topic extraction from title
            title = paper.get("title", "").lower()
            if "machine learning" in title or "ml" in title:
                topics.add("Machine Learning")
            if "artificial intelligence" in title or "ai" in title:
                topics.add("Artificial Intelligence")
            if "deep learning" in title:
                topics.add("Deep Learning")
            if "neural network" in title:
                topics.add("Neural Networks")

        return list(topics)[:5]

    def _get_main_topic(self, papers: List[Dict]) -> str:
        """Get main topic from papers."""

        topics = self._extract_topics(papers)
        return topics[0] if topics else "Research"

    async def _get_audio_duration(self, audio_path: str) -> int:
        """Get audio duration in seconds."""
        # Placeholder - in production, use audio library to get actual duration
        return 900  # 15 minutes default

    def _get_default_interview_questions(self) -> List[str]:
        """Get default interview questions."""

        return [
            "What was the main motivation behind this research?",
            "Can you explain the methodology in simple terms?",
            "What were the most surprising findings?",
            "How does this work build on previous research?",
            "What are the practical applications of these findings?",
            "What challenges did you face during the research?",
            "How might this research impact the field in the next 5 years?",
            "What questions does this research raise for future study?",
        ]

    def _generate_basic_interview_script(
        self, papers: List[Dict], questions: List[str]
    ) -> str:
        """Generate basic interview script without LLM."""

        script = "[INTERVIEWER]: Welcome to Research Talk. Today we're discussing some fascinating recent research.\n\n"

        for question in questions[:5]:  # Limit to 5 questions for basic version
            script += f"[INTERVIEWER]: {question}\n"
            script += "[EXPERT]: That's a great question. Based on the research we've reviewed, "
            script += "this work represents an important step forward in our understanding...\n\n"

        script += "[INTERVIEWER]: Thank you for that insightful discussion!"

        return script

    def _generate_basic_debate_script(self, papers: List[Dict], topic: str) -> str:
        """Generate basic debate script without LLM."""

        script = f"[MODERATOR]: Welcome to Research Debate. Today we're discussing {topic}.\n\n"
        script += "[ADVOCATE]: Based on recent research, there are compelling reasons to support this position...\n"
        script += "[SKEPTIC]: However, we must consider the potential limitations and concerns...\n"
        script += "[MODERATOR]: Let's examine the evidence from both perspectives.\n"

        return script
