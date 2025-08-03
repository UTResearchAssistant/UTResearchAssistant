"""Writing Assistance Service with AI-powered writing help.

This service provides comprehensive writing assistance including content
improvement, style suggestions, citation formatting, and readability analysis.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Optional, List, Any
from datetime import datetime

import openai
from django.conf import settings
import textstat

from core.models import WritingAssistance

logger = logging.getLogger(__name__)


class WritingAssistanceService:
    """AI-powered writing assistance service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.task_types = {
            'literature_review': self._assist_literature_review,
            'abstract': self._assist_abstract,
            'introduction': self._assist_introduction,
            'methodology': self._assist_methodology,
            'results': self._assist_results,
            'discussion': self._assist_discussion,
            'conclusion': self._assist_conclusion,
            'citation': self._assist_citation,
        }
        self.tones = ['academic', 'formal', 'technical', 'accessible']
    
    async def assist_writing(
        self,
        content: str,
        task_type: str,
        tone: str = 'academic',
        target_audience: str = 'researchers',
        user=None
    ) -> Dict:
        """
        Provide comprehensive writing assistance.
        
        Args:
            content: Text content to improve
            task_type: Type of writing task
            tone: Desired tone for the writing
            target_audience: Target audience for the content
            user: User requesting assistance
            
        Returns:
            Dictionary containing writing assistance results
        """
        assistance_start = datetime.now()
        
        try:
            logger.info(f"Providing {task_type} writing assistance")
            
            # Analyze current content
            content_analysis = await self._analyze_content(content)
            
            # Generate improvements based on task type
            if task_type in self.task_types:
                improvements = await self.task_types[task_type](
                    content, tone, target_audience
                )
            else:
                improvements = await self._general_writing_assistance(
                    content, tone, target_audience
                )
            
            # Get style suggestions
            style_suggestions = await self._get_style_suggestions(
                content, tone, target_audience
            )
            
            # Check citations if applicable
            citation_suggestions = await self._check_citations(content)
            
            # Calculate readability metrics
            readability = self._calculate_readability(content)
            
            # Create writing assistance record
            processing_time = (datetime.now() - assistance_start).total_seconds()
            
            assistance = WritingAssistance.objects.create(
                user=user,
                title=f"{task_type.replace('_', ' ').title()} Assistance",
                task_type=task_type,
                content=content,
                improved_content=improvements['improved_content'],
                suggestions=json.dumps(improvements['suggestions']),
                style_improvements=json.dumps(style_suggestions),
                citation_corrections=json.dumps(citation_suggestions),
                readability_score=readability['flesch_reading_ease'],
                word_count=content_analysis['word_count'],
                tone=tone,
                target_audience=target_audience
            )
            
            logger.info(f"Successfully provided writing assistance: {assistance.id}")
            
            return {
                'success': True,
                'assistance': assistance,
                'content_analysis': content_analysis,
                'improvements': improvements,
                'style_suggestions': style_suggestions,
                'citation_suggestions': citation_suggestions,
                'readability': readability,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error providing writing assistance: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - assistance_start).total_seconds()
            }
    
    async def _analyze_content(self, content: str) -> Dict:
        """Analyze content for basic metrics."""
        return {
            'word_count': len(content.split()),
            'character_count': len(content),
            'sentence_count': len(re.split(r'[.!?]+', content)),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'avg_sentence_length': len(content.split()) / max(len(re.split(r'[.!?]+', content)), 1)
        }
    
    async def _general_writing_assistance(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Provide general writing assistance and improvements."""
        try:
            prompt = f"""
            Improve this academic writing for {target_audience} with a {tone} tone.
            
            Original content:
            {content}
            
            Please provide:
            1. An improved version of the content
            2. Specific suggestions for improvement
            3. Explanation of changes made
            
            Format as JSON:
            {{
                "improved_content": "enhanced version",
                "suggestions": [
                    {{"type": "clarity", "description": "suggestion"}},
                    {{"type": "style", "description": "suggestion"}}
                ],
                "changes_summary": "explanation of major changes"
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic writing assistant and editor with expertise in improving clarity, style, and academic rigor."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.4
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                content_text = response.choices[0].message.content
                return {
                    "improved_content": content_text[:len(content_text)//2],
                    "suggestions": [{"type": "general", "description": "Content has been improved for clarity and style"}],
                    "changes_summary": "General improvements applied"
                }
                
        except Exception as e:
            logger.error(f"Error in general writing assistance: {e}")
            return {
                "improved_content": content,
                "suggestions": [],
                "changes_summary": "No improvements applied due to processing error"
            }
    
    async def _assist_literature_review(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with literature review writing."""
        prompt = f"""
        Improve this literature review section for {target_audience} with a {tone} tone.
        Focus on:
        - Proper synthesis of sources
        - Clear thematic organization
        - Critical analysis rather than just summary
        - Smooth transitions between ideas
        - Identification of gaps in the literature
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "literature review specialist")
    
    async def _assist_abstract(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with abstract writing."""
        prompt = f"""
        Improve this abstract for {target_audience} with a {tone} tone.
        Ensure it includes:
        - Clear problem statement
        - Methodology overview
        - Key findings
        - Significance/implications
        - Proper word count (150-250 words typically)
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "abstract writing specialist")
    
    async def _assist_introduction(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with introduction writing."""
        prompt = f"""
        Improve this introduction for {target_audience} with a {tone} tone.
        Focus on:
        - Engaging opening
        - Clear context and background
        - Problem statement
        - Research questions/objectives
        - Study significance
        - Paper structure overview
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "introduction writing specialist")
    
    async def _assist_methodology(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with methodology writing."""
        prompt = f"""
        Improve this methodology section for {target_audience} with a {tone} tone.
        Ensure:
        - Clear description of methods
        - Justification for chosen approaches
        - Sufficient detail for replication
        - Proper organization of subsections
        - Appropriate level of technical detail
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "methodology writing specialist")
    
    async def _assist_results(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with results section writing."""
        prompt = f"""
        Improve this results section for {target_audience} with a {tone} tone.
        Focus on:
        - Objective presentation of findings
        - Clear organization of results
        - Proper reference to figures/tables
        - Statistical reporting accuracy
        - Avoiding interpretation (save for discussion)
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "results writing specialist")
    
    async def _assist_discussion(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with discussion section writing."""
        prompt = f"""
        Improve this discussion section for {target_audience} with a {tone} tone.
        Ensure:
        - Interpretation of results
        - Comparison with existing literature
        - Addressing limitations
        - Implications and significance
        - Future research directions
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "discussion writing specialist")
    
    async def _assist_conclusion(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with conclusion writing."""
        prompt = f"""
        Improve this conclusion for {target_audience} with a {tone} tone.
        Include:
        - Summary of key findings
        - Broader implications
        - Contributions to the field
        - Final thoughts on significance
        - Avoid introducing new information
        
        Content:
        {content}
        
        Provide improved version and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "conclusion writing specialist")
    
    async def _assist_citation(
        self, content: str, tone: str, target_audience: str
    ) -> Dict:
        """Assist with citation formatting and accuracy."""
        prompt = f"""
        Check and improve the citations in this text.
        Focus on:
        - Proper citation format consistency
        - Appropriate use of citations
        - Missing citations for claims
        - Over-citation issues
        - Citation accuracy
        
        Content:
        {content}
        
        Provide improved version with corrected citations and specific suggestions in JSON format.
        """
        
        return await self._call_writing_api(prompt, "citation specialist")
    
    async def _call_writing_api(self, prompt: str, specialist_role: str) -> Dict:
        """Generic method to call OpenAI API for writing assistance."""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert academic {specialist_role}. Provide detailed, constructive feedback and improvements."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.4
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                content_text = response.choices[0].message.content
                return {
                    "improved_content": content_text,
                    "suggestions": [{"type": "general", "description": "Content has been improved"}],
                    "changes_summary": "Improvements applied"
                }
                
        except Exception as e:
            logger.error(f"Error in writing API call: {e}")
            return {
                "improved_content": "Original content",
                "suggestions": [],
                "changes_summary": "No improvements applied due to processing error"
            }
    
    async def _get_style_suggestions(
        self, content: str, tone: str, target_audience: str
    ) -> List[Dict]:
        """Get style-specific suggestions for improvement."""
        try:
            prompt = f"""
            Analyze this text for style improvements targeting {target_audience} with a {tone} tone.
            
            Text:
            {content}
            
            Provide specific style suggestions as JSON array:
            [
                {{"category": "clarity", "suggestion": "specific suggestion", "severity": "high/medium/low"}},
                {{"category": "conciseness", "suggestion": "specific suggestion", "severity": "high/medium/low"}}
            ]
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in academic writing style and clarity. Provide specific, actionable suggestions."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return [{"category": "general", "suggestion": "Consider improving overall clarity and style", "severity": "medium"}]
                
        except Exception as e:
            logger.error(f"Error getting style suggestions: {e}")
            return []
    
    async def _check_citations(self, content: str) -> List[Dict]:
        """Check citation formatting and completeness."""
        try:
            # Basic citation pattern detection
            citation_patterns = [
                r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
                r'\[\d+\]',              # [1]
                r'\([^)]*et al\.[^)]*\)', # (Smith et al., 2023)
            ]
            
            citations_found = []
            for pattern in citation_patterns:
                citations_found.extend(re.findall(pattern, content))
            
            suggestions = []
            
            if not citations_found:
                suggestions.append({
                    "type": "missing_citations",
                    "description": "Consider adding citations to support your claims",
                    "severity": "medium"
                })
            
            # Check for common citation issues
            if re.search(r'[.!?]\s*\([^)]*\d{4}[^)]*\)', content):
                suggestions.append({
                    "type": "citation_placement",
                    "description": "Consider placing citations before punctuation",
                    "severity": "low"
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error checking citations: {e}")
            return []
    
    def _calculate_readability(self, content: str) -> Dict:
        """Calculate various readability metrics."""
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(content),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(content),
                'gunning_fog': textstat.gunning_fog(content),
                'automated_readability_index': textstat.automated_readability_index(content),
                'coleman_liau_index': textstat.coleman_liau_index(content),
                'reading_time': textstat.reading_time(content, ms_per_char=14.69)
            }
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'reading_time': 0
            }
    
    async def get_writing_analytics(self, user) -> Dict:
        """Get writing analytics for a user."""
        assistances = WritingAssistance.objects.filter(user=user)
        
        if not assistances.exists():
            return {
                'total_sessions': 0,
                'avg_word_count': 0,
                'avg_readability': 0,
                'common_task_types': [],
                'improvement_trend': []
            }
        
        task_type_counts = {}
        for assistance in assistances:
            task_type = assistance.task_type
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        return {
            'total_sessions': assistances.count(),
            'avg_word_count': assistances.aggregate(
                avg=models.Avg('word_count')
            )['avg'] or 0,
            'avg_readability': assistances.aggregate(
                avg=models.Avg('readability_score')
            )['avg'] or 0,
            'common_task_types': sorted(
                task_type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'recent_sessions': assistances.order_by('-created_at')[:10]
        }


# Create singleton instance
writing_assistance_service = WritingAssistanceService()
