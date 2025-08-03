"""Academic Integrity Service with AI-powered plagiarism detection and citation checking.

This service provides comprehensive academic integrity checking including
plagiarism detection, citation verification, and academic standards compliance.
"""

import asyncio
import json
import logging
import re
import hashlib
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import difflib

import openai
import requests
from django.conf import settings
from django.db.models import Q

from core.models import IntegrityCheck, Paper

logger = logging.getLogger(__name__)


class AcademicIntegrityService:
    """AI-powered academic integrity checking service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.check_types = {
            'plagiarism': self._check_plagiarism,
            'citation': self._check_citations,
            'style': self._check_style,
            'comprehensive': self._comprehensive_check,
        }
        self.similarity_threshold = 0.15  # 15% similarity threshold
        self.chunk_size = 500  # Words per chunk for analysis
    
    async def check_integrity(
        self,
        content: str,
        title: str,
        check_type: str = 'comprehensive',
        user=None
    ) -> Dict:
        """
        Perform comprehensive academic integrity checking.
        
        Args:
            content: Text content to check
            title: Title for the check
            check_type: Type of check to perform
            user: User requesting the check
            
        Returns:
            Dictionary containing integrity check results
        """
        check_start = datetime.now()
        
        try:
            logger.info(f"Starting {check_type} integrity check: {title}")
            
            # Create integrity check record
            integrity_check = IntegrityCheck.objects.create(
                user=user,
                title=title,
                content=content,
                check_type=check_type,
                status='processing'
            )
            
            # Perform the requested check
            if check_type in self.check_types:
                results = await self.check_types[check_type](content)
            else:
                raise ValueError(f"Unsupported check type: {check_type}")
            
            # Calculate overall score and recommendation
            overall_analysis = await self._calculate_overall_score(results)
            
            # Update integrity check record
            processing_time = (datetime.now() - check_start).total_seconds()
            
            integrity_check.plagiarism_score = results.get('plagiarism_score', 0.0)
            integrity_check.similarity_matches = json.dumps(results.get('similarity_matches', []))
            integrity_check.citation_issues = json.dumps(results.get('citation_issues', []))
            integrity_check.style_violations = json.dumps(results.get('style_violations', []))
            integrity_check.overall_score = overall_analysis['score']
            integrity_check.recommendation = overall_analysis['recommendation']
            integrity_check.issues_found = overall_analysis['issues_count']
            integrity_check.status = 'completed'
            integrity_check.save()
            
            logger.info(f"Successfully completed integrity check: {integrity_check.id}")
            
            return {
                'success': True,
                'check': integrity_check,
                'results': results,
                'overall_analysis': overall_analysis,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in integrity check: {e}")
            
            # Update record with error
            if 'integrity_check' in locals():
                integrity_check.status = 'failed'
                integrity_check.save()
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - check_start).total_seconds()
            }
    
    async def _check_plagiarism(self, content: str) -> Dict:
        """Check for plagiarism against database and online sources."""
        try:
            logger.info("Checking for plagiarism...")
            
            # Split content into chunks for analysis
            chunks = self._split_into_chunks(content)
            
            similarity_matches = []
            max_similarity = 0.0
            
            # Check against local database
            local_matches = await self._check_against_local_database(chunks)
            similarity_matches.extend(local_matches)
            
            # Check against web sources (simulated)
            web_matches = await self._check_against_web_sources(chunks)
            similarity_matches.extend(web_matches)
            
            # Calculate overall plagiarism score
            if similarity_matches:
                max_similarity = max(match['similarity'] for match in similarity_matches)
            
            # Use AI to analyze potential plagiarism
            ai_analysis = await self._analyze_similarity_with_ai(content, similarity_matches)
            
            return {
                'plagiarism_score': max_similarity,
                'similarity_matches': similarity_matches,
                'ai_analysis': ai_analysis,
                'total_matches': len(similarity_matches)
            }
            
        except Exception as e:
            logger.error(f"Error checking plagiarism: {e}")
            return {
                'plagiarism_score': 0.0,
                'similarity_matches': [],
                'ai_analysis': {'error': str(e)},
                'total_matches': 0
            }
    
    async def _check_citations(self, content: str) -> Dict:
        """Check citation formatting and completeness."""
        try:
            logger.info("Checking citations...")
            
            # Extract citations from content
            citations = self._extract_citations(content)
            
            # Check citation formatting
            formatting_issues = await self._check_citation_formatting(citations)
            
            # Check for missing citations
            missing_citations = await self._identify_missing_citations(content)
            
            # Check citation accuracy (if possible)
            accuracy_issues = await self._verify_citation_accuracy(citations)
            
            citation_issues = formatting_issues + missing_citations + accuracy_issues
            
            # Calculate citation quality score
            citation_score = max(0, 1.0 - (len(citation_issues) * 0.1))
            
            return {
                'citation_score': citation_score,
                'citation_issues': citation_issues,
                'total_citations': len(citations),
                'issues_count': len(citation_issues)
            }
            
        except Exception as e:
            logger.error(f"Error checking citations: {e}")
            return {
                'citation_score': 0.0,
                'citation_issues': [],
                'total_citations': 0,
                'issues_count': 0
            }
    
    async def _check_style(self, content: str) -> Dict:
        """Check for academic style violations."""
        try:
            logger.info("Checking academic style...")
            
            style_violations = []
            
            # Check for common style issues
            style_violations.extend(self._check_passive_voice(content))
            style_violations.extend(self._check_first_person(content))
            style_violations.extend(self._check_contractions(content))
            style_violations.extend(self._check_informal_language(content))
            style_violations.extend(await self._check_academic_tone(content))
            
            # Calculate style score
            style_score = max(0, 1.0 - (len(style_violations) * 0.05))
            
            return {
                'style_score': style_score,
                'style_violations': style_violations,
                'violations_count': len(style_violations)
            }
            
        except Exception as e:
            logger.error(f"Error checking style: {e}")
            return {
                'style_score': 0.0,
                'style_violations': [],
                'violations_count': 0
            }
    
    async def _comprehensive_check(self, content: str) -> Dict:
        """Perform comprehensive integrity checking."""
        try:
            logger.info("Performing comprehensive integrity check...")
            
            # Run all checks
            plagiarism_results = await self._check_plagiarism(content)
            citation_results = await self._check_citations(content)
            style_results = await self._check_style(content)
            
            # Combine results
            return {
                **plagiarism_results,
                **citation_results,
                **style_results,
                'check_type': 'comprehensive'
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive check: {e}")
            return {
                'error': str(e),
                'check_type': 'comprehensive'
            }
    
    def _split_into_chunks(self, content: str, chunk_size: int = None) -> List[str]:
        """Split content into overlapping chunks for analysis."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    async def _check_against_local_database(self, chunks: List[str]) -> List[Dict]:
        """Check chunks against local paper database."""
        matches = []
        
        try:
            for chunk in chunks:
                # Create hash for efficient lookup
                chunk_hash = hashlib.md5(chunk.lower().encode()).hexdigest()
                
                # Search for similar content in database
                papers = Paper.objects.filter(
                    Q(content__icontains=chunk[:100]) |
                    Q(abstract__icontains=chunk[:100])
                )[:5]  # Limit search
                
                for paper in papers:
                    # Calculate similarity
                    similarity = self._calculate_text_similarity(
                        chunk, paper.content or paper.abstract
                    )
                    
                    if similarity > 0.3:  # 30% similarity threshold
                        matches.append({
                            'source': f"Local Database - {paper.title}",
                            'similarity': similarity,
                            'matched_text': chunk[:200],
                            'source_text': (paper.content or paper.abstract)[:200],
                            'paper_id': str(paper.id),
                            'type': 'local_database'
                        })
        
        except Exception as e:
            logger.error(f"Error checking local database: {e}")
        
        return matches
    
    async def _check_against_web_sources(self, chunks: List[str]) -> List[Dict]:
        """Check chunks against web sources (simulated)."""
        matches = []
        
        try:
            # In a real implementation, you would:
            # 1. Use a plagiarism detection API (like Copyleaks, Turnitin API)
            # 2. Search Google for exact phrases
            # 3. Check against academic databases
            
            # For demo purposes, we'll simulate some matches
            for i, chunk in enumerate(chunks[:3]):  # Limit for demo
                if len(chunk.split()) > 20:
                    # Simulate finding a match
                    if i % 3 == 0:  # Simulate occasional matches
                        matches.append({
                            'source': 'Web Source (Simulated)',
                            'similarity': 0.25,
                            'matched_text': chunk[:200],
                            'source_text': 'Simulated matching content from web...',
                            'url': 'https://example.com/source',
                            'type': 'web_source'
                        })
        
        except Exception as e:
            logger.error(f"Error checking web sources: {e}")
        
        return matches
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            # Use difflib for basic similarity calculation
            similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            return similarity
        except Exception:
            return 0.0
    
    async def _analyze_similarity_with_ai(
        self, content: str, matches: List[Dict]
    ) -> Dict:
        """Use AI to analyze similarity matches for potential plagiarism."""
        try:
            if not matches:
                return {'assessment': 'No significant similarities found'}
            
            # Prepare matches summary for AI analysis
            matches_summary = []
            for match in matches[:5]:  # Limit to top 5 matches
                matches_summary.append(
                    f"Similarity: {match['similarity']:.2%}, Source: {match['source']}"
                )
            
            prompt = f"""
            Analyze these similarity matches for potential plagiarism concerns:
            
            Content length: {len(content.split())} words
            
            Similarity matches found:
            {chr(10).join(matches_summary)}
            
            Provide an assessment of whether this constitutes plagiarism and recommendations.
            Format as JSON:
            {{
                "assessment": "assessment of plagiarism risk",
                "severity": "low/medium/high",
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in academic integrity and plagiarism detection. Provide objective assessments based on similarity evidence."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return {
                    'assessment': response.choices[0].message.content,
                    'severity': 'medium',
                    'recommendations': ['Review similarity matches carefully']
                }
                
        except Exception as e:
            logger.error(f"Error in AI similarity analysis: {e}")
            return {'assessment': 'Analysis unavailable', 'severity': 'unknown'}
    
    def _extract_citations(self, content: str) -> List[Dict]:
        """Extract citations from content."""
        citations = []
        
        # Common citation patterns
        patterns = [
            (r'\(([^)]*\d{4}[^)]*)\)', 'parenthetical'),  # (Author, 2023)
            (r'\[(\d+)\]', 'numbered'),                    # [1]
            (r'(\w+\s+et\s+al\.\s*,?\s*\d{4})', 'et_al'), # Smith et al., 2023
        ]
        
        for pattern, citation_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                citations.append({
                    'text': match.group(0),
                    'type': citation_type,
                    'position': match.start(),
                    'content': match.group(1) if match.groups() else match.group(0)
                })
        
        return citations
    
    async def _check_citation_formatting(self, citations: List[Dict]) -> List[Dict]:
        """Check citation formatting consistency."""
        issues = []
        
        if not citations:
            return issues
        
        # Check for mixed citation styles
        citation_types = set(citation['type'] for citation in citations)
        if len(citation_types) > 1:
            issues.append({
                'type': 'mixed_citation_styles',
                'description': f'Mixed citation styles found: {", ".join(citation_types)}',
                'severity': 'medium'
            })
        
        # Check for formatting inconsistencies within the same style
        for citation_type in citation_types:
            type_citations = [c for c in citations if c['type'] == citation_type]
            
            if citation_type == 'parenthetical':
                # Check for consistent format in parenthetical citations
                formats = set()
                for citation in type_citations:
                    # Simplified format checking
                    if ',' in citation['content']:
                        formats.add('author_year_comma')
                    else:
                        formats.add('author_year_no_comma')
                
                if len(formats) > 1:
                    issues.append({
                        'type': 'inconsistent_parenthetical_format',
                        'description': 'Inconsistent parenthetical citation formatting',
                        'severity': 'low'
                    })
        
        return issues
    
    async def _identify_missing_citations(self, content: str) -> List[Dict]:
        """Identify potential claims that need citations."""
        issues = []
        
        try:
            # Use AI to identify claims that might need citations
            prompt = f"""
            Identify claims in this text that likely need citations but don't appear to have them.
            Look for factual statements, statistics, research findings, or claims about previous work.
            
            Text:
            {content[:2000]}  # Limit text length for API
            
            Provide response as JSON array:
            [
                {{"claim": "specific claim text", "reason": "why it needs citation"}}
            ]
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in academic writing standards. Identify claims that require citations according to academic standards."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            try:
                missing_citations = json.loads(response.choices[0].message.content)
                for missing in missing_citations:
                    issues.append({
                        'type': 'missing_citation',
                        'description': f"Claim may need citation: {missing['claim'][:100]}...",
                        'reason': missing['reason'],
                        'severity': 'medium'
                    })
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.error(f"Error identifying missing citations: {e}")
        
        return issues
    
    async def _verify_citation_accuracy(self, citations: List[Dict]) -> List[Dict]:
        """Verify citation accuracy (simplified)."""
        issues = []
        
        # Basic checks for citation accuracy
        for citation in citations:
            citation_text = citation['content']
            
            # Check for obvious formatting errors
            if citation['type'] == 'parenthetical':
                if not re.search(r'\d{4}', citation_text):
                    issues.append({
                        'type': 'missing_year',
                        'description': f"Citation appears to be missing year: {citation_text}",
                        'severity': 'high'
                    })
                
                # Check for very old citations that might be outdated
                years = re.findall(r'\b(19|20)\d{2}\b', citation_text)
                for year in years:
                    if int(year) < 1990:
                        issues.append({
                            'type': 'outdated_citation',
                            'description': f"Very old citation found: {year}",
                            'severity': 'low'
                        })
        
        return issues
    
    def _check_passive_voice(self, content: str) -> List[Dict]:
        """Check for excessive passive voice usage."""
        issues = []
        
        # Simple passive voice detection
        passive_patterns = [
            r'\b(was|were|being|been)\s+\w+ed\b',
            r'\b(is|are|was|were)\s+\w+en\b',
        ]
        
        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        sentence_count = len(re.split(r'[.!?]+', content))
        
        if sentence_count > 0:
            passive_ratio = passive_count / sentence_count
            
            if passive_ratio > 0.3:  # More than 30% passive voice
                issues.append({
                    'type': 'excessive_passive_voice',
                    'description': f'High passive voice usage: {passive_ratio:.1%} of sentences',
                    'severity': 'medium'
                })
        
        return issues
    
    def _check_first_person(self, content: str) -> List[Dict]:
        """Check for inappropriate first-person usage."""
        issues = []
        
        first_person_patterns = [
            r'\b(I|we|my|our|us|me)\b',
        ]
        
        for pattern in first_person_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append({
                    'type': 'first_person_usage',
                    'description': f'First-person pronouns found: {len(matches)} instances',
                    'severity': 'low'
                })
                break
        
        return issues
    
    def _check_contractions(self, content: str) -> List[Dict]:
        """Check for contractions that should be avoided in academic writing."""
        issues = []
        
        contractions = [
            "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
            "hasn't", "haven't", "doesn't", "didn't", "shouldn't", "wouldn't",
            "couldn't", "mustn't", "needn't", "oughtn't", "mightn't"
        ]
        
        found_contractions = []
        for contraction in contractions:
            if contraction in content.lower():
                found_contractions.append(contraction)
        
        if found_contractions:
            issues.append({
                'type': 'contractions',
                'description': f'Contractions found: {", ".join(found_contractions)}',
                'severity': 'low'
            })
        
        return issues
    
    def _check_informal_language(self, content: str) -> List[Dict]:
        """Check for informal language inappropriate for academic writing."""
        issues = []
        
        informal_words = [
            "really", "very", "pretty", "quite", "rather", "sort of", "kind of",
            "a lot", "lots", "tons", "loads", "huge", "tiny", "massive"
        ]
        
        found_informal = []
        for word in informal_words:
            if re.search(r'\b' + word + r'\b', content, re.IGNORECASE):
                found_informal.append(word)
        
        if found_informal:
            issues.append({
                'type': 'informal_language',
                'description': f'Informal language found: {", ".join(found_informal[:5])}',
                'severity': 'low'
            })
        
        return issues
    
    async def _check_academic_tone(self, content: str) -> List[Dict]:
        """Use AI to check for appropriate academic tone."""
        issues = []
        
        try:
            prompt = f"""
            Analyze this text for academic tone and identify any issues:
            
            Text:
            {content[:1000]}  # Limit for API
            
            Check for:
            - Appropriate academic register
            - Objective tone
            - Formal language use
            - Clarity and precision
            
            Provide response as JSON array:
            [
                {{"type": "tone_issue", "description": "specific issue", "severity": "low/medium/high"}}
            ]
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in academic writing standards. Identify tone and register issues in academic texts."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            try:
                tone_issues = json.loads(response.choices[0].message.content)
                issues.extend(tone_issues)
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            logger.error(f"Error checking academic tone: {e}")
        
        return issues
    
    async def _calculate_overall_score(self, results: Dict) -> Dict:
        """Calculate overall integrity score and provide recommendations."""
        try:
            scores = []
            issues_count = 0
            recommendations = []
            
            # Plagiarism score (inverted - lower plagiarism = higher score)
            if 'plagiarism_score' in results:
                plagiarism_score = 1.0 - results['plagiarism_score']
                scores.append(plagiarism_score)
                
                if results['plagiarism_score'] > 0.3:
                    recommendations.append("Review content for potential plagiarism issues")
                    issues_count += results.get('total_matches', 0)
            
            # Citation score
            if 'citation_score' in results:
                scores.append(results['citation_score'])
                citation_issues = len(results.get('citation_issues', []))
                if citation_issues > 0:
                    recommendations.append("Improve citation formatting and completeness")
                    issues_count += citation_issues
            
            # Style score
            if 'style_score' in results:
                scores.append(results['style_score'])
                style_violations = len(results.get('style_violations', []))
                if style_violations > 0:
                    recommendations.append("Address academic style violations")
                    issues_count += style_violations
            
            # Calculate overall score
            overall_score = sum(scores) / len(scores) if scores else 0.5
            
            # Generate overall recommendation
            if overall_score >= 0.8:
                overall_recommendation = "Content meets high academic integrity standards"
            elif overall_score >= 0.6:
                overall_recommendation = "Content is generally acceptable with minor improvements needed"
            elif overall_score >= 0.4:
                overall_recommendation = "Content requires significant improvements before submission"
            else:
                overall_recommendation = "Content has serious integrity issues that must be addressed"
            
            return {
                'score': overall_score,
                'recommendation': overall_recommendation,
                'specific_recommendations': recommendations,
                'issues_count': issues_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return {
                'score': 0.0,
                'recommendation': 'Unable to calculate integrity score',
                'specific_recommendations': [],
                'issues_count': 0
            }


# Create singleton instance
academic_integrity_service = AcademicIntegrityService()
