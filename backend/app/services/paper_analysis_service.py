"""
Advanced Paper Analysis Service - AI-powered research paper understanding
Provides comprehensive analysis including summarization, contribution extraction,
and impact prediction using NLP and ML techniques
"""

import re
import spacy
import numpy as np
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PaperAnalysisService:
    """
    Advanced service for analyzing and understanding research papers
    using natural language processing and machine learning techniques
    """
    
    def __init__(self):
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize sentence transformers for embeddings (optional)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")
            self.embedding_model = None
    
    async def analyze_paper(self, paper_data: Dict) -> Dict:
        """
        Comprehensive analysis of a research paper
        
        Args:
            paper_data: Dictionary containing paper information
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            analysis = {
                "summary": await self.generate_summary(paper_data),
                "key_contributions": await self.extract_contributions(paper_data),
                "methodology": await self.extract_methodology(paper_data),
                "results": await self.extract_results(paper_data),
                "keywords": await self.extract_keywords(paper_data),
                "novelty_score": await self.calculate_novelty_score(paper_data),
                "impact_prediction": await self.predict_impact(paper_data),
                "research_fields": await self.classify_research_fields(paper_data),
                "technical_depth": await self.assess_technical_depth(paper_data),
                "readability_score": await self.calculate_readability(paper_data)
            }
            return analysis
        except Exception as e:
            logger.error(f"Paper analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_summary(self, paper_data: Dict) -> str:
        """Generate a concise summary of the paper"""
        try:
            abstract = paper_data.get("abstract", "")
            title = paper_data.get("title", "")
            
            if not abstract:
                return f"Summary not available. Title: {title}"
            
            # Simple extractive summarization
            sentences = self._split_into_sentences(abstract)
            
            if len(sentences) <= 3:
                return abstract
            
            # Score sentences by position and keyword frequency
            scored_sentences = []
            keywords = self._extract_important_terms(abstract)
            
            for i, sentence in enumerate(sentences):
                score = 0
                # Position score (first and last sentences are important)
                if i == 0 or i == len(sentences) - 1:
                    score += 2
                
                # Keyword score
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        score += 1
                
                # Length penalty for very short sentences
                if len(sentence.split()) < 5:
                    score -= 1
                
                scored_sentences.append((score, sentence))
            
            # Select top sentences
            scored_sentences.sort(reverse=True)
            summary_sentences = [sent for _, sent in scored_sentences[:3]]
            
            return " ".join(summary_sentences)
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Summary generation failed"
    
    async def extract_contributions(self, paper_data: Dict) -> List[str]:
        """Extract key contributions from the paper"""
        try:
            abstract = paper_data.get("abstract", "")
            title = paper_data.get("title", "")
            
            contributions = []
            
            # Look for contribution indicators
            contribution_patterns = [
                r"we propose", r"we present", r"we introduce", r"we develop",
                r"our contribution", r"our approach", r"we show", r"we demonstrate",
                r"novel", r"new", r"first time", r"state-of-the-art",
                r"breakthrough", r"innovative", r"pioneering"
            ]
            
            text = f"{title} {abstract}".lower()
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                for pattern in contribution_patterns:
                    if re.search(pattern, sentence.lower()):
                        # Clean up the sentence
                        clean_sentence = sentence.strip().capitalize()
                        if len(clean_sentence) > 20:  # Filter out very short matches
                            contributions.append(clean_sentence)
                        break
            
            # Remove duplicates and limit to top 5
            unique_contributions = list(dict.fromkeys(contributions))
            return unique_contributions[:5]
        except Exception as e:
            logger.error(f"Contribution extraction error: {e}")
            return []
    
    async def extract_methodology(self, paper_data: Dict) -> str:
        """Extract methodology information from the paper"""
        try:
            abstract = paper_data.get("abstract", "")
            
            # Look for methodology indicators
            method_patterns = [
                r"method", r"approach", r"algorithm", r"technique", r"framework",
                r"model", r"system", r"experiment", r"evaluation", r"analysis",
                r"implementation", r"design", r"architecture", r"procedure"
            ]
            
            sentences = self._split_into_sentences(abstract)
            method_sentences = []
            
            for sentence in sentences:
                for pattern in method_patterns:
                    if re.search(pattern, sentence.lower()):
                        method_sentences.append(sentence.strip())
                        break
            
            return " ".join(method_sentences[:3])
        except Exception as e:
            logger.error(f"Methodology extraction error: {e}")
            return "Methodology extraction failed"
    
    async def extract_results(self, paper_data: Dict) -> str:
        """Extract results information from the paper"""
        try:
            abstract = paper_data.get("abstract", "")
            
            # Look for results indicators
            result_patterns = [
                r"result", r"finding", r"achieve", r"performance", r"improvement",
                r"accuracy", r"precision", r"recall", r"f1", r"score", r"metric",
                r"outperform", r"superior", r"significant", r"effectiveness"
            ]
            
            sentences = self._split_into_sentences(abstract)
            result_sentences = []
            
            for sentence in sentences:
                for pattern in result_patterns:
                    if re.search(pattern, sentence.lower()):
                        result_sentences.append(sentence.strip())
                        break
            
            return " ".join(result_sentences[:3])
        except Exception as e:
            logger.error(f"Results extraction error: {e}")
            return "Results extraction failed"
    
    async def extract_keywords(self, paper_data: Dict) -> List[str]:
        """Extract important keywords from the paper"""
        try:
            text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
            
            # Use existing keywords if available
            existing_keywords = paper_data.get("keywords", [])
            if existing_keywords:
                return existing_keywords
            
            # Extract keywords using NLP
            if self.nlp:
                doc = self.nlp(text)
                
                # Extract noun phrases and named entities
                keywords = set()
                
                # Named entities
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "PRODUCT", "EVENT", "LAW"]:
                        if len(ent.text.split()) <= 3:  # Limit to 3 words
                            keywords.add(ent.text.lower())
                
                # Noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Limit to 3 words
                        chunk_text = chunk.text.lower().strip()
                        if len(chunk_text) > 3:  # Filter very short phrases
                            keywords.add(chunk_text)
                
                # Technical terms (words ending in common technical suffixes)
                tech_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ing']
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ'] and 
                        len(token.text) > 5 and 
                        any(token.text.lower().endswith(suffix) for suffix in tech_suffixes)):
                        keywords.add(token.text.lower())
                
                return list(keywords)[:15]
            else:
                # Simple keyword extraction
                return self._extract_important_terms(text)
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    async def calculate_novelty_score(self, paper_data: Dict) -> float:
        """Calculate a novelty score for the paper"""
        try:
            title = paper_data.get("title", "").lower()
            abstract = paper_data.get("abstract", "").lower()
            
            # Novelty indicators
            novelty_terms = [
                "novel", "new", "first", "innovative", "breakthrough", "pioneering",
                "unprecedented", "original", "unique", "cutting-edge", "state-of-the-art",
                "revolutionary", "groundbreaking", "emerging", "advanced"
            ]
            
            text = f"{title} {abstract}"
            score = 0
            
            for term in novelty_terms:
                # Count occurrences with different weights
                title_count = title.count(term) * 2  # Title mentions are more important
                abstract_count = abstract.count(term)
                score += title_count + abstract_count
            
            # Publication recency boost
            pub_date = paper_data.get("publication_date")
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        year = int(pub_date.split('-')[0])
                    else:
                        year = pub_date.year
                    
                    current_year = datetime.now().year
                    recency_boost = max(0, (year - (current_year - 5)) / 5)  # Boost for recent papers
                    score += recency_boost
                except:
                    pass
            
            # Normalize score (0-1 range)
            max_score = len(novelty_terms) * 3 + 1  # Max possible score
            normalized_score = min(score / max_score, 1.0)
            
            return normalized_score
        except Exception as e:
            logger.error(f"Novelty score calculation error: {e}")
            return 0.0
    
    async def predict_impact(self, paper_data: Dict) -> Dict:
        """Predict potential impact of the paper"""
        try:
            # Multi-factor impact prediction
            venue = paper_data.get("journal", "").lower()
            authors = paper_data.get("authors", [])
            keywords = await self.extract_keywords(paper_data)
            citation_count = paper_data.get("citation_count", 0)
            
            impact_score = 0
            factors = []
            
            # Venue quality (simplified)
            high_impact_venues = [
                "nature", "science", "cell", "lancet", "nejm",
                "nips", "icml", "iclr", "acl", "cvpr", "iccv", "eccv",
                "jmlr", "pami", "tochi", "tacl", "emnlp"
            ]
            
            if any(venue_name in venue for venue_name in high_impact_venues):
                impact_score += 0.3
                factors.append("High-impact venue")
            
            # Citation count boost
            if citation_count > 100:
                impact_score += 0.25
                factors.append("High citation count")
            elif citation_count > 50:
                impact_score += 0.15
                factors.append("Good citation count")
            
            # Number of authors (collaboration indicator)
            if len(authors) >= 5:
                impact_score += 0.1
                factors.append("Large collaboration")
            elif len(authors) >= 3:
                impact_score += 0.05
                factors.append("Multi-author collaboration")
            
            # Hot keywords and trending topics
            hot_keywords = [
                "ai", "artificial intelligence", "machine learning", "deep learning", 
                "nlp", "natural language processing", "computer vision", "robotics",
                "blockchain", "quantum", "climate", "covid", "sustainability",
                "biomedical", "genomics", "neuroscience", "social media"
            ]
            
            keyword_text = " ".join(keywords).lower()
            hot_matches = sum(1 for keyword in hot_keywords if keyword in keyword_text)
            if hot_matches > 0:
                impact_score += min(hot_matches * 0.05, 0.2)
                factors.append("Trending research area")
            
            # Novelty contribution
            novelty_score = await self.calculate_novelty_score(paper_data)
            impact_score += novelty_score * 0.3
            if novelty_score > 0.5:
                factors.append("High novelty")
            
            # Technical depth assessment
            tech_depth = await self.assess_technical_depth(paper_data)
            impact_score += tech_depth * 0.1
            if tech_depth > 0.7:
                factors.append("High technical depth")
            
            return {
                "predicted_impact_score": min(impact_score, 1.0),
                "contributing_factors": factors,
                "confidence": "medium",  # Could be improved with more data
                "explanation": f"Score based on {len(factors)} contributing factors"
            }
        except Exception as e:
            logger.error(f"Impact prediction error: {e}")
            return {"predicted_impact_score": 0.0, "contributing_factors": [], "confidence": "low"}
    
    async def classify_research_fields(self, paper_data: Dict) -> List[Dict]:
        """Classify the paper into research fields"""
        try:
            text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
            keywords = await self.extract_keywords(paper_data)
            
            # Research field keywords mapping
            field_keywords = {
                "Computer Science": ["algorithm", "computer", "software", "programming", "data structure"],
                "Machine Learning": ["machine learning", "neural network", "deep learning", "ai", "artificial intelligence"],
                "Natural Language Processing": ["nlp", "language", "text", "linguistic", "parsing", "sentiment"],
                "Computer Vision": ["vision", "image", "visual", "object detection", "recognition", "segmentation"],
                "Biomedical": ["medical", "clinical", "patient", "disease", "treatment", "biological"],
                "Physics": ["quantum", "particle", "energy", "physics", "theoretical", "experimental"],
                "Mathematics": ["mathematical", "theorem", "proof", "optimization", "statistics", "probability"],
                "Social Sciences": ["social", "psychology", "behavior", "society", "human", "cultural"],
                "Engineering": ["engineering", "design", "system", "control", "optimization", "technical"],
                "Environmental": ["environmental", "climate", "sustainability", "green", "ecosystem", "carbon"]
            }
            
            field_scores = []
            text_lower = text.lower()
            keywords_text = " ".join(keywords).lower()
            
            for field, field_kws in field_keywords.items():
                score = 0
                matched_keywords = []
                
                for kw in field_kws:
                    if kw in text_lower or kw in keywords_text:
                        score += 1
                        matched_keywords.append(kw)
                
                if score > 0:
                    confidence = min(score / len(field_kws), 1.0)
                    field_scores.append({
                        "field": field,
                        "confidence": confidence,
                        "matched_keywords": matched_keywords
                    })
            
            # Sort by confidence
            field_scores.sort(key=lambda x: x["confidence"], reverse=True)
            return field_scores[:3]  # Top 3 fields
            
        except Exception as e:
            logger.error(f"Research field classification error: {e}")
            return []
    
    async def assess_technical_depth(self, paper_data: Dict) -> float:
        """Assess the technical depth of the paper"""
        try:
            text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
            
            # Technical indicators
            technical_terms = [
                "algorithm", "implementation", "optimization", "complexity",
                "evaluation", "experimental", "theoretical", "empirical",
                "quantitative", "qualitative", "statistical", "mathematical",
                "model", "framework", "architecture", "methodology"
            ]
            
            math_indicators = ["equation", "formula", "theorem", "proof", "lemma"]
            experimental_indicators = ["experiment", "dataset", "benchmark", "evaluation", "results"]
            
            text_lower = text.lower()
            
            tech_score = sum(1 for term in technical_terms if term in text_lower)
            math_score = sum(1 for term in math_indicators if term in text_lower) * 1.5
            exp_score = sum(1 for term in experimental_indicators if term in text_lower) * 1.2
            
            total_score = tech_score + math_score + exp_score
            
            # Normalize (rough estimation)
            max_possible = len(technical_terms) + len(math_indicators) * 1.5 + len(experimental_indicators) * 1.2
            normalized_score = min(total_score / max_possible, 1.0)
            
            return normalized_score
        except Exception as e:
            logger.error(f"Technical depth assessment error: {e}")
            return 0.0
    
    async def calculate_readability(self, paper_data: Dict) -> Dict:
        """Calculate readability metrics for the paper"""
        try:
            abstract = paper_data.get("abstract", "")
            if not abstract:
                return {"error": "No abstract available"}
            
            sentences = self._split_into_sentences(abstract)
            words = abstract.split()
            
            # Basic readability metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Flesch Reading Ease Score
            if avg_sentence_length > 0:
                flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
                flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
            else:
                flesch_score = 0
            
            # Determine reading level
            if flesch_score >= 90:
                level = "Very Easy"
            elif flesch_score >= 80:
                level = "Easy"
            elif flesch_score >= 70:
                level = "Fairly Easy"
            elif flesch_score >= 60:
                level = "Standard"
            elif flesch_score >= 50:
                level = "Fairly Difficult"
            elif flesch_score >= 30:
                level = "Difficult"
            else:
                level = "Very Difficult"
            
            return {
                "flesch_score": flesch_score,
                "reading_level": level,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "total_words": len(words),
                "total_sentences": len(sentences)
            }
        except Exception as e:
            logger.error(f"Readability calculation error: {e}")
            return {"error": str(e)}
    
    async def compare_papers(self, papers: List[Dict]) -> Dict:
        """Compare multiple papers and generate a comparison analysis"""
        try:
            if len(papers) < 2:
                return {"error": "Need at least 2 papers for comparison"}
            
            comparison = {
                "papers": [],
                "common_themes": [],
                "key_differences": [],
                "methodology_comparison": {},
                "impact_ranking": [],
                "novelty_ranking": [],
                "field_distribution": {}
            }
            
            # Analyze each paper
            analyzed_papers = []
            for paper in papers:
                analysis = await self.analyze_paper(paper)
                paper_info = {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": self._extract_year_from_date(paper.get("publication_date")),
                    "citation_count": paper.get("citation_count", 0),
                    "analysis": analysis
                }
                analyzed_papers.append(paper_info)
            
            comparison["papers"] = analyzed_papers
            
            # Find common themes
            all_keywords = []
            for paper in analyzed_papers:
                all_keywords.extend(paper["analysis"].get("keywords", []))
            
            keyword_counts = Counter(all_keywords)
            common_keywords = [kw for kw, count in keyword_counts.items() if count >= 2]
            comparison["common_themes"] = common_keywords[:10]
            
            # Rank by predicted impact
            impact_ranking = sorted(
                analyzed_papers, 
                key=lambda x: x["analysis"].get("impact_prediction", {}).get("predicted_impact_score", 0), 
                reverse=True
            )
            comparison["impact_ranking"] = [p["title"] for p in impact_ranking]
            
            # Rank by novelty
            novelty_ranking = sorted(
                analyzed_papers,
                key=lambda x: x["analysis"].get("novelty_score", 0),
                reverse=True
            )
            comparison["novelty_ranking"] = [p["title"] for p in novelty_ranking]
            
            # Field distribution
            all_fields = []
            for paper in analyzed_papers:
                fields = paper["analysis"].get("research_fields", [])
                all_fields.extend([field["field"] for field in fields])
            
            field_counts = Counter(all_fields)
            comparison["field_distribution"] = dict(field_counts)
            
            return comparison
        except Exception as e:
            logger.error(f"Paper comparison error: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_important_terms(self, text: str) -> List[str]:
        """Extract important terms using simple frequency analysis"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'paper', 'research', 'study', 'work', 'approach', 'method', 'results'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_freq = Counter(word for word in words if word not in stop_words and len(word) > 3)
        
        return [word for word, count in word_freq.most_common(15)]
    
    def _extract_year_from_date(self, date_str) -> Optional[str]:
        """Extract year from date string"""
        if not date_str:
            return None
        
        try:
            if isinstance(date_str, datetime):
                return str(date_str.year)
            elif isinstance(date_str, str):
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    return year_match.group()
        except:
            pass
        
        return None


# Create global instance
paper_analysis_service = PaperAnalysisService()
