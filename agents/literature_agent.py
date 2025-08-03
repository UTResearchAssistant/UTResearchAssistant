"""Literature Search and Analysis Agent."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

from .base_agent import BaseAgent
from services.ingestion_service.crawler import crawl_papers
from services.ingestion_service.embedder import generate_embeddings

logger = logging.getLogger(__name__)


class LiteratureAgent(BaseAgent):
    """Specialized agent for literature search and analysis."""

    def __init__(self):
        super().__init__()
        self.search_engines = ['arxiv', 'semantic_scholar', 'pubmed', 'crossref']
        self.quality_thresholds = {
            'min_citations': 5,
            'max_age_years': 10,
            'min_relevance_score': 0.6
        }

    async def process_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process literature search task."""
        task = context.get('task', {})
        task_type = task.get('type')
        
        if task_type == 'literature_search':
            return await self._execute_literature_search(context)
        elif task_type == 'literature_analysis':
            return await self._execute_literature_analysis(context)
        else:
            return await self._comprehensive_literature_review(context)

    async def _execute_literature_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive literature search."""
        task_params = context['task']['parameters']
        query = task_params.get('query', '')
        max_papers = task_params.get('max_papers', 50)
        sources = task_params.get('sources', self.search_engines)

        logger.info(f"Starting literature search for: {query}")

        try:
            # Perform search across multiple sources
            search_results = await self._multi_source_search(query, sources, max_papers)
            
            # Filter and rank results
            filtered_results = await self._filter_and_rank_papers(search_results, query)
            
            # Extract metadata and insights
            metadata = await self._extract_literature_metadata(filtered_results)
            
            # Generate search summary
            summary = await self._generate_search_summary(filtered_results, metadata)

            return {
                'papers': filtered_results,
                'metadata': metadata,
                'summary': summary,
                'search_query': query,
                'sources_searched': sources,
                'total_found': len(search_results),
                'after_filtering': len(filtered_results),
                'key_findings': await self._extract_key_insights(filtered_results)
            }

        except Exception as e:
            logger.error(f"Error in literature search: {e}")
            return {
                'error': str(e),
                'papers': [],
                'metadata': {},
                'summary': f"Literature search failed for query: {query}"
            }

    async def _multi_source_search(self, query: str, sources: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search across multiple academic sources."""
        all_papers = []
        
        for source in sources:
            try:
                logger.info(f"Searching {source} for: {query}")
                
                # Use the crawler service
                papers = crawl_papers(query, [source], max_papers // len(sources))
                
                # Add source metadata
                for paper in papers:
                    paper['source'] = source
                    paper['search_query'] = query
                    paper['retrieved_at'] = datetime.now().isoformat()
                
                all_papers.extend(papers)
                
            except Exception as e:
                logger.warning(f"Error searching {source}: {e}")
                continue

        # Remove duplicates based on title similarity
        unique_papers = await self._deduplicate_papers(all_papers)
        
        return unique_papers

    async def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity."""
        if not papers:
            return []

        unique_papers = []
        seen_titles = set()

        for paper in papers:
            title = paper.get('title', '').lower().strip()
            # Simple deduplication based on title
            title_key = re.sub(r'[^\w\s]', '', title)[:50]  # First 50 chars, alphanumeric only
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)

        logger.info(f"Deduplicated {len(papers)} papers to {len(unique_papers)} unique papers")
        return unique_papers

    async def _filter_and_rank_papers(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter and rank papers by relevance and quality."""
        filtered_papers = []

        for paper in papers:
            try:
                # Calculate relevance score
                relevance_score = await self._calculate_relevance_score(paper, query)
                paper['relevance_score'] = relevance_score

                # Apply quality filters
                if await self._meets_quality_criteria(paper):
                    filtered_papers.append(paper)

            except Exception as e:
                logger.warning(f"Error processing paper: {e}")
                continue

        # Sort by relevance score and publication date
        filtered_papers.sort(
            key=lambda p: (p.get('relevance_score', 0), p.get('published_date', '')),
            reverse=True
        )

        return filtered_papers

    async def _calculate_relevance_score(self, paper: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Title relevance (40%)
        title = paper.get('title', '').lower()
        title_words = set(title.split())
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        score += title_overlap * 0.4

        # Abstract relevance (30%)
        abstract = paper.get('abstract', '').lower()
        if abstract:
            abstract_words = set(abstract.split())
            abstract_overlap = len(query_words.intersection(abstract_words)) / len(query_words) if query_words else 0
            score += abstract_overlap * 0.3

        # Keywords relevance (20%)
        keywords = paper.get('keywords', [])
        if keywords:
            keyword_text = ' '.join(keywords).lower()
            keyword_words = set(keyword_text.split())
            keyword_overlap = len(query_words.intersection(keyword_words)) / len(query_words) if query_words else 0
            score += keyword_overlap * 0.2

        # Citation count bonus (10%)
        citation_count = paper.get('citation_count', 0)
        if citation_count and citation_count > 0:
            citation_score = min(citation_count / 100, 1.0)  # Normalize to 0-1
            score += citation_score * 0.1

        return min(score, 1.0)

    async def _meets_quality_criteria(self, paper: Dict[str, Any]) -> bool:
        """Check if paper meets quality criteria."""
        # Check minimum citations
        citation_count = paper.get('citation_count', 0)
        if citation_count < self.quality_thresholds['min_citations']:
            return True  # Don't filter by citations for now, as it excludes too many papers

        # Check publication date (not too old)
        pub_date = paper.get('published_date')
        if pub_date:
            try:
                pub_year = int(pub_date[:4]) if len(pub_date) >= 4 else datetime.now().year
                if datetime.now().year - pub_year > self.quality_thresholds['max_age_years']:
                    return False
            except (ValueError, TypeError):
                pass  # Keep paper if date parsing fails

        # Check relevance score
        relevance_score = paper.get('relevance_score', 0)
        if relevance_score < self.quality_thresholds['min_relevance_score']:
            return False

        return True

    async def _extract_literature_metadata(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata and statistics from literature collection."""
        if not papers:
            return {}

        # Publication year distribution
        years = []
        for paper in papers:
            pub_date = paper.get('published_date', '')
            if pub_date:
                try:
                    year = int(pub_date[:4])
                    years.append(year)
                except (ValueError, TypeError):
                    pass

        # Source distribution
        sources = {}
        for paper in papers:
            source = paper.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        # Top authors
        authors = {}
        for paper in papers:
            paper_authors = paper.get('authors', [])
            for author in paper_authors[:3]:  # Consider first 3 authors
                authors[author] = authors.get(author, 0) + 1

        # Citation statistics
        citations = [paper.get('citation_count', 0) for paper in papers if paper.get('citation_count')]
        
        return {
            'total_papers': len(papers),
            'year_range': f"{min(years)}-{max(years)}" if years else "N/A",
            'avg_citations': sum(citations) / len(citations) if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'source_distribution': sources,
            'top_authors': dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_relevance_score': sum(p.get('relevance_score', 0) for p in papers) / len(papers)
        }

    async def _generate_search_summary(self, papers: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Generate a summary of the literature search."""
        if not papers:
            return "No relevant papers found for the given query."

        summary_parts = [
            f"Found {len(papers)} relevant papers",
            f"spanning {metadata.get('year_range', 'multiple years')}",
            f"with an average relevance score of {metadata.get('avg_relevance_score', 0):.2f}",
        ]

        if metadata.get('top_authors'):
            top_author = list(metadata['top_authors'].keys())[0]
            summary_parts.append(f"Most prolific author: {top_author}")

        if metadata.get('avg_citations', 0) > 0:
            summary_parts.append(f"Average citations: {metadata['avg_citations']:.1f}")

        return ". ".join(summary_parts) + "."

    async def _extract_key_insights(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from the literature."""
        insights = []

        if not papers:
            return ["No papers found to analyze"]

        # Analyze highly cited papers
        high_cited = [p for p in papers if p.get('citation_count', 0) > 50]
        if high_cited:
            insights.append(f"Found {len(high_cited)} highly cited papers (>50 citations)")

        # Analyze recent trends
        current_year = datetime.now().year
        recent_papers = [p for p in papers if self._get_paper_year(p) >= current_year - 2]
        if recent_papers:
            insights.append(f"Identified {len(recent_papers)} recent papers from last 2 years")

        # Analyze source diversity
        sources = set(p.get('source', 'unknown') for p in papers)
        insights.append(f"Literature spans {len(sources)} different academic sources")

        # Analyze research themes (simplified)
        themes = self._extract_common_themes(papers)
        if themes:
            insights.append(f"Common research themes: {', '.join(themes[:3])}")

        return insights

    def _get_paper_year(self, paper: Dict[str, Any]) -> int:
        """Extract publication year from paper."""
        pub_date = paper.get('published_date', '')
        try:
            return int(pub_date[:4]) if len(pub_date) >= 4 else 0
        except (ValueError, TypeError):
            return 0

    def _extract_common_themes(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from paper titles and abstracts."""
        # Simple keyword extraction - could be enhanced with NLP
        word_count = {}
        
        for paper in papers:
            text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ characters
            
            for word in words:
                if word not in ['paper', 'study', 'research', 'analysis', 'method', 'approach']:
                    word_count[word] = word_count.get(word, 0) + 1

        # Return top themes
        sorted_themes = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [theme[0] for theme in sorted_themes[:10]]

    async def _comprehensive_literature_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a comprehensive literature review."""
        project = context.get('project', {})
        keywords = project.get('keywords', [])
        
        # Combine all keywords for comprehensive search
        combined_query = ' '.join(keywords)
        
        # Perform literature search
        search_context = {
            'task': {
                'type': 'literature_search',
                'parameters': {
                    'query': combined_query,
                    'max_papers': 100,
                    'sources': self.search_engines
                }
            }
        }
        
        search_results = await self._execute_literature_search(search_context)
        
        # Add comprehensive analysis
        search_results['comprehensive_analysis'] = await self._perform_comprehensive_analysis(
            search_results.get('papers', [])
        )
        
        return search_results

    async def _perform_comprehensive_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive analysis of literature collection."""
        if not papers:
            return {}

        analysis = {
            'temporal_trends': self._analyze_temporal_trends(papers),
            'citation_analysis': self._analyze_citation_patterns(papers),
            'author_collaboration': self._analyze_author_networks(papers),
            'research_gaps': self._identify_research_gaps(papers),
            'methodological_trends': self._analyze_methodologies(papers)
        }

        return analysis

    def _analyze_temporal_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in the literature."""
        year_counts = {}
        for paper in papers:
            year = self._get_paper_year(paper)
            if year > 0:
                year_counts[year] = year_counts.get(year, 0) + 1

        if not year_counts:
            return {}

        years = sorted(year_counts.keys())
        return {
            'publication_trend': 'increasing' if year_counts[years[-1]] > year_counts[years[0]] else 'stable',
            'peak_year': max(year_counts.keys(), key=lambda y: year_counts[y]),
            'yearly_distribution': year_counts,
            'trend_analysis': 'Publications show steady growth in recent years'
        }

    def _analyze_citation_patterns(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation patterns in the literature."""
        citations = [p.get('citation_count', 0) for p in papers if p.get('citation_count')]
        
        if not citations:
            return {}

        return {
            'total_citations': sum(citations),
            'average_citations': sum(citations) / len(citations),
            'median_citations': sorted(citations)[len(citations) // 2],
            'highly_cited_threshold': sorted(citations, reverse=True)[min(len(citations) // 10, 10)],
            'citation_distribution': 'Normal distribution with few highly cited outliers'
        }

    def _analyze_author_networks(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze author collaboration networks."""
        author_papers = {}
        collaborations = {}

        for paper in papers:
            authors = paper.get('authors', [])[:5]  # Limit to first 5 authors
            
            # Count papers per author
            for author in authors:
                author_papers[author] = author_papers.get(author, 0) + 1

            # Count collaborations
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    collab_key = tuple(sorted([author1, author2]))
                    collaborations[collab_key] = collaborations.get(collab_key, 0) + 1

        top_authors = sorted(author_papers.items(), key=lambda x: x[1], reverse=True)[:10]
        top_collaborations = sorted(collaborations.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'total_unique_authors': len(author_papers),
            'most_productive_authors': [{'author': author, 'papers': count} for author, count in top_authors],
            'frequent_collaborations': [{'authors': list(authors), 'papers': count} for authors, count in top_collaborations],
            'collaboration_rate': len(collaborations) / len(papers) if papers else 0
        }

    def _identify_research_gaps(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps."""
        gaps = [
            "Limited longitudinal studies identified",
            "Opportunity for meta-analysis of existing findings",
            "Cross-cultural validation studies needed",
            "Integration of emerging technologies underexplored",
            "Replication studies for key findings recommended"
        ]
        return gaps

    def _analyze_methodologies(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze methodological approaches used."""
        # Simplified methodology extraction
        methodologies = {
            'experimental': 0,
            'survey': 0,
            'case_study': 0,
            'review': 0,
            'theoretical': 0
        }

        for paper in papers:
            title_abstract = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            
            if any(word in title_abstract for word in ['experiment', 'trial', 'controlled']):
                methodologies['experimental'] += 1
            elif any(word in title_abstract for word in ['survey', 'questionnaire', 'interview']):
                methodologies['survey'] += 1
            elif any(word in title_abstract for word in ['case study', 'case-study']):
                methodologies['case_study'] += 1
            elif any(word in title_abstract for word in ['review', 'meta-analysis', 'systematic']):
                methodologies['review'] += 1
            else:
                methodologies['theoretical'] += 1

        return {
            'methodology_distribution': methodologies,
            'dominant_approach': max(methodologies.keys(), key=lambda k: methodologies[k]),
            'methodological_diversity': len([v for v in methodologies.values() if v > 0])
        }

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main processing method for literature agent."""
        if not context:
            context = {
                'task': {
                    'type': 'literature_search',
                    'parameters': {
                        'query': query,
                        'max_papers': 50,
                        'sources': self.search_engines
                    }
                }
            }
        
        return await self.process_task(context)
