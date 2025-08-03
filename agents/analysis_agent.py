"""Content Analysis Agent for research data."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
from collections import Counter

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """Specialized agent for content analysis and data processing."""

    def __init__(self):
        super().__init__()
        self.analysis_types = {
            'thematic': self._thematic_analysis,
            'statistical': self._statistical_analysis,
            'sentiment': self._sentiment_analysis,
            'trend': self._trend_analysis,
            'network': self._network_analysis,
            'content': self._content_analysis
        }

    async def process_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis task."""
        task = context.get('task', {})
        task_type = task.get('type')
        
        if task_type == 'content_analysis':
            return await self._execute_content_analysis(context)
        elif task_type == 'thematic_analysis':
            return await self._execute_thematic_analysis(context)
        elif task_type == 'statistical_analysis':
            return await self._execute_statistical_analysis(context)
        else:
            return await self._comprehensive_analysis(context)

    async def _execute_content_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive content analysis."""
        task_params = context['task']['parameters']
        analysis_types = task_params.get('analysis_types', ['thematic', 'statistical', 'trend'])
        
        # Get literature data from previous results
        previous_results = context.get('previous_results', {})
        papers = self._extract_papers_from_results(previous_results)

        logger.info(f"Starting content analysis on {len(papers)} papers")

        if not papers:
            return {
                'error': 'No papers found for analysis',
                'analysis_results': {},
                'summary': 'Content analysis could not be performed due to lack of data'
            }

        # Perform multiple types of analysis
        analysis_results = {}
        
        for analysis_type in analysis_types:
            if analysis_type in self.analysis_types:
                try:
                    logger.info(f"Performing {analysis_type} analysis")
                    result = await self.analysis_types[analysis_type](papers)
                    analysis_results[analysis_type] = result
                except Exception as e:
                    logger.error(f"Error in {analysis_type} analysis: {e}")
                    analysis_results[analysis_type] = {'error': str(e)}

        # Generate comprehensive insights
        insights = await self._generate_analysis_insights(analysis_results, papers)
        
        # Create summary
        summary = await self._create_analysis_summary(analysis_results, insights)

        return {
            'analysis_results': analysis_results,
            'insights': insights,
            'summary': summary,
            'papers_analyzed': len(papers),
            'analysis_types_performed': list(analysis_results.keys()),
            'key_findings': insights.get('key_findings', [])
        }

    def _extract_papers_from_results(self, previous_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract papers from previous task results."""
        all_papers = []
        
        for task_id, result in previous_results.items():
            if isinstance(result, dict) and 'result' in result:
                task_result = result['result']
                if 'papers' in task_result:
                    all_papers.extend(task_result['papers'])

        return all_papers

    async def _thematic_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform thematic analysis on papers."""
        themes = {}
        keywords_count = Counter()
        abstract_themes = Counter()

        for paper in papers:
            # Analyze title for themes
            title = paper.get('title', '').lower()
            title_themes = self._extract_themes_from_text(title)
            for theme in title_themes:
                themes[theme] = themes.get(theme, 0) + 1

            # Analyze keywords
            keywords = paper.get('keywords', [])
            if keywords:
                keywords_count.update([kw.lower().strip() for kw in keywords])

            # Analyze abstract for themes
            abstract = paper.get('abstract', '').lower()
            if abstract:
                abstract_themes.update(self._extract_themes_from_text(abstract))

        # Identify dominant themes
        dominant_themes = dict(themes.most_common(10))
        top_keywords = dict(keywords_count.most_common(15))
        emerging_themes = dict(abstract_themes.most_common(20))

        # Categorize themes
        theme_categories = self._categorize_themes(dominant_themes.keys())

        return {
            'dominant_themes': dominant_themes,
            'top_keywords': top_keywords,
            'emerging_themes': emerging_themes,
            'theme_categories': theme_categories,
            'thematic_diversity': len(set(themes.keys())),
            'analysis_summary': f"Identified {len(dominant_themes)} dominant themes across {len(papers)} papers"
        }

    async def _statistical_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on paper metadata."""
        if not papers:
            return {}

        # Publication year analysis
        years = [self._extract_year(paper.get('published_date', '')) for paper in papers]
        years = [y for y in years if y > 0]

        # Citation analysis
        citations = [paper.get('citation_count', 0) for paper in papers if paper.get('citation_count') is not None]

        # Author analysis
        author_counts = []
        all_authors = []
        for paper in papers:
            authors = paper.get('authors', [])
            if authors:
                author_counts.append(len(authors))
                all_authors.extend(authors)

        # Source analysis
        sources = [paper.get('source', 'unknown') for paper in papers]
        source_distribution = Counter(sources)

        # Relevance score analysis
        relevance_scores = [paper.get('relevance_score', 0) for paper in papers if paper.get('relevance_score') is not None]

        stats = {
            'total_papers': len(papers),
            'year_statistics': self._calculate_year_stats(years),
            'citation_statistics': self._calculate_citation_stats(citations),
            'author_statistics': self._calculate_author_stats(author_counts, all_authors),
            'source_distribution': dict(source_distribution),
            'relevance_statistics': self._calculate_relevance_stats(relevance_scores),
            'temporal_coverage': f"{min(years)}-{max(years)}" if years else "N/A"
        }

        return stats

    async def _sentiment_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform sentiment analysis on paper abstracts and titles."""
        # Simplified sentiment analysis - could be enhanced with NLP models
        positive_words = ['improved', 'enhanced', 'efficient', 'effective', 'novel', 'innovative', 'breakthrough', 'significant', 'promising']
        negative_words = ['limited', 'challenging', 'difficult', 'problematic', 'inadequate', 'insufficient', 'complex']
        neutral_words = ['analysis', 'study', 'research', 'investigation', 'examination', 'evaluation']

        sentiment_scores = []
        overall_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}

        for paper in papers:
            text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            
            pos_count = sum(text.count(word) for word in positive_words)
            neg_count = sum(text.count(word) for word in negative_words)
            neu_count = sum(text.count(word) for word in neutral_words)

            total_sentiment_words = pos_count + neg_count + neu_count
            
            if total_sentiment_words > 0:
                pos_ratio = pos_count / total_sentiment_words
                neg_ratio = neg_count / total_sentiment_words
                neu_ratio = neu_count / total_sentiment_words
                
                sentiment_scores.append({
                    'paper_id': paper.get('id', f"paper_{len(sentiment_scores)}"),
                    'positive': pos_ratio,
                    'negative': neg_ratio,
                    'neutral': neu_ratio
                })

                # Determine dominant sentiment
                if pos_ratio > neg_ratio and pos_ratio > neu_ratio:
                    overall_sentiment['positive'] += 1
                elif neg_ratio > pos_ratio and neg_ratio > neu_ratio:
                    overall_sentiment['negative'] += 1
                else:
                    overall_sentiment['neutral'] += 1

        return {
            'individual_sentiments': sentiment_scores,
            'overall_distribution': overall_sentiment,
            'sentiment_summary': f"Most papers show neutral to positive sentiment with {overall_sentiment['positive']} positive, {overall_sentiment['neutral']} neutral, and {overall_sentiment['negative']} negative papers",
            'analysis_method': 'Keyword-based sentiment analysis'
        }

    async def _trend_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the research papers."""
        # Temporal trends
        year_themes = {}
        year_keywords = {}

        for paper in papers:
            year = self._extract_year(paper.get('published_date', ''))
            if year > 0:
                # Collect themes by year
                title_abstract = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
                themes = self._extract_themes_from_text(title_abstract)
                
                if year not in year_themes:
                    year_themes[year] = Counter()
                year_themes[year].update(themes)

                # Collect keywords by year
                keywords = paper.get('keywords', [])
                if keywords:
                    if year not in year_keywords:
                        year_keywords[year] = Counter()
                    year_keywords[year].update([kw.lower().strip() for kw in keywords])

        # Identify emerging and declining trends
        emerging_trends = self._identify_emerging_trends(year_themes)
        declining_trends = self._identify_declining_trends(year_themes)
        
        # Analyze publication trends
        publication_trends = self._analyze_publication_trends(papers)

        return {
            'emerging_themes': emerging_trends,
            'declining_themes': declining_trends,
            'publication_trends': publication_trends,
            'temporal_theme_evolution': self._summarize_theme_evolution(year_themes),
            'trend_analysis_summary': f"Identified {len(emerging_trends)} emerging and {len(declining_trends)} declining research themes"
        }

    async def _network_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collaboration and citation networks."""
        # Author collaboration network
        collaborations = {}
        author_productivity = Counter()

        for paper in papers:
            authors = paper.get('authors', [])[:5]  # Limit to first 5 authors
            
            # Count author productivity
            author_productivity.update(authors)

            # Build collaboration network
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    edge = tuple(sorted([author1, author2]))
                    collaborations[edge] = collaborations.get(edge, 0) + 1

        # Keyword co-occurrence network
        keyword_cooccurrence = {}
        for paper in papers:
            keywords = paper.get('keywords', [])[:5]  # Limit to first 5 keywords
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    edge = tuple(sorted([kw1.lower(), kw2.lower()]))
                    keyword_cooccurrence[edge] = keyword_cooccurrence.get(edge, 0) + 1

        return {
            'collaboration_network': {
                'total_collaborations': len(collaborations),
                'top_collaborations': dict(Counter(collaborations).most_common(10)),
                'most_collaborative_authors': dict(author_productivity.most_common(10))
            },
            'keyword_network': {
                'total_cooccurrences': len(keyword_cooccurrence),
                'top_keyword_pairs': dict(Counter(keyword_cooccurrence).most_common(15)),
                'network_density': len(keyword_cooccurrence) / (len(papers) ** 2) if papers else 0
            },
            'network_summary': f"Analyzed {len(collaborations)} author collaborations and {len(keyword_cooccurrence)} keyword co-occurrences"
        }

    async def _content_analysis(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive content analysis."""
        # Text length analysis
        title_lengths = [len(paper.get('title', '')) for paper in papers]
        abstract_lengths = [len(paper.get('abstract', '')) for paper in papers if paper.get('abstract')]

        # Vocabulary analysis
        all_text = []
        for paper in papers:
            text = paper.get('title', '') + ' ' + paper.get('abstract', '')
            all_text.append(text.lower())

        vocabulary = set()
        word_frequencies = Counter()
        
        for text in all_text:
            words = re.findall(r'\b\w{3,}\b', text)  # Words with 3+ characters
            vocabulary.update(words)
            word_frequencies.update(words)

        # Content complexity analysis
        avg_words_per_title = sum(len(title.split()) for title in [p.get('title', '') for p in papers]) / len(papers)
        avg_words_per_abstract = sum(len(abstract.split()) for abstract in [p.get('abstract', '') for p in papers if p.get('abstract')]) / len([p for p in papers if p.get('abstract')])

        return {
            'text_statistics': {
                'avg_title_length': sum(title_lengths) / len(title_lengths) if title_lengths else 0,
                'avg_abstract_length': sum(abstract_lengths) / len(abstract_lengths) if abstract_lengths else 0,
                'avg_words_per_title': avg_words_per_title,
                'avg_words_per_abstract': avg_words_per_abstract
            },
            'vocabulary_analysis': {
                'unique_vocabulary_size': len(vocabulary),
                'most_frequent_words': dict(word_frequencies.most_common(20)),
                'vocabulary_richness': len(vocabulary) / sum(word_frequencies.values()) if word_frequencies else 0
            },
            'content_summary': f"Analyzed {len(papers)} papers with {len(vocabulary)} unique terms in vocabulary"
        }

    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from text using keyword matching."""
        # Common research themes
        theme_keywords = {
            'machine_learning': ['machine learning', 'ml', 'artificial intelligence', 'ai', 'neural network'],
            'data_science': ['data science', 'big data', 'analytics', 'data mining'],
            'deep_learning': ['deep learning', 'neural network', 'cnn', 'rnn', 'transformer'],
            'nlp': ['natural language processing', 'nlp', 'text mining', 'sentiment analysis'],
            'computer_vision': ['computer vision', 'image processing', 'object detection'],
            'robotics': ['robotics', 'robot', 'autonomous', 'automation'],
            'healthcare': ['healthcare', 'medical', 'clinical', 'diagnosis', 'treatment'],
            'education': ['education', 'learning', 'teaching', 'pedagogical'],
            'sustainability': ['sustainability', 'environment', 'green', 'renewable'],
            'security': ['security', 'privacy', 'encryption', 'cybersecurity']
        }

        found_themes = []
        text_lower = text.lower()

        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_themes.append(theme)

        return found_themes

    def _categorize_themes(self, themes: List[str]) -> Dict[str, List[str]]:
        """Categorize themes into broader categories."""
        categories = {
            'Technology': ['machine_learning', 'deep_learning', 'nlp', 'computer_vision', 'robotics'],
            'Data & Analytics': ['data_science', 'analytics', 'data_mining'],
            'Applications': ['healthcare', 'education', 'sustainability'],
            'Infrastructure': ['security', 'privacy', 'cybersecurity']
        }

        categorized = {cat: [] for cat in categories.keys()}
        uncategorized = []

        for theme in themes:
            categorized_theme = False
            for category, theme_list in categories.items():
                if theme in theme_list:
                    categorized[category].append(theme)
                    categorized_theme = True
                    break
            
            if not categorized_theme:
                uncategorized.append(theme)

        if uncategorized:
            categorized['Other'] = uncategorized

        return {k: v for k, v in categorized.items() if v}

    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string."""
        if not date_str:
            return 0
        try:
            return int(date_str[:4]) if len(date_str) >= 4 else 0
        except (ValueError, TypeError):
            return 0

    def _calculate_year_stats(self, years: List[int]) -> Dict[str, Any]:
        """Calculate statistics for publication years."""
        if not years:
            return {}

        return {
            'min_year': min(years),
            'max_year': max(years),
            'avg_year': sum(years) / len(years),
            'median_year': sorted(years)[len(years) // 2],
            'year_range': max(years) - min(years),
            'distribution': dict(Counter(years))
        }

    def _calculate_citation_stats(self, citations: List[int]) -> Dict[str, Any]:
        """Calculate citation statistics."""
        if not citations:
            return {}

        citations_sorted = sorted(citations, reverse=True)
        
        return {
            'total_citations': sum(citations),
            'avg_citations': sum(citations) / len(citations),
            'median_citations': sorted(citations)[len(citations) // 2],
            'max_citations': max(citations),
            'min_citations': min(citations),
            'top_10_percent': citations_sorted[:max(1, len(citations) // 10)],
            'highly_cited_count': len([c for c in citations if c > 50])
        }

    def _calculate_author_stats(self, author_counts: List[int], all_authors: List[str]) -> Dict[str, Any]:
        """Calculate author statistics."""
        if not author_counts:
            return {}

        author_productivity = Counter(all_authors)

        return {
            'avg_authors_per_paper': sum(author_counts) / len(author_counts),
            'max_authors_per_paper': max(author_counts),
            'min_authors_per_paper': min(author_counts),
            'total_unique_authors': len(set(all_authors)),
            'most_productive_authors': dict(author_productivity.most_common(10)),
            'collaboration_rate': sum(1 for count in author_counts if count > 1) / len(author_counts)
        }

    def _calculate_relevance_stats(self, relevance_scores: List[float]) -> Dict[str, Any]:
        """Calculate relevance score statistics."""
        if not relevance_scores:
            return {}

        return {
            'avg_relevance': sum(relevance_scores) / len(relevance_scores),
            'max_relevance': max(relevance_scores),
            'min_relevance': min(relevance_scores),
            'median_relevance': sorted(relevance_scores)[len(relevance_scores) // 2],
            'high_relevance_count': len([s for s in relevance_scores if s > 0.7])
        }

    def _identify_emerging_trends(self, year_themes: Dict[int, Counter]) -> List[str]:
        """Identify emerging trends across years."""
        if not year_themes or len(year_themes) < 2:
            return []

        years = sorted(year_themes.keys())
        recent_years = years[-2:]  # Last 2 years
        earlier_years = years[:-2]  # Earlier years

        recent_themes = Counter()
        earlier_themes = Counter()

        for year in recent_years:
            recent_themes.update(year_themes[year])

        for year in earlier_years:
            earlier_themes.update(year_themes[year])

        # Themes that appear more frequently in recent years
        emerging = []
        for theme in recent_themes:
            recent_freq = recent_themes[theme] / len(recent_years)
            earlier_freq = earlier_themes.get(theme, 0) / max(len(earlier_years), 1)
            
            if recent_freq > earlier_freq * 1.5:  # 50% increase threshold
                emerging.append(theme)

        return emerging[:10]

    def _identify_declining_trends(self, year_themes: Dict[int, Counter]) -> List[str]:
        """Identify declining trends across years."""
        if not year_themes or len(year_themes) < 2:
            return []

        years = sorted(year_themes.keys())
        recent_years = years[-2:]  # Last 2 years
        earlier_years = years[:-2]  # Earlier years

        recent_themes = Counter()
        earlier_themes = Counter()

        for year in recent_years:
            recent_themes.update(year_themes[year])

        for year in earlier_years:
            earlier_themes.update(year_themes[year])

        # Themes that appear less frequently in recent years
        declining = []
        for theme in earlier_themes:
            if earlier_themes[theme] >= 2:  # Must have appeared at least twice
                recent_freq = recent_themes.get(theme, 0) / len(recent_years)
                earlier_freq = earlier_themes[theme] / len(earlier_years)
                
                if recent_freq < earlier_freq * 0.5:  # 50% decrease threshold
                    declining.append(theme)

        return declining[:10]

    def _analyze_publication_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze publication trends over time."""
        year_counts = Counter()
        
        for paper in papers:
            year = self._extract_year(paper.get('published_date', ''))
            if year > 0:
                year_counts[year] += 1

        if not year_counts:
            return {}

        years = sorted(year_counts.keys())
        trend = "stable"
        
        if len(years) >= 2:
            recent_avg = sum(year_counts[year] for year in years[-2:]) / 2
            earlier_avg = sum(year_counts[year] for year in years[:-2]) / max(len(years) - 2, 1)
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"

        return {
            'trend': trend,
            'yearly_distribution': dict(year_counts),
            'peak_year': max(year_counts.keys(), key=lambda y: year_counts[y]),
            'total_years_covered': len(years),
            'average_papers_per_year': sum(year_counts.values()) / len(years)
        }

    def _summarize_theme_evolution(self, year_themes: Dict[int, Counter]) -> str:
        """Summarize how themes have evolved over time."""
        if not year_themes:
            return "No temporal theme data available"

        years = sorted(year_themes.keys())
        if len(years) < 2:
            return "Insufficient temporal data for evolution analysis"

        early_themes = set(year_themes[years[0]].keys())
        recent_themes = set(year_themes[years[-1]].keys())
        
        new_themes = recent_themes - early_themes
        persistent_themes = early_themes & recent_themes

        return f"Theme evolution shows {len(new_themes)} new themes emerging, {len(persistent_themes)} themes persisting across time periods"

    async def _generate_analysis_insights(self, analysis_results: Dict[str, Any], papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive insights from all analysis results."""
        insights = {
            'key_findings': [],
            'research_landscape': {},
            'methodological_insights': {},
            'temporal_insights': {},
            'collaboration_insights': {}
        }

        # Extract key findings from each analysis type
        for analysis_type, results in analysis_results.items():
            if 'error' not in results:
                insights['key_findings'].extend(self._extract_key_findings_from_analysis(analysis_type, results))

        # Research landscape insights
        if 'thematic' in analysis_results:
            thematic = analysis_results['thematic']
            insights['research_landscape'] = {
                'dominant_research_areas': list(thematic.get('dominant_themes', {}).keys())[:5],
                'thematic_diversity': thematic.get('thematic_diversity', 0),
                'emerging_keywords': list(thematic.get('top_keywords', {}).keys())[:10]
            }

        # Temporal insights
        if 'trend' in analysis_results:
            trend = analysis_results['trend']
            insights['temporal_insights'] = {
                'publication_trend': trend.get('publication_trends', {}).get('trend', 'stable'),
                'emerging_themes': trend.get('emerging_themes', [])[:5],
                'declining_themes': trend.get('declining_themes', [])[:5]
            }

        # Collaboration insights
        if 'network' in analysis_results:
            network = analysis_results['network']
            insights['collaboration_insights'] = {
                'collaboration_level': 'high' if network.get('collaboration_network', {}).get('total_collaborations', 0) > len(papers) * 0.5 else 'moderate',
                'top_collaborators': list(network.get('collaboration_network', {}).get('most_collaborative_authors', {}).keys())[:5],
                'research_connectivity': network.get('keyword_network', {}).get('network_density', 0)
            }

        return insights

    def _extract_key_findings_from_analysis(self, analysis_type: str, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from specific analysis type."""
        findings = []

        if analysis_type == 'thematic':
            dominant_themes = results.get('dominant_themes', {})
            if dominant_themes:
                top_theme = max(dominant_themes.keys(), key=lambda k: dominant_themes[k])
                findings.append(f"'{top_theme}' is the most dominant research theme")

        elif analysis_type == 'statistical':
            year_stats = results.get('year_statistics', {})
            if year_stats:
                findings.append(f"Research spans {year_stats.get('year_range', 0)} years from {year_stats.get('min_year', 'N/A')} to {year_stats.get('max_year', 'N/A')}")

        elif analysis_type == 'trend':
            pub_trends = results.get('publication_trends', {})
            if pub_trends:
                findings.append(f"Publication trend is {pub_trends.get('trend', 'stable')}")

        elif analysis_type == 'network':
            collab = results.get('collaboration_network', {})
            if collab:
                findings.append(f"Identified {collab.get('total_collaborations', 0)} unique research collaborations")

        return findings

    async def _create_analysis_summary(self, analysis_results: Dict[str, Any], insights: Dict[str, Any]) -> str:
        """Create a comprehensive analysis summary."""
        summary_parts = []

        # Overview
        completed_analyses = [k for k, v in analysis_results.items() if 'error' not in v]
        summary_parts.append(f"Completed {len(completed_analyses)} types of analysis: {', '.join(completed_analyses)}")

        # Key insights
        key_findings = insights.get('key_findings', [])
        if key_findings:
            summary_parts.append(f"Key findings include: {'; '.join(key_findings[:3])}")

        # Research landscape
        landscape = insights.get('research_landscape', {})
        if landscape.get('dominant_research_areas'):
            summary_parts.append(f"Dominant research areas: {', '.join(landscape['dominant_research_areas'][:3])}")

        # Temporal insights
        temporal = insights.get('temporal_insights', {})
        if temporal.get('publication_trend'):
            summary_parts.append(f"Publications show {temporal['publication_trend']} trend over time")

        return ". ".join(summary_parts) + "."

    async def _execute_thematic_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute thematic analysis specifically."""
        previous_results = context.get('previous_results', {})
        papers = self._extract_papers_from_results(previous_results)
        return await self._thematic_analysis(papers)

    async def _execute_statistical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis specifically."""
        previous_results = context.get('previous_results', {})
        papers = self._extract_papers_from_results(previous_results)
        return await self._statistical_analysis(papers)

    async def _comprehensive_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using all available methods."""
        context_with_all_types = context.copy()
        context_with_all_types['task']['parameters'] = {
            'analysis_types': list(self.analysis_types.keys())
        }
        return await self._execute_content_analysis(context_with_all_types)

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main processing method for analysis agent."""
        if not context:
            context = {
                'task': {
                    'type': 'comprehensive_analysis',
                    'parameters': {
                        'analysis_types': ['thematic', 'statistical', 'trend']
                    }
                },
                'previous_results': {}
            }
        
        return await self.process_task(context)
