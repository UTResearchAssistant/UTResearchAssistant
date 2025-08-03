"""Trend Analysis Agent for identifying and analyzing research trends."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict, Counter

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TrendAgent(BaseAgent):
    """Specialized agent for identifying and analyzing research trends."""

    def __init__(self):
        super().__init__()
        self.trend_analysis_methods = {
            'keyword_trend': self._analyze_keyword_trends,
            'topic_trend': self._analyze_topic_trends,
            'methodology_trend': self._analyze_methodology_trends,
            'citation_trend': self._analyze_citation_trends,
            'emerging_trends': self._identify_emerging_trends,
            'trend_forecasting': self._forecast_future_trends
        }

    async def process_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process trend analysis tasks."""
        task = context.get('task', {})
        task_type = task.get('type')
        
        if task_type == 'trend_analysis':
            return await self._execute_comprehensive_trend_analysis(context)
        elif task_type == 'keyword_trend':
            return await self._execute_keyword_trend_analysis(context)
        elif task_type == 'topic_trend':
            return await self._execute_topic_trend_analysis(context)
        elif task_type == 'emerging_trends':
            return await self._execute_emerging_trend_identification(context)
        else:
            return await self._comprehensive_trend_analysis(context)

    async def _execute_comprehensive_trend_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive trend analysis."""
        task_params = context['task']['parameters']
        papers = context.get('papers', [])
        
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Starting comprehensive trend analysis for {len(papers)} papers")

        # Perform various trend analyses
        keyword_trends = await self._analyze_keyword_trends(papers)
        topic_trends = await self._analyze_topic_trends(papers)
        methodology_trends = await self._analyze_methodology_trends(papers)
        citation_trends = await self._analyze_citation_trends(papers)
        emerging_trends = await self._identify_emerging_trends(papers, keyword_trends, topic_trends)
        future_forecast = await self._forecast_future_trends(keyword_trends, topic_trends)
        
        result = {
            'keyword_trends': keyword_trends,
            'topic_trends': topic_trends,
            'methodology_trends': methodology_trends,
            'citation_trends': citation_trends,
            'emerging_trends': emerging_trends,
            'future_forecast': future_forecast,
            'trend_summary': await self._generate_trend_summary(
                keyword_trends, topic_trends, emerging_trends
            ),
            'metadata': {
                'total_papers': len(papers),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'comprehensive'
            }
        }

        return result

    async def _execute_keyword_trend_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute keyword trend analysis."""
        papers = context.get('papers', [])
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Analyzing keyword trends for {len(papers)} papers")
        
        keyword_trends = await self._analyze_keyword_trends(papers)
        
        return {
            'keyword_trends': keyword_trends,
            'summary': f"Analyzed keyword trends from {len(papers)} papers.",
            'metadata': {'analysis_timestamp': datetime.now().isoformat()}
        }

    async def _execute_topic_trend_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute topic trend analysis."""
        papers = context.get('papers', [])
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Analyzing topic trends for {len(papers)} papers")
        
        topic_trends = await self._analyze_topic_trends(papers)
        
        return {
            'topic_trends': topic_trends,
            'summary': f"Analyzed topic trends from {len(papers)} papers.",
            'metadata': {'analysis_timestamp': datetime.now().isoformat()}
        }

    async def _execute_emerging_trend_identification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emerging trend identification."""
        papers = context.get('papers', [])
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Identifying emerging trends from {len(papers)} papers")
        
        keyword_trends = await self._analyze_keyword_trends(papers)
        topic_trends = await self._analyze_topic_trends(papers)
        emerging_trends = await self._identify_emerging_trends(papers, keyword_trends, topic_trends)
        
        return {
            'emerging_trends': emerging_trends,
            'summary': f"Identified {len(emerging_trends.get('emerging_keywords', []))} emerging keywords and {len(emerging_trends.get('emerging_topics', []))} emerging topics.",
            'metadata': {'analysis_timestamp': datetime.now().isoformat()}
        }

    def _extract_papers_from_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract papers from context or previous results."""
        papers = []
        
        if 'papers' in context:
            papers.extend(context['papers'])
        
        previous_results = context.get('previous_results', {})
        for task_id, result in previous_results.items():
            if isinstance(result, dict) and 'result' in result:
                task_result = result['result']
                if 'papers' in task_result:
                    papers.extend(task_result['papers'])
        
        return papers

    async def _analyze_keyword_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends of keywords over time."""
        keyword_data = defaultdict(lambda: defaultdict(int))
        
        for paper in papers:
            year = paper.get('year')
            keywords = paper.get('keywords', [])
            
            if year and keywords:
                for keyword in keywords:
                    keyword_data[keyword][year] += 1
        
        # Analyze trends for each keyword
        trends = {}
        for keyword, year_counts in keyword_data.items():
            trends[keyword] = self._calculate_trend_metrics(year_counts)
        
        # Identify top and emerging keywords
        top_keywords = sorted(
            trends.items(),
            key=lambda x: x[1]['total_occurrences'],
            reverse=True
        )[:20]
        
        emerging_keywords = sorted(
            [item for item in trends.items() if item[1]['trend_type'] == 'emerging'],
            key=lambda x: x[1]['growth_rate'],
            reverse=True
        )[:10]

        return {
            'keyword_trends': trends,
            'top_keywords': dict(top_keywords),
            'emerging_keywords': dict(emerging_keywords),
            'keyword_frequency_distribution': self._get_frequency_distribution(keyword_data)
        }

    async def _analyze_topic_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends of research topics over time."""
        # Simplified topic modeling
        topic_data = defaultdict(lambda: defaultdict(int))
        
        for paper in papers:
            year = paper.get('year')
            topics = self._extract_topics(paper)
            
            if year and topics:
                for topic in topics:
                    topic_data[topic][year] += 1
        
        # Analyze trends for each topic
        trends = {}
        for topic, year_counts in topic_data.items():
            trends[topic] = self._calculate_trend_metrics(year_counts)
        
        # Identify top and emerging topics
        top_topics = sorted(
            trends.items(),
            key=lambda x: x[1]['total_occurrences'],
            reverse=True
        )[:10]
        
        emerging_topics = sorted(
            [item for item in trends.items() if item[1]['trend_type'] == 'emerging'],
            key=lambda x: x[1]['growth_rate'],
            reverse=True
        )[:5]

        return {
            'topic_trends': trends,
            'top_topics': dict(top_topics),
            'emerging_topics': dict(emerging_topics),
            'topic_evolution': self._analyze_topic_evolution(topic_data)
        }

    async def _analyze_methodology_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in research methodologies."""
        methodology_data = defaultdict(lambda: defaultdict(int))
        
        for paper in papers:
            year = paper.get('year')
            methodologies = self._extract_methodologies(paper)
            
            if year and methodologies:
                for methodology in methodologies:
                    methodology_data[methodology][year] += 1
        
        # Analyze trends for each methodology
        trends = {}
        for methodology, year_counts in methodology_data.items():
            trends[methodology] = self._calculate_trend_metrics(year_counts)
        
        # Identify dominant and emerging methodologies
        dominant_methodologies = sorted(
            trends.items(),
            key=lambda x: x[1]['total_occurrences'],
            reverse=True
        )[:5]
        
        emerging_methodologies = sorted(
            [item for item in trends.items() if item[1]['trend_type'] == 'emerging'],
            key=lambda x: x[1]['growth_rate'],
            reverse=True
        )[:3]

        return {
            'methodology_trends': trends,
            'dominant_methodologies': dict(dominant_methodologies),
            'emerging_methodologies': dict(emerging_methodologies),
            'methodological_shifts': self._identify_methodological_shifts(trends)
        }

    async def _analyze_citation_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in citation patterns."""
        citation_data = defaultdict(lambda: {'citations': 0, 'papers': 0})
        
        for paper in papers:
            year = paper.get('year')
            citations = paper.get('citations', 0)
            
            if year:
                citation_data[year]['citations'] += citations
                citation_data[year]['papers'] += 1
        
        # Calculate yearly averages
        yearly_avg_citations = {}
        for year, data in citation_data.items():
            yearly_avg_citations[year] = data['citations'] / data['papers'] if data['papers'] > 0 else 0
        
        # Analyze trend in average citations
        trend_metrics = self._calculate_trend_metrics(yearly_avg_citations)
        
        return {
            'yearly_average_citations': yearly_avg_citations,
            'citation_trend_metrics': trend_metrics,
            'citation_velocity': self._calculate_citation_velocity(papers),
            'citation_half_life': self._calculate_citation_half_life(papers)
        }

    async def _identify_emerging_trends(self, papers: List[Dict[str, Any]], 
                                      keyword_trends: Dict[str, Any], 
                                      topic_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Identify emerging trends from various analyses."""
        emerging = {
            'emerging_keywords': list(keyword_trends.get('emerging_keywords', {}).keys()),
            'emerging_topics': list(topic_trends.get('emerging_topics', {}).keys()),
            'hot_papers': self._identify_hot_papers(papers),
            'new_collaborations': self._identify_new_collaborations(papers),
            'trend_signals': self._detect_trend_signals(papers)
        }
        
        return emerging

    async def _forecast_future_trends(self, keyword_trends: Dict[str, Any], 
                                    topic_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast future trends based on current data."""
        forecast = {
            'predicted_keywords': [],
            'predicted_topics': [],
            'potential_breakthroughs': [],
            'forecast_confidence': 'medium'
        }
        
        # Simple forecasting based on growth rate
        for keyword, metrics in keyword_trends.get('emerging_keywords', {}).items():
            if metrics['growth_rate'] > 0.5:
                forecast['predicted_keywords'].append({
                    'keyword': keyword,
                    'prediction': 'likely to grow',
                    'confidence': metrics['growth_rate']
                })
        
        for topic, metrics in topic_trends.get('emerging_topics', {}).items():
            if metrics['growth_rate'] > 0.3:
                forecast['predicted_topics'].append({
                    'topic': topic,
                    'prediction': 'likely to become more prominent',
                    'confidence': metrics['growth_rate']
                })
        
        # Identify potential breakthroughs
        forecast['potential_breakthroughs'] = self._identify_potential_breakthroughs(
            keyword_trends, topic_trends
        )
        
        return forecast

    async def _generate_trend_summary(self, keyword_trends: Dict[str, Any], 
                                    topic_trends: Dict[str, Any], 
                                    emerging_trends: Dict[str, Any]) -> str:
        """Generate a summary of trend analysis."""
        summary_parts = []
        
        # Top trends
        if keyword_trends.get('top_keywords'):
            top_keyword = list(keyword_trends['top_keywords'].keys())[0]
            summary_parts.append(f"Dominant keyword trend is '{top_keyword}'")
        
        if topic_trends.get('top_topics'):
            top_topic = list(topic_trends['top_topics'].keys())[0]
            summary_parts.append(f"and the main research topic is '{top_topic}'")
        
        # Emerging trends
        if emerging_trends.get('emerging_keywords'):
            emerging_keyword = emerging_trends['emerging_keywords'][0]
            summary_parts.append(f"Emerging trends are seen in '{emerging_keyword}'")
        
        if not summary_parts:
            summary_parts.append("Trend analysis completed successfully")
        
        return ". ".join(summary_parts) + "."

    def _calculate_trend_metrics(self, year_counts: Dict[int, int]) -> Dict[str, Any]:
        """Calculate trend metrics for a given item over years."""
        if not year_counts:
            return {}
        
        sorted_years = sorted(year_counts.keys())
        total_occurrences = sum(year_counts.values())
        
        # Trend type and growth rate
        trend_type = 'stable'
        growth_rate = 0
        
        if len(sorted_years) > 1:
            start_year = sorted_years[0]
            end_year = sorted_years[-1]
            
            start_count = year_counts[start_year]
            end_count = year_counts[end_year]
            
            # Simple growth rate calculation
            if start_count > 0:
                growth_rate = (end_count - start_count) / start_count
            elif end_count > 0:
                growth_rate = float('inf') # Infinite growth if starting from zero
            
            # Classify trend type
            if growth_rate > 0.5 and end_count > 2:
                trend_type = 'emerging'
            elif growth_rate > 0.1:
                trend_type = 'growing'
            elif growth_rate < -0.1:
                trend_type = 'declining'
        
        return {
            'total_occurrences': total_occurrences,
            'year_distribution': year_counts,
            'first_appearance': min(sorted_years),
            'last_appearance': max(sorted_years),
            'trend_type': trend_type,
            'growth_rate': growth_rate,
            'peak_year': max(year_counts, key=year_counts.get)
        }

    def _get_frequency_distribution(self, data: Dict[str, Dict[int, int]]) -> Dict[str, int]:
        """Get frequency distribution of items."""
        distribution = defaultdict(int)
        for item, year_counts in data.items():
            distribution[item] = sum(year_counts.values())
        
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def _extract_topics(self, paper: Dict[str, Any]) -> List[str]:
        """Extract topics from a paper (simplified)."""
        # In a real implementation, this would use NLP/topic modeling
        topics = set()
        
        keywords = paper.get('keywords', [])
        title = paper.get('title', '').lower()
        
        # Simple topic extraction
        if any(kw in keywords for kw in ['machine learning', 'ai', 'deep learning']):
            topics.add('artificial_intelligence')
        if any(kw in keywords for kw in ['data analysis', 'statistics']):
            topics.add('data_science')
        if 'human-computer interaction' in keywords:
            topics.add('hci')
        
        if not topics:
            topics.add('general_computing')
            
        return list(topics)

    def _analyze_topic_evolution(self, topic_data: Dict[str, Dict[int, int]]) -> Dict[str, Any]:
        """Analyze the evolution of topics over time."""
        evolution = {
            'topic_lifecycles': {},
            'topic_correlations': {}
        }
        
        # Analyze lifecycles
        for topic, year_counts in topic_data.items():
            metrics = self._calculate_trend_metrics(year_counts)
            evolution['topic_lifecycles'][topic] = {
                'lifecycle_stage': metrics['trend_type'],
                'peak_year': metrics['peak_year']
            }
        
        # Analyze correlations (simplified)
        # In a real system, this would check for co-occurrence in papers
        evolution['topic_correlations'] = {
            'artificial_intelligence': ['data_science'],
            'data_science': ['artificial_intelligence']
        }
        
        return evolution

    def _extract_methodologies(self, paper: Dict[str, Any]) -> List[str]:
        """Extract methodologies from a paper (simplified)."""
        methodologies = set()
        
        abstract = paper.get('abstract', '').lower()
        
        if 'case study' in abstract:
            methodologies.add('case_study')
        if 'survey' in abstract or 'questionnaire' in abstract:
            methodologies.add('survey')
        if 'experiment' in abstract or 'controlled trial' in abstract:
            methodologies.add('experiment')
        if 'review' in abstract or 'literature review' in abstract:
            methodologies.add('literature_review')
        
        if not methodologies:
            methodologies.add('unspecified')
            
        return list(methodologies)

    def _identify_methodological_shifts(self, trends: Dict[str, Any]) -> List[str]:
        """Identify shifts in research methodologies."""
        shifts = []
        
        growing_methods = [m for m, t in trends.items() if t['trend_type'] == 'growing']
        declining_methods = [m for m, t in trends.items() if t['trend_type'] == 'declining']
        
        if growing_methods and declining_methods:
            shifts.append(f"Shift from {declining_methods[0]} towards {growing_methods[0]}")
        
        if not shifts:
            shifts.append("Methodological landscape appears stable")
            
        return shifts

    def _calculate_citation_velocity(self, papers: List[Dict[str, Any]]) -> float:
        """Calculate citation velocity (average citations per year)."""
        total_velocity = 0
        current_year = datetime.now().year
        
        for paper in papers:
            year = paper.get('year')
            citations = paper.get('citations', 0)
            
            if year and citations > 0:
                age = current_year - year + 1
                if age > 0:
                    total_velocity += citations / age
        
        return total_velocity / len(papers) if papers else 0

    def _calculate_citation_half_life(self, papers: List[Dict[str, Any]]) -> float:
        """Estimate citation half-life."""
        # Simplified estimation
        return 5.0  # Placeholder value

    def _identify_hot_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify 'hot' papers (recent and highly cited)."""
        hot_papers = []
        current_year = datetime.now().year
        
        for paper in papers:
            year = paper.get('year')
            citations = paper.get('citations', 0)
            
            if year and current_year - year <= 2 and citations > 10:
                hot_papers.append({
                    'title': paper.get('title'),
                    'year': year,
                    'citations': citations,
                    'velocity': citations / (current_year - year + 1)
                })
        
        return sorted(hot_papers, key=lambda x: x['velocity'], reverse=True)

    def _identify_new_collaborations(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Identify new collaboration patterns."""
        # Simplified implementation
        return ["Cross-institutional collaborations on the rise"]

    def _detect_trend_signals(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect weak signals of future trends."""
        signals = {
            'unusual_keyword_combinations': [],
            'novel_methodologies': []
        }
        
        # Detect unusual keyword combinations
        all_keywords = [kw for p in papers for kw in p.get('keywords', [])]
        keyword_counts = Counter(all_keywords)
        
        for paper in papers:
            keywords = paper.get('keywords', [])
            if len(keywords) > 1:
                # Check for pairs of rare keywords
                rare_keywords = [kw for kw in keywords if keyword_counts.get(kw, 0) <= 2]
                if len(rare_keywords) >= 2:
                    signals['unusual_keyword_combinations'].append(tuple(sorted(rare_keywords)))
        
        signals['unusual_keyword_combinations'] = list(set(signals['unusual_keyword_combinations']))[:5]
        
        return signals

    def _identify_potential_breakthroughs(self, keyword_trends: Dict[str, Any], 
                                        topic_trends: Dict[str, Any]) -> List[str]:
        """Identify areas with potential for breakthroughs."""
        breakthroughs = []
        
        # Look for intersection of emerging keywords and topics
        emerging_keywords = set(keyword_trends.get('emerging_keywords', {}).keys())
        emerging_topics = set(topic_trends.get('emerging_topics', {}).keys())
        
        # This is a heuristic - real breakthroughs are hard to predict
        if 'quantum_computing' in emerging_keywords and 'artificial_intelligence' in emerging_topics:
            breakthroughs.append("Potential breakthrough at the intersection of AI and Quantum Computing")
        
        if not breakthroughs:
            breakthroughs.append("Interdisciplinary research areas show high potential for breakthroughs")
            
        return breakthroughs

    async def _comprehensive_trend_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive trend analysis."""
        context_with_comprehensive = context.copy()
        context_with_comprehensive['task'] = {
            'type': 'trend_analysis',
            'parameters': {
                'analysis_type': 'comprehensive'
            }
        }
        return await self._execute_comprehensive_trend_analysis(context_with_comprehensive)

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main processing method for trend agent."""
        if not context:
            context = {
                'task': {
                    'type': 'trend_analysis',
                    'parameters': {}
                },
                'papers': []
            }
        
        return await self.process_task(context)
