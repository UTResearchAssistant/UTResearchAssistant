"""Citation Agent for managing citations and bibliographic analysis."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
import re
from collections import defaultdict, Counter

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CitationAgent(BaseAgent):
    """Specialized agent for citation analysis and bibliographic management."""

    def __init__(self):
        super().__init__()
        self.citation_formats = {
            'apa': self._format_apa,
            'mla': self._format_mla,
            'chicago': self._format_chicago,
            'ieee': self._format_ieee,
            'nature': self._format_nature,
            'vancouver': self._format_vancouver
        }
        
        self.citation_patterns = {
            'doi': re.compile(r'10\.\d{4,}[/.]\S+'),
            'pmid': re.compile(r'PMID:\s*(\d+)'),
            'arxiv': re.compile(r'arXiv:(\d{4}\.\d{4,5})')
        }

    async def process_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process citation-related tasks."""
        task = context.get('task', {})
        task_type = task.get('type')
        
        if task_type == 'citation_analysis':
            return await self._execute_citation_analysis(context)
        elif task_type == 'format_citations':
            return await self._execute_citation_formatting(context)
        elif task_type == 'impact_analysis':
            return await self._execute_impact_analysis(context)
        elif task_type == 'bibliography_generation':
            return await self._execute_bibliography_generation(context)
        else:
            return await self._comprehensive_citation_analysis(context)

    async def _execute_citation_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive citation analysis."""
        task_params = context['task']['parameters']
        papers = context.get('papers', [])
        
        if not papers:
            # Extract papers from previous results
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Starting citation analysis for {len(papers)} papers")

        # Perform various citation analyses
        citation_network = await self._build_citation_network(papers)
        impact_metrics = await self._calculate_impact_metrics(papers)
        temporal_analysis = await self._analyze_temporal_patterns(papers)
        author_analysis = await self._analyze_author_citations(papers)
        venue_analysis = await self._analyze_venue_impact(papers)
        citation_quality = await self._assess_citation_quality(papers)
        
        result = {
            'citation_network': citation_network,
            'impact_metrics': impact_metrics,
            'temporal_analysis': temporal_analysis,
            'author_analysis': author_analysis,
            'venue_analysis': venue_analysis,
            'citation_quality': citation_quality,
            'summary': await self._generate_citation_summary(
                citation_network, impact_metrics, temporal_analysis
            ),
            'metadata': {
                'total_papers': len(papers),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'comprehensive'
            }
        }

        return result

    async def _execute_citation_formatting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute citation formatting task."""
        task_params = context['task']['parameters']
        papers = context.get('papers', [])
        format_style = task_params.get('format', 'apa')
        
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Formatting {len(papers)} citations in {format_style} style")

        formatted_citations = []
        bibliography = []
        
        for i, paper in enumerate(papers):
            formatted_citation = await self._format_citation(paper, format_style)
            bibliography_entry = await self._format_bibliography_entry(paper, format_style)
            
            formatted_citations.append({
                'index': i + 1,
                'formatted_citation': formatted_citation,
                'bibliography_entry': bibliography_entry,
                'original_paper': paper
            })
            
            bibliography.append(bibliography_entry)

        result = {
            'formatted_citations': formatted_citations,
            'bibliography': sorted(bibliography),
            'format_style': format_style,
            'citation_count': len(papers),
            'metadata': {
                'formatting_timestamp': datetime.now().isoformat(),
                'format_guidelines': self._get_format_guidelines(format_style)
            }
        }

        return result

    async def _execute_impact_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute impact analysis of citations."""
        papers = context.get('papers', [])
        
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Analyzing impact metrics for {len(papers)} papers")

        # Calculate various impact metrics
        h_index = self._calculate_h_index(papers)
        citation_distribution = self._analyze_citation_distribution(papers)
        high_impact_papers = self._identify_high_impact_papers(papers)
        influence_metrics = await self._calculate_influence_metrics(papers)
        
        result = {
            'h_index': h_index,
            'citation_distribution': citation_distribution,
            'high_impact_papers': high_impact_papers,
            'influence_metrics': influence_metrics,
            'impact_summary': await self._generate_impact_summary(
                h_index, citation_distribution, high_impact_papers
            ),
            'recommendations': self._generate_impact_recommendations(papers),
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_papers_analyzed': len(papers)
            }
        }

        return result

    async def _execute_bibliography_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bibliography generation."""
        task_params = context['task']['parameters']
        papers = context.get('papers', [])
        format_style = task_params.get('format', 'apa')
        sort_by = task_params.get('sort_by', 'author')
        
        if not papers:
            papers = self._extract_papers_from_context(context)
        
        logger.info(f"Generating bibliography for {len(papers)} papers")

        # Generate bibliography entries
        bibliography_entries = []
        for paper in papers:
            entry = await self._format_bibliography_entry(paper, format_style)
            bibliography_entries.append({
                'entry': entry,
                'paper_data': paper,
                'sort_key': self._get_sort_key(paper, sort_by)
            })

        # Sort bibliography
        bibliography_entries.sort(key=lambda x: x['sort_key'])
        
        # Generate final bibliography
        bibliography = [entry['entry'] for entry in bibliography_entries]
        
        result = {
            'bibliography': bibliography,
            'bibliography_entries': bibliography_entries,
            'format_style': format_style,
            'sort_method': sort_by,
            'entry_count': len(bibliography),
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'format_guidelines': self._get_format_guidelines(format_style)
            }
        }

        return result

    def _extract_papers_from_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract papers from context or previous results."""
        papers = []
        
        # Check direct papers
        if 'papers' in context:
            papers.extend(context['papers'])
        
        # Check previous results
        previous_results = context.get('previous_results', {})
        for task_id, result in previous_results.items():
            if isinstance(result, dict) and 'result' in result:
                task_result = result['result']
                if 'papers' in task_result:
                    papers.extend(task_result['papers'])
        
        return papers

    async def _build_citation_network(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build citation network from papers."""
        network = {
            'nodes': [],
            'edges': [],
            'statistics': {},
            'communities': [],
            'central_papers': []
        }

        # Create nodes for each paper
        for i, paper in enumerate(papers):
            node = {
                'id': i,
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'year': paper.get('year', 0),
                'citations': paper.get('citations', 0),
                'doi': paper.get('doi', ''),
                'venue': paper.get('venue', '')
            }
            network['nodes'].append(node)

        # Identify citation relationships (simplified)
        citation_pairs = self._find_citation_relationships(papers)
        
        for source_idx, target_idx, relationship_type in citation_pairs:
            edge = {
                'source': source_idx,
                'target': target_idx,
                'type': relationship_type,
                'weight': 1
            }
            network['edges'].append(edge)

        # Calculate network statistics
        network['statistics'] = {
            'total_nodes': len(network['nodes']),
            'total_edges': len(network['edges']),
            'network_density': len(network['edges']) / (len(network['nodes']) * (len(network['nodes']) - 1)) if len(network['nodes']) > 1 else 0,
            'average_degree': (2 * len(network['edges'])) / len(network['nodes']) if len(network['nodes']) > 0 else 0
        }

        # Identify central papers
        network['central_papers'] = self._identify_central_papers(network)
        
        # Detect communities (simplified)
        network['communities'] = self._detect_communities(network)

        return network

    async def _calculate_impact_metrics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate various impact metrics."""
        if not papers:
            return {}

        citations = [paper.get('citations', 0) for paper in papers]
        total_citations = sum(citations)
        
        metrics = {
            'total_citations': total_citations,
            'average_citations': total_citations / len(papers) if papers else 0,
            'median_citations': sorted(citations)[len(citations) // 2] if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'min_citations': min(citations) if citations else 0,
            'h_index': self._calculate_h_index(papers),
            'g_index': self._calculate_g_index(papers),
            'citation_distribution': self._analyze_citation_distribution(papers),
            'highly_cited_threshold': self._calculate_highly_cited_threshold(papers),
            'impact_percentiles': self._calculate_impact_percentiles(citations)
        }

        return metrics

    async def _analyze_temporal_patterns(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal citation patterns."""
        temporal_data = defaultdict(list)
        
        for paper in papers:
            year = paper.get('year')
            citations = paper.get('citations', 0)
            if year:
                temporal_data[year].append(citations)

        temporal_analysis = {
            'year_distribution': {},
            'citation_trends': {},
            'peak_years': [],
            'growth_analysis': {}
        }

        # Calculate yearly statistics
        for year, year_citations in temporal_data.items():
            temporal_analysis['year_distribution'][year] = {
                'paper_count': len(year_citations),
                'total_citations': sum(year_citations),
                'average_citations': sum(year_citations) / len(year_citations),
                'max_citations': max(year_citations)
            }

        # Identify trends
        sorted_years = sorted(temporal_data.keys())
        if len(sorted_years) > 1:
            temporal_analysis['citation_trends'] = self._calculate_citation_trends(sorted_years, temporal_data)
            temporal_analysis['peak_years'] = self._identify_peak_years(temporal_data)
            temporal_analysis['growth_analysis'] = self._analyze_growth_patterns(sorted_years, temporal_data)

        return temporal_analysis

    async def _analyze_author_citations(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation patterns by author."""
        author_stats = defaultdict(lambda: {
            'papers': 0,
            'total_citations': 0,
            'citations_list': [],
            'h_index': 0,
            'collaborators': set()
        })

        # Collect author statistics
        for paper in papers:
            authors = paper.get('authors', [])
            citations = paper.get('citations', 0)
            
            for author in authors:
                author_name = self._normalize_author_name(author)
                author_stats[author_name]['papers'] += 1
                author_stats[author_name]['total_citations'] += citations
                author_stats[author_name]['citations_list'].append(citations)
                
                # Track collaborators
                for other_author in authors:
                    if other_author != author:
                        author_stats[author_name]['collaborators'].add(
                            self._normalize_author_name(other_author)
                        )

        # Calculate derived metrics
        for author, stats in author_stats.items():
            stats['average_citations'] = stats['total_citations'] / stats['papers']
            stats['h_index'] = self._calculate_author_h_index(stats['citations_list'])
            stats['collaboration_count'] = len(stats['collaborators'])
            stats['collaborators'] = list(stats['collaborators'])  # Convert set to list for JSON

        # Identify top authors
        top_authors = sorted(
            author_stats.items(),
            key=lambda x: x[1]['total_citations'],
            reverse=True
        )[:10]

        return {
            'author_statistics': dict(author_stats),
            'top_authors_by_citations': top_authors,
            'total_unique_authors': len(author_stats),
            'collaboration_network': self._build_author_collaboration_network(author_stats),
            'prolific_authors': self._identify_prolific_authors(author_stats)
        }

    async def _analyze_venue_impact(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation patterns by publication venue."""
        venue_stats = defaultdict(lambda: {
            'papers': 0,
            'total_citations': 0,
            'citations_list': [],
            'average_citations': 0
        })

        # Collect venue statistics
        for paper in papers:
            venue = paper.get('venue', 'Unknown')
            citations = paper.get('citations', 0)
            
            venue_stats[venue]['papers'] += 1
            venue_stats[venue]['total_citations'] += citations
            venue_stats[venue]['citations_list'].append(citations)

        # Calculate derived metrics
        for venue, stats in venue_stats.items():
            stats['average_citations'] = stats['total_citations'] / stats['papers']
            stats['max_citations'] = max(stats['citations_list']) if stats['citations_list'] else 0
            stats['impact_factor_estimate'] = self._estimate_impact_factor(stats)

        # Identify top venues
        top_venues = sorted(
            venue_stats.items(),
            key=lambda x: x[1]['average_citations'],
            reverse=True
        )[:10]

        return {
            'venue_statistics': dict(venue_stats),
            'top_venues_by_impact': top_venues,
            'venue_diversity': len(venue_stats),
            'high_impact_venues': self._identify_high_impact_venues(venue_stats)
        }

    async def _assess_citation_quality(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of citations."""
        quality_metrics = {
            'citation_completeness': 0,
            'identifier_coverage': {},
            'metadata_quality': {},
            'citation_recency': {},
            'quality_score': 0
        }

        # Assess completeness
        complete_citations = 0
        total_papers = len(papers)
        
        doi_count = 0
        pmid_count = 0
        arxiv_count = 0
        
        for paper in papers:
            completeness_score = self._assess_paper_completeness(paper)
            if completeness_score > 0.8:
                complete_citations += 1
            
            # Check identifiers
            if paper.get('doi'):
                doi_count += 1
            if paper.get('pmid'):
                pmid_count += 1
            if 'arxiv' in paper.get('doi', '').lower() or 'arxiv' in paper.get('title', '').lower():
                arxiv_count += 1

        if total_papers > 0:
            quality_metrics['citation_completeness'] = complete_citations / total_papers
            quality_metrics['identifier_coverage'] = {
                'doi_coverage': doi_count / total_papers,
                'pmid_coverage': pmid_count / total_papers,
                'arxiv_coverage': arxiv_count / total_papers
            }
        
        # Calculate overall quality score
        quality_metrics['quality_score'] = self._calculate_overall_quality_score(quality_metrics)

        return quality_metrics

    async def _generate_citation_summary(self, citation_network: Dict[str, Any], 
                                       impact_metrics: Dict[str, Any],
                                       temporal_analysis: Dict[str, Any]) -> str:
        """Generate a summary of citation analysis."""
        summary_parts = []

        # Network summary
        if citation_network.get('statistics'):
            stats = citation_network['statistics']
            summary_parts.append(f"Citation network contains {stats['total_nodes']} papers with {stats['total_edges']} citation relationships")

        # Impact summary
        if impact_metrics:
            h_index = impact_metrics.get('h_index', 0)
            total_citations = impact_metrics.get('total_citations', 0)
            summary_parts.append(f"Collection has H-index of {h_index} with {total_citations} total citations")

        # Temporal summary
        if temporal_analysis.get('year_distribution'):
            years = list(temporal_analysis['year_distribution'].keys())
            if years:
                year_range = f"{min(years)}-{max(years)}"
                summary_parts.append(f"Research spans {year_range}")

        if not summary_parts:
            summary_parts.append("Citation analysis completed successfully")

        return ". ".join(summary_parts) + "."

    def _find_citation_relationships(self, papers: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        """Find citation relationships between papers (simplified implementation)."""
        relationships = []
        
        # This is a simplified implementation
        # In practice, you would match DOIs, titles, or other identifiers
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers):
                if i != j:
                    # Simple heuristic: papers by same authors might cite each other
                    authors1 = set(paper1.get('authors', []))
                    authors2 = set(paper2.get('authors', []))
                    
                    if authors1 & authors2:  # Common authors
                        relationships.append((i, j, 'co_authorship'))
                    
                    # Year-based heuristic: newer papers might cite older ones
                    year1 = paper1.get('year', 0)
                    year2 = paper2.get('year', 0)
                    
                    if year1 > year2 and year1 - year2 <= 5:
                        relationships.append((i, j, 'potential_citation'))

        return relationships[:50]  # Limit for performance

    def _identify_central_papers(self, network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify central papers in the citation network."""
        nodes = network['nodes']
        edges = network['edges']
        
        # Calculate degree centrality
        degree_count = defaultdict(int)
        for edge in edges:
            degree_count[edge['source']] += 1
            degree_count[edge['target']] += 1

        # Sort by degree centrality
        central_papers = []
        for i, node in enumerate(nodes):
            centrality = degree_count[i]
            if centrality > 0:
                central_papers.append({
                    'node_id': i,
                    'title': node['title'],
                    'centrality': centrality,
                    'citations': node['citations']
                })

        return sorted(central_papers, key=lambda x: x['centrality'], reverse=True)[:10]

    def _detect_communities(self, network: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect communities in the citation network (simplified)."""
        # Simplified community detection based on edge density
        communities = []
        
        if len(network['nodes']) > 0:
            # Simple clustering based on collaboration patterns
            communities.append({
                'id': 1,
                'size': len(network['nodes']),
                'description': 'Main research community',
                'central_papers': network.get('central_papers', [])[:3]
            })

        return communities

    def _calculate_h_index(self, papers: List[Dict[str, Any]]) -> int:
        """Calculate H-index for a collection of papers."""
        citations = [paper.get('citations', 0) for paper in papers]
        citations.sort(reverse=True)
        
        h_index = 0
        for i, citation_count in enumerate(citations):
            if citation_count >= i + 1:
                h_index = i + 1
            else:
                break
        
        return h_index

    def _calculate_g_index(self, papers: List[Dict[str, Any]]) -> int:
        """Calculate G-index for a collection of papers."""
        citations = [paper.get('citations', 0) for paper in papers]
        citations.sort(reverse=True)
        
        g_index = 0
        cumulative_citations = 0
        
        for i, citation_count in enumerate(citations):
            cumulative_citations += citation_count
            if cumulative_citations >= (i + 1) ** 2:
                g_index = i + 1
            else:
                break
        
        return g_index

    def _analyze_citation_distribution(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of citations."""
        citations = [paper.get('citations', 0) for paper in papers]
        citation_counts = Counter(citations)
        
        return {
            'distribution': dict(citation_counts),
            'uncited_papers': citation_counts.get(0, 0),
            'single_cited_papers': citation_counts.get(1, 0),
            'highly_cited_count': len([c for c in citations if c > 10]),
            'citation_ranges': {
                '0': len([c for c in citations if c == 0]),
                '1-5': len([c for c in citations if 1 <= c <= 5]),
                '6-10': len([c for c in citations if 6 <= c <= 10]),
                '11-50': len([c for c in citations if 11 <= c <= 50]),
                '50+': len([c for c in citations if c > 50])
            }
        }

    def _identify_high_impact_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify high-impact papers."""
        citations = [paper.get('citations', 0) for paper in papers]
        if not citations:
            return []
        
        # Calculate threshold (top 10% or papers with >50 citations)
        threshold = max(50, sorted(citations, reverse=True)[min(len(citations)//10, len(citations)-1)])
        
        high_impact = []
        for paper in papers:
            if paper.get('citations', 0) >= threshold:
                high_impact.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'year': paper.get('year'),
                    'citations': paper.get('citations', 0),
                    'venue': paper.get('venue', ''),
                    'doi': paper.get('doi', '')
                })
        
        return sorted(high_impact, key=lambda x: x['citations'], reverse=True)

    async def _calculate_influence_metrics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate influence metrics beyond simple citation counts."""
        return {
            'influence_score': self._calculate_overall_influence(papers),
            'temporal_influence': self._calculate_temporal_influence(papers),
            'network_influence': self._calculate_network_influence(papers),
            'field_influence': self._calculate_field_influence(papers)
        }

    def _calculate_overall_influence(self, papers: List[Dict[str, Any]]) -> float:
        """Calculate overall influence score."""
        if not papers:
            return 0.0
        
        total_citations = sum(paper.get('citations', 0) for paper in papers)
        paper_count = len(papers)
        h_index = self._calculate_h_index(papers)
        
        # Weighted influence score
        influence = (total_citations * 0.4 + h_index * 50 * 0.4 + paper_count * 10 * 0.2)
        return influence

    def _calculate_temporal_influence(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate influence over time."""
        current_year = datetime.now().year
        
        recent_papers = [p for p in papers if p.get('year', 0) >= current_year - 5]
        older_papers = [p for p in papers if p.get('year', 0) < current_year - 5]
        
        return {
            'recent_influence': sum(p.get('citations', 0) for p in recent_papers),
            'historical_influence': sum(p.get('citations', 0) for p in older_papers),
            'sustained_influence': len(older_papers) > 0 and len(recent_papers) > 0
        }

    def _calculate_network_influence(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate network-based influence metrics."""
        # Simplified network influence calculation
        unique_authors = set()
        unique_venues = set()
        
        for paper in papers:
            unique_authors.update(paper.get('authors', []))
            if paper.get('venue'):
                unique_venues.add(paper.get('venue'))
        
        return {
            'author_network_size': len(unique_authors),
            'venue_diversity': len(unique_venues),
            'collaboration_score': len(unique_authors) / len(papers) if papers else 0
        }

    def _calculate_field_influence(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate field-specific influence."""
        return {
            'cross_disciplinary_impact': self._assess_cross_disciplinary_impact(papers),
            'methodological_influence': self._assess_methodological_influence(papers),
            'theoretical_influence': self._assess_theoretical_influence(papers)
        }

    async def _generate_impact_summary(self, h_index: int, citation_distribution: Dict[str, Any],
                                     high_impact_papers: List[Dict[str, Any]]) -> str:
        """Generate summary of impact analysis."""
        summary_parts = []
        
        summary_parts.append(f"H-index of {h_index}")
        
        if citation_distribution:
            uncited = citation_distribution.get('uncited_papers', 0)
            highly_cited = citation_distribution.get('highly_cited_count', 0)
            summary_parts.append(f"{highly_cited} highly cited papers, {uncited} uncited papers")
        
        if high_impact_papers:
            top_paper = high_impact_papers[0]
            summary_parts.append(f"Most cited paper has {top_paper['citations']} citations")
        
        return ". ".join(summary_parts) + "."

    def _generate_impact_recommendations(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving impact."""
        recommendations = []
        
        citations = [paper.get('citations', 0) for paper in papers]
        avg_citations = sum(citations) / len(citations) if citations else 0
        
        if avg_citations < 10:
            recommendations.append("Consider targeting higher-impact venues for future publications")
        
        uncited_count = len([c for c in citations if c == 0])
        if uncited_count > len(citations) * 0.3:
            recommendations.append("Focus on promoting uncited papers through social media and conferences")
        
        venues = set(paper.get('venue', '') for paper in papers)
        if len(venues) < len(papers) * 0.5:
            recommendations.append("Diversify publication venues to reach broader audiences")
        
        if not recommendations:
            recommendations.append("Continue current publication strategy for sustained impact")
        
        return recommendations

    def _calculate_citation_trends(self, sorted_years: List[int], temporal_data: Dict[int, List[int]]) -> Dict[str, Any]:
        """Calculate citation trends over time."""
        yearly_totals = [sum(temporal_data[year]) for year in sorted_years]
        yearly_averages = [sum(temporal_data[year]) / len(temporal_data[year]) for year in sorted_years]
        
        trends = {
            'total_citation_trend': 'stable',
            'average_citation_trend': 'stable',
            'growth_rate': 0
        }
        
        if len(yearly_totals) > 1:
            # Simple trend analysis
            if yearly_totals[-1] > yearly_totals[0]:
                trends['total_citation_trend'] = 'increasing'
            elif yearly_totals[-1] < yearly_totals[0]:
                trends['total_citation_trend'] = 'decreasing'
            
            if yearly_averages[-1] > yearly_averages[0]:
                trends['average_citation_trend'] = 'increasing'
            elif yearly_averages[-1] < yearly_averages[0]:
                trends['average_citation_trend'] = 'decreasing'
        
        return trends

    def _identify_peak_years(self, temporal_data: Dict[int, List[int]]) -> List[int]:
        """Identify peak citation years."""
        year_totals = {year: sum(citations) for year, citations in temporal_data.items()}
        
        if not year_totals:
            return []
        
        max_citations = max(year_totals.values())
        peak_years = [year for year, total in year_totals.items() if total >= max_citations * 0.8]
        
        return sorted(peak_years)

    def _analyze_growth_patterns(self, sorted_years: List[int], temporal_data: Dict[int, List[int]]) -> Dict[str, Any]:
        """Analyze growth patterns in citations."""
        if len(sorted_years) < 2:
            return {'pattern': 'insufficient_data'}
        
        yearly_totals = [sum(temporal_data[year]) for year in sorted_years]
        
        # Simple growth pattern analysis
        increases = sum(1 for i in range(1, len(yearly_totals)) if yearly_totals[i] > yearly_totals[i-1])
        total_comparisons = len(yearly_totals) - 1
        
        growth_ratio = increases / total_comparisons if total_comparisons > 0 else 0
        
        if growth_ratio > 0.7:
            pattern = 'strong_growth'
        elif growth_ratio > 0.4:
            pattern = 'moderate_growth'
        elif growth_ratio > 0.3:
            pattern = 'stable'
        else:
            pattern = 'declining'
        
        return {
            'pattern': pattern,
            'growth_ratio': growth_ratio,
            'consistent_growth': growth_ratio > 0.8
        }

    def _normalize_author_name(self, author: str) -> str:
        """Normalize author name for consistency."""
        # Simple normalization - in practice, this would be more sophisticated
        return author.strip().title()

    def _calculate_author_h_index(self, citations_list: List[int]) -> int:
        """Calculate H-index for a specific author."""
        citations_list.sort(reverse=True)
        
        h_index = 0
        for i, citations in enumerate(citations_list):
            if citations >= i + 1:
                h_index = i + 1
            else:
                break
        
        return h_index

    def _build_author_collaboration_network(self, author_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build author collaboration network."""
        return {
            'total_collaborations': sum(len(stats['collaborators']) for stats in author_stats.values()),
            'highly_collaborative_authors': [
                author for author, stats in author_stats.items() 
                if len(stats['collaborators']) > 5
            ][:10],
            'collaboration_density': 'moderate'  # Simplified
        }

    def _identify_prolific_authors(self, author_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify prolific authors."""
        prolific = []
        
        for author, stats in author_stats.items():
            if stats['papers'] >= 3:  # Threshold for being prolific
                prolific.append({
                    'author': author,
                    'papers': stats['papers'],
                    'total_citations': stats['total_citations'],
                    'h_index': stats['h_index']
                })
        
        return sorted(prolific, key=lambda x: x['papers'], reverse=True)[:10]

    def _estimate_impact_factor(self, venue_stats: Dict[str, Any]) -> float:
        """Estimate impact factor for a venue."""
        # Simplified impact factor estimation
        avg_citations = venue_stats.get('average_citations', 0)
        paper_count = venue_stats.get('papers', 1)
        
        # Basic formula considering both average citations and number of papers
        impact_estimate = avg_citations * (1 + 0.1 * min(paper_count, 10))
        
        return round(impact_estimate, 2)

    def _identify_high_impact_venues(self, venue_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify high-impact venues."""
        high_impact = []
        
        for venue, stats in venue_stats.items():
            if stats['average_citations'] > 10 and stats['papers'] >= 2:
                high_impact.append({
                    'venue': venue,
                    'average_citations': stats['average_citations'],
                    'impact_factor_estimate': stats['impact_factor_estimate'],
                    'papers': stats['papers']
                })
        
        return sorted(high_impact, key=lambda x: x['average_citations'], reverse=True)

    def _assess_paper_completeness(self, paper: Dict[str, Any]) -> float:
        """Assess completeness of paper metadata."""
        required_fields = ['title', 'authors', 'year', 'venue']
        optional_fields = ['doi', 'abstract', 'keywords', 'citations']
        
        required_score = sum(1 for field in required_fields if paper.get(field))
        optional_score = sum(1 for field in optional_fields if paper.get(field))
        
        completeness = (required_score / len(required_fields)) * 0.7 + (optional_score / len(optional_fields)) * 0.3
        
        return completeness

    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        completeness = quality_metrics.get('citation_completeness', 0)
        
        identifier_coverage = quality_metrics.get('identifier_coverage', {})
        avg_identifier_coverage = sum(identifier_coverage.values()) / len(identifier_coverage) if identifier_coverage else 0
        
        quality_score = (completeness * 0.6 + avg_identifier_coverage * 0.4)
        
        return round(quality_score, 3)

    def _calculate_highly_cited_threshold(self, papers: List[Dict[str, Any]]) -> int:
        """Calculate threshold for highly cited papers."""
        citations = [paper.get('citations', 0) for paper in papers]
        if not citations:
            return 10
        
        # Use 90th percentile or minimum of 10
        citations.sort(reverse=True)
        percentile_90_index = int(len(citations) * 0.1)
        threshold = max(10, citations[percentile_90_index] if percentile_90_index < len(citations) else 10)
        
        return threshold

    def _calculate_impact_percentiles(self, citations: List[int]) -> Dict[str, int]:
        """Calculate impact percentiles."""
        if not citations:
            return {}
        
        sorted_citations = sorted(citations)
        n = len(sorted_citations)
        
        percentiles = {}
        for p in [25, 50, 75, 90, 95]:
            index = int(n * p / 100)
            if index >= n:
                index = n - 1
            percentiles[f'p{p}'] = sorted_citations[index]
        
        return percentiles

    def _assess_cross_disciplinary_impact(self, papers: List[Dict[str, Any]]) -> str:
        """Assess cross-disciplinary impact."""
        return "moderate"  # Simplified implementation

    def _assess_methodological_influence(self, papers: List[Dict[str, Any]]) -> str:
        """Assess methodological influence."""
        return "significant"  # Simplified implementation

    def _assess_theoretical_influence(self, papers: List[Dict[str, Any]]) -> str:
        """Assess theoretical influence."""
        return "moderate"  # Simplified implementation

    async def _format_citation(self, paper: Dict[str, Any], format_style: str) -> str:
        """Format a citation in the specified style."""
        formatter = self.citation_formats.get(format_style, self._format_apa)
        return await formatter(paper)

    async def _format_bibliography_entry(self, paper: Dict[str, Any], format_style: str) -> str:
        """Format a bibliography entry in the specified style."""
        # For now, use the same formatter as citations
        # In practice, bibliography entries might have slightly different formatting
        return await self._format_citation(paper, format_style)

    async def _format_apa(self, paper: Dict[str, Any]) -> str:
        """Format citation in APA style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        year = paper.get('year', 'n.d.')
        venue = paper.get('venue', 'Unknown Venue')
        
        # Format authors
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) <= 7:
            author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
        else:
            author_str = ", ".join(authors[:6]) + ", ... " + authors[-1]
        
        citation = f"{author_str} ({year}). {title}. {venue}."
        
        # Add DOI if available
        if paper.get('doi'):
            citation += f" https://doi.org/{paper['doi']}"
        
        return citation

    async def _format_mla(self, paper: Dict[str, Any]) -> str:
        """Format citation in MLA style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        venue = paper.get('venue', 'Unknown Venue')
        year = paper.get('year', 'n.d.')
        
        # Format authors (last name, first name for first author)
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) == 1:
            author_str = authors[0]
        else:
            author_str = f"{authors[0]}, et al."
        
        citation = f'{author_str}. "{title}." {venue}, {year}.'
        
        return citation

    async def _format_chicago(self, paper: Dict[str, Any]) -> str:
        """Format citation in Chicago style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        venue = paper.get('venue', 'Unknown Venue')
        year = paper.get('year', 'n.d.')
        
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) == 1:
            author_str = authors[0]
        else:
            author_str = f"{authors[0]} et al."
        
        citation = f'{author_str}. "{title}." {venue} ({year}).'
        
        return citation

    async def _format_ieee(self, paper: Dict[str, Any]) -> str:
        """Format citation in IEEE style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        venue = paper.get('venue', 'Unknown Venue')
        year = paper.get('year', 'n.d.')
        
        # IEEE uses initials for first names
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) <= 3:
            author_str = ", ".join(authors)
        else:
            author_str = f"{authors[0]} et al."
        
        citation = f'{author_str}, "{title}," {venue}, {year}.'
        
        return citation

    async def _format_nature(self, paper: Dict[str, Any]) -> str:
        """Format citation in Nature style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        venue = paper.get('venue', 'Unknown Venue')
        year = paper.get('year', 'n.d.')
        
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) <= 3:
            author_str = ", ".join(authors)
        else:
            author_str = f"{authors[0]} et al."
        
        citation = f"{author_str}. {title}. {venue} ({year})."
        
        return citation

    async def _format_vancouver(self, paper: Dict[str, Any]) -> str:
        """Format citation in Vancouver style."""
        authors = paper.get('authors', [])
        title = paper.get('title', 'Unknown Title')
        venue = paper.get('venue', 'Unknown Venue')
        year = paper.get('year', 'n.d.')
        
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) <= 6:
            author_str = ", ".join(authors)
        else:
            author_str = f"{', '.join(authors[:3])}, et al."
        
        citation = f"{author_str}. {title}. {venue}. {year}."
        
        return citation

    def _get_format_guidelines(self, format_style: str) -> str:
        """Get formatting guidelines for the specified style."""
        guidelines = {
            'apa': 'APA 7th Edition formatting guidelines',
            'mla': 'MLA 9th Edition formatting guidelines', 
            'chicago': 'Chicago Manual of Style 17th Edition',
            'ieee': 'IEEE Citation Reference standards',
            'nature': 'Nature journal citation format',
            'vancouver': 'Vancouver/ICMJE citation style'
        }
        
        return guidelines.get(format_style, 'Standard academic citation format')

    def _get_sort_key(self, paper: Dict[str, Any], sort_by: str) -> str:
        """Get sort key for bibliography entry."""
        if sort_by == 'author':
            authors = paper.get('authors', ['Unknown'])
            return authors[0] if authors else 'Unknown'
        elif sort_by == 'year':
            return str(paper.get('year', 0))
        elif sort_by == 'title':
            return paper.get('title', 'Unknown')
        else:
            return paper.get('title', 'Unknown')

    async def _comprehensive_citation_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive citation analysis."""
        context_with_comprehensive = context.copy()
        context_with_comprehensive['task'] = {
            'type': 'citation_analysis',
            'parameters': {
                'analysis_type': 'comprehensive'
            }
        }
        return await self._execute_citation_analysis(context_with_comprehensive)

    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main processing method for citation agent."""
        if not context:
            context = {
                'task': {
                    'type': 'citation_analysis',
                    'parameters': {}
                },
                'papers': []
            }
        
        return await self.process_task(context)
