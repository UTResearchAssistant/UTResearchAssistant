"""
Enhanced API endpoints for the Research Assistant Backend
Integrates all advanced services from the notebook implementation
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
import json
import asyncio
import logging
from typing import Dict, Any

# Import our enhanced services
from ..services.literature_search_service import literature_search_service
from ..services.paper_analysis_service import paper_analysis_service

logger = logging.getLogger(__name__)


def async_view(view_func):
    """Decorator to handle async views in Django"""
    def wrapper(request, *args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(view_func(request, *args, **kwargs))
        finally:
            loop.close()
    return wrapper


@method_decorator(csrf_exempt, name='dispatch')
class EnhancedLiteratureSearchView(View):
    """Enhanced literature search with unified multi-database search"""
    
    @async_view
    async def post(self, request):
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            max_results = data.get('max_results', 20)
            filters = data.get('filters', {})
            
            if not query:
                return JsonResponse({'error': 'Query parameter is required'}, status=400)
            
            # Use the enhanced literature search service
            async with literature_search_service as service:
                if filters:
                    results = await service.search_with_filters(query, filters)
                else:
                    results = await service.unified_search(query, max_results)
            
            response_data = {
                'query': query,
                'total_results': len(results),
                'results': results,
                'sources': list(set(paper.get('source', 'Unknown') for paper in results)),
                'filters_applied': filters
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaperAnalysisView(View):
    """Advanced paper analysis with AI-powered insights"""
    
    @async_view
    async def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            required_fields = ['title']
            for field in required_fields:
                if field not in data:
                    return JsonResponse({'error': f'Missing required field: {field}'}, status=400)
            
            # Analyze the paper
            analysis = await paper_analysis_service.analyze_paper(data)
            
            response_data = {
                'paper_info': {
                    'title': data.get('title'),
                    'authors': data.get('authors', []),
                    'publication_date': data.get('publication_date'),
                    'journal': data.get('journal'),
                },
                'analysis': analysis,
                'timestamp': str(asyncio.get_event_loop().time())
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Paper analysis error: {e}")
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaperComparisonView(View):
    """Compare multiple papers for similarities and differences"""
    
    @async_view
    async def post(self, request):
        try:
            data = json.loads(request.body)
            papers = data.get('papers', [])
            
            if len(papers) < 2:
                return JsonResponse({'error': 'At least 2 papers required for comparison'}, status=400)
            
            # Compare papers
            comparison = await paper_analysis_service.compare_papers(papers)
            
            response_data = {
                'comparison_type': 'multi_paper_analysis',
                'paper_count': len(papers),
                'comparison': comparison,
                'timestamp': str(asyncio.get_event_loop().time())
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Paper comparison error: {e}")
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class SmartRecommendationView(View):
    """AI-powered research recommendations"""
    
    @async_view
    async def post(self, request):
        try:
            data = json.loads(request.body)
            user_interests = data.get('interests', [])
            research_context = data.get('context', '')
            recommendation_type = data.get('type', 'papers')  # papers, researchers, topics
            
            recommendations = await self._generate_recommendations(
                user_interests, research_context, recommendation_type
            )
            
            response_data = {
                'recommendation_type': recommendation_type,
                'user_interests': user_interests,
                'recommendations': recommendations,
                'timestamp': str(asyncio.get_event_loop().time())
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    async def _generate_recommendations(self, interests, context, rec_type):
        """Generate AI-powered recommendations"""
        try:
            recommendations = []
            
            if rec_type == 'papers':
                # Recommend papers based on interests
                for interest in interests[:3]:  # Limit to top 3 interests
                    async with literature_search_service as service:
                        papers = await service.unified_search(interest, max_results=3)
                        for paper in papers:
                            # Add relevance scoring
                            paper['relevance_score'] = self._calculate_relevance(paper, interests)
                            recommendations.append(paper)
                
                # Sort by relevance and return top 10
                recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                return recommendations[:10]
            
            elif rec_type == 'topics':
                # Recommend related topics
                topic_suggestions = []
                for interest in interests:
                    # Find related topics through keyword expansion
                    async with literature_search_service as service:
                        papers = await service.unified_search(interest, max_results=20)
                        
                    # Extract keywords from papers to find related topics
                    all_keywords = []
                    for paper in papers:
                        analysis = await paper_analysis_service.analyze_paper(paper)
                        all_keywords.extend(analysis.get('keywords', []))
                    
                    # Find most common keywords not in original interests
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    for keyword, count in keyword_counts.most_common(10):
                        if keyword.lower() not in [i.lower() for i in interests]:
                            topic_suggestions.append({
                                'topic': keyword,
                                'relevance_count': count,
                                'related_to': interest
                            })
                
                return topic_suggestions[:15]
            
            else:
                return []
                
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return []
    
    def _calculate_relevance(self, paper, interests):
        """Calculate relevance score for a paper given user interests"""
        score = 0
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        
        for interest in interests:
            interest_lower = interest.lower()
            if interest_lower in title:
                score += 2  # Title match is more important
            if interest_lower in abstract:
                score += 1  # Abstract match
        
        # Citation count boost
        citations = paper.get('citation_count', 0)
        score += min(citations / 1000, 1)  # Normalize citation boost
        
        return score


@method_decorator(csrf_exempt, name='dispatch')
class ResearchTrendAnalysisView(View):
    """Analyze research trends and emerging topics"""
    
    @async_view
    async def get(self, request, field):
        try:
            time_range = request.GET.get('time_range', '5_years')
            
            # Analyze trends in the field
            trends = await self._analyze_field_trends(field, time_range)
            
            response_data = {
                'field': field,
                'time_range': time_range,
                'trends': trends,
                'timestamp': str(asyncio.get_event_loop().time())
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    async def _analyze_field_trends(self, field, time_range):
        """Analyze trends in a research field"""
        try:
            # Get papers from the field
            async with literature_search_service as service:
                papers = await service.unified_search(field, max_results=100)
            
            if not papers:
                return {'error': 'No papers found for the field'}
            
            # Filter by time range
            filtered_papers = self._filter_papers_by_time(papers, time_range)
            
            # Analyze keywords and trends
            all_keywords = []
            papers_by_year = {}
            
            for paper in filtered_papers:
                # Extract year
                year = self._extract_year_from_date(paper.get('publication_date'))
                if year:
                    if year not in papers_by_year:
                        papers_by_year[year] = []
                    papers_by_year[year].append(paper)
                
                # Get keywords
                analysis = await paper_analysis_service.analyze_paper(paper)
                all_keywords.extend(analysis.get('keywords', []))
            
            # Find trending keywords
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            trending_keywords = [
                {'keyword': kw, 'frequency': count} 
                for kw, count in keyword_counts.most_common(20)
            ]
            
            # Analyze publication growth
            growth_data = []
            sorted_years = sorted(papers_by_year.keys())
            for year in sorted_years:
                growth_data.append({
                    'year': year,
                    'paper_count': len(papers_by_year[year])
                })
            
            return {
                'total_papers': len(filtered_papers),
                'trending_keywords': trending_keywords,
                'publication_growth': growth_data,
                'year_range': f"{min(sorted_years)}-{max(sorted_years)}" if sorted_years else "Unknown",
                'most_productive_year': max(papers_by_year.keys(), key=lambda y: len(papers_by_year[y])) if papers_by_year else None
            }
            
        except Exception as e:
            logger.error(f"Field trend analysis error: {e}")
            return {'error': str(e)}
    
    def _filter_papers_by_time(self, papers, time_range):
        """Filter papers by time range"""
        from datetime import datetime
        current_year = datetime.now().year
        
        if time_range == "1_year":
            cutoff_year = current_year - 1
        elif time_range == "3_years":
            cutoff_year = current_year - 3
        elif time_range == "5_years":
            cutoff_year = current_year - 5
        elif time_range == "10_years":
            cutoff_year = current_year - 10
        else:
            cutoff_year = current_year - 5
        
        filtered = []
        for paper in papers:
            year = self._extract_year_from_date(paper.get("publication_date"))
            if year and year >= cutoff_year:
                filtered.append(paper)
        
        return filtered
    
    def _extract_year_from_date(self, date_str):
        """Extract year from date string"""
        if not date_str:
            return None
        
        try:
            import re
            from datetime import datetime
            
            if isinstance(date_str, datetime):
                return date_str.year
            elif isinstance(date_str, str):
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    return int(year_match.group())
            return None
        except:
            return None


@method_decorator(csrf_exempt, name='dispatch')
class ResearchGapAnalysisView(View):
    """Identify research gaps and opportunities"""
    
    @async_view
    async def get(self, request, field):
        try:
            depth = request.GET.get('depth', 'comprehensive')
            
            # Find research gaps
            gaps = await self._find_research_gaps(field, depth)
            
            response_data = {
                'field': field,
                'analysis_depth': depth,
                'gaps': gaps,
                'gap_count': len(gaps),
                'timestamp': str(asyncio.get_event_loop().time())
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Research gap analysis error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    async def _find_research_gaps(self, field, depth):
        """Find research gaps in a specific field"""
        try:
            # Get comprehensive literature for the field
            async with literature_search_service as service:
                papers = await service.unified_search(field, max_results=150)
            
            if not papers:
                return []
            
            gaps = []
            
            # Analyze methodology gaps
            methodology_gaps = await self._find_methodology_gaps(papers)
            gaps.extend(methodology_gaps)
            
            # Analyze application gaps
            application_gaps = await self._find_application_gaps(papers)
            gaps.extend(application_gaps)
            
            # Analyze temporal gaps
            temporal_gaps = await self._find_temporal_gaps(papers)
            gaps.extend(temporal_gaps)
            
            # Score and rank gaps
            scored_gaps = []
            for gap in gaps:
                score = self._calculate_gap_importance(gap, papers)
                gap["importance_score"] = score
                scored_gaps.append(gap)
            
            scored_gaps.sort(key=lambda x: x["importance_score"], reverse=True)
            return scored_gaps[:15]
            
        except Exception as e:
            logger.error(f"Research gap finding error: {e}")
            return []
    
    async def _find_methodology_gaps(self, papers):
        """Find gaps in research methodologies"""
        methodology_terms = {}
        
        # Extract methodology terms from papers
        for paper in papers:
            analysis = await paper_analysis_service.analyze_paper(paper)
            methodology = analysis.get('methodology', '')
            
            # Simple methodology term extraction
            method_keywords = ['machine learning', 'deep learning', 'statistical analysis', 
                             'experimental', 'survey', 'case study', 'simulation']
            
            for method in method_keywords:
                if method in methodology.lower():
                    methodology_terms[method] = methodology_terms.get(method, 0) + 1
        
        gaps = []
        expected_methods = ['deep learning', 'statistical analysis', 'experimental', 'simulation']
        
        for method in expected_methods:
            count = methodology_terms.get(method, 0)
            if count < 3:  # Threshold for underexplored
                gaps.append({
                    "type": "methodology",
                    "gap_description": f"Limited application of {method} in this field",
                    "method": method,
                    "evidence": f"Only {count} papers found using this methodology",
                    "research_directions": [
                        f"Apply {method} to existing problems in the field",
                        f"Develop novel {method} approaches for domain-specific challenges"
                    ]
                })
        
        return gaps[:3]
    
    async def _find_application_gaps(self, papers):
        """Find gaps in application domains"""
        application_domains = {}
        
        domain_keywords = {
            "healthcare": ["medical", "health", "clinical", "patient"],
            "education": ["learning", "teaching", "student", "educational"],
            "finance": ["financial", "banking", "investment", "trading"],
            "environment": ["environmental", "climate", "sustainability", "green"]
        }
        
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            for domain, keywords in domain_keywords.items():
                if any(keyword in text for keyword in keywords):
                    application_domains[domain] = application_domains.get(domain, 0) + 1
        
        gaps = []
        total_papers = len(papers)
        
        for domain, count in application_domains.items():
            coverage_percentage = (count / total_papers) * 100 if total_papers > 0 else 0
            
            if coverage_percentage < 10:  # Less than 10% coverage
                gaps.append({
                    "type": "application",
                    "gap_description": f"Underexplored application in {domain}",
                    "domain": domain,
                    "coverage_percentage": coverage_percentage,
                    "evidence": f"Only {count} papers ({coverage_percentage:.1f}%) address {domain} applications",
                    "research_directions": [
                        f"Explore applications in {domain}",
                        f"Develop {domain}-specific methodologies"
                    ]
                })
        
        return gaps[:3]
    
    async def _find_temporal_gaps(self, papers):
        """Find temporal gaps in research coverage"""
        from datetime import datetime
        current_year = datetime.now().year
        papers_by_year = {}
        
        for paper in papers:
            year = self._extract_year_from_date(paper.get("publication_date"))
            if year and year >= current_year - 10:  # Last 10 years
                papers_by_year[year] = papers_by_year.get(year, 0) + 1
        
        gaps = []
        if papers_by_year:
            avg_papers = sum(papers_by_year.values()) / len(papers_by_year)
            
            for year, count in papers_by_year.items():
                if count < avg_papers * 0.5:  # Less than 50% of average
                    gaps.append({
                        "type": "temporal",
                        "gap_description": f"Low research activity in {year}",
                        "year": year,
                        "paper_count": count,
                        "evidence": f"Only {count} papers published vs average of {avg_papers:.1f}",
                        "research_directions": [
                            f"Investigate knowledge gaps from {year}",
                            f"Update research from {year} with modern approaches"
                        ]
                    })
        
        return gaps[:2]
    
    def _calculate_gap_importance(self, gap, papers):
        """Calculate importance score for a research gap"""
        base_score = 1.0
        
        # Weight by gap type
        type_weights = {
            "methodology": 1.2,
            "application": 1.0,
            "temporal": 0.8
        }
        
        gap_type = gap.get("type", "unknown")
        score = base_score * type_weights.get(gap_type, 1.0)
        
        # Boost score for coverage metrics
        if "coverage_percentage" in gap:
            coverage = gap["coverage_percentage"]
            if coverage < 5:
                score *= 1.3
            elif coverage < 10:
                score *= 1.1
        
        return score
    
    def _extract_year_from_date(self, date_str):
        """Extract year from date string"""
        if not date_str:
            return None
        
        try:
            import re
            from datetime import datetime
            
            if isinstance(date_str, datetime):
                return date_str.year
            elif isinstance(date_str, str):
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    return int(year_match.group())
            return None
        except:
            return None


# Health check endpoint
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    return JsonResponse({
        'status': 'healthy',
        'timestamp': str(asyncio.get_event_loop().time()),
        'services': {
            'literature_search': 'active',
            'paper_analysis': 'active',
            'trend_analysis': 'active',
            'gap_analysis': 'active'
        }
    })
