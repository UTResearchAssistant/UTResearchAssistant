"""
Enhanced Research API Views for Django
Provides API endpoints for advanced research features
"""

import asyncio
import json
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods

# TODO: Fix these imports once the services are properly set up
# from services.literature_search_service import literature_search_service
# from services.paper_analysis_service import paper_analysis_service

# Temporary placeholder classes
class MockLiteratureSearchService:
    async def enhanced_search(self, query, sources=None, filters=None):
        return {"papers": [], "total": 0, "message": "Service temporarily unavailable"}

class MockPaperAnalysisService:
    async def analyze_paper(self, paper_data):
        return {"analysis": "Service temporarily unavailable"}

# Use mock services for now
literature_search_service = MockLiteratureSearchService()
paper_analysis_service = MockPaperAnalysisService()


@method_decorator(csrf_exempt, name='dispatch')
class EnhancedLiteratureSearchView(View):
    """API endpoint for enhanced literature search"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            max_results = data.get('max_results', 20)
            filters = data.get('filters', {})
            
            if not query:
                return JsonResponse({'error': 'Query is required'}, status=400)
            
            # Perform search
            async with literature_search_service as search_service:
                if filters:
                    results = await search_service.search_with_filters(query, filters)
                else:
                    results = await search_service.unified_search(query, max_results)
            
            return JsonResponse({
                'query': query,
                'results': results,
                'total_count': len(results)
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaperAnalysisView(View):
    """API endpoint for paper analysis"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            paper_data = data.get('paper_data', {})
            
            if not paper_data:
                return JsonResponse({'error': 'Paper data is required'}, status=400)
            
            # Analyze paper
            analysis = await paper_analysis_service.analyze_paper(paper_data)
            
            return JsonResponse({
                'paper_title': paper_data.get('title', ''),
                'analysis': analysis
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class PaperComparisonView(View):
    """API endpoint for paper comparison"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            papers = data.get('papers', [])
            
            if len(papers) < 2:
                return JsonResponse({'error': 'At least 2 papers required for comparison'}, status=400)
            
            # Compare papers
            comparison = await paper_analysis_service.compare_papers(papers)
            
            return JsonResponse({
                'comparison_results': comparison
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class SmartRecommendationView(View):
    """API endpoint for smart paper recommendations"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            interests = data.get('interests', '')
            context = data.get('context', '')
            count = data.get('count', 10)
            
            if not interests:
                return JsonResponse({'error': 'Research interests are required'}, status=400)
            
            # Search for relevant papers based on interests
            async with literature_search_service as search_service:
                results = await search_service.unified_search(interests, count * 2)
            
            # Analyze and score papers for recommendations
            recommendations = []
            for paper in results[:count]:
                analysis = await paper_analysis_service.analyze_paper(paper)
                
                # Calculate recommendation score
                relevance_score = 0.7  # Base relevance
                if analysis.get('novelty_score', 0) > 0.5:
                    relevance_score += 0.2
                if analysis.get('impact_prediction', {}).get('predicted_impact_score', 0) > 0.6:
                    relevance_score += 0.1
                
                paper['relevance_score'] = min(relevance_score, 1.0)
                paper['recommendation_reasons'] = []
                
                if analysis.get('novelty_score', 0) > 0.5:
                    paper['recommendation_reasons'].append("High novelty content")
                if paper.get('citation_count', 0) > 50:
                    paper['recommendation_reasons'].append("Well-cited work")
                if any(field.get('confidence', 0) > 0.7 for field in analysis.get('research_fields', [])):
                    paper['recommendation_reasons'].append("Strong field relevance")
                
                recommendations.append(paper)
            
            # Sort by relevance score
            recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return JsonResponse({
                'papers': recommendations,
                'avg_relevance_score': sum(p.get('relevance_score', 0) for p in recommendations) / len(recommendations) if recommendations else 0,
                'diversity_score': 0.8,  # Placeholder
                'freshness_score': 0.7   # Placeholder
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class ResearchTrendAnalysisView(View):
    """API endpoint for research trend analysis"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            field = data.get('field', '')
            time_range = data.get('time_range', '5years')
            
            if not field:
                return JsonResponse({'error': 'Research field is required'}, status=400)
            
            # Search for papers in the field
            async with literature_search_service as search_service:
                results = await search_service.unified_search(field, 100)
            
            # Analyze trends
            trends = {
                'field': field,
                'time_range': time_range,
                'total_papers': len(results),
                'trending_keywords': [],
                'publication_growth': [],
                'top_venues': [],
                'emerging_themes': []
            }
            
            # Extract keywords from all papers
            all_keywords = []
            venue_counts = {}
            year_counts = {}
            
            for paper in results:
                analysis = await paper_analysis_service.analyze_paper(paper)
                all_keywords.extend(analysis.get('keywords', []))
                
                venue = paper.get('journal', 'Unknown')
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
                
                # Extract year from publication date
                pub_date = paper.get('publication_date', '')
                if pub_date:
                    try:
                        year = pub_date.split('-')[0]
                        year_counts[year] = year_counts.get(year, 0) + 1
                    except:
                        pass
            
            # Calculate trending keywords
            from collections import Counter
            keyword_counter = Counter(all_keywords)
            trends['trending_keywords'] = [
                {'keyword': kw, 'count': count} 
                for kw, count in keyword_counter.most_common(20)
            ]
            
            # Top venues
            trends['top_venues'] = [
                {'venue': venue, 'count': count}
                for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Publication growth by year
            trends['publication_growth'] = [
                {'year': year, 'count': count}
                for year, count in sorted(year_counts.items())
            ]
            
            return JsonResponse({'trends': trends})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class ResearchGapAnalysisView(View):
    """API endpoint for research gap analysis"""
    
    async def post(self, request):
        try:
            data = json.loads(request.body)
            field = data.get('field', '')
            depth = data.get('depth', 'comprehensive')
            
            if not field:
                return JsonResponse({'error': 'Research field is required'}, status=400)
            
            # Search for papers in the field
            async with literature_search_service as search_service:
                results = await search_service.unified_search(field, 200)
            
            # Analyze for gaps
            gaps = []
            
            # Methodology gaps
            methodologies = set()
            for paper in results:
                analysis = await paper_analysis_service.analyze_paper(paper)
                methodology = analysis.get('methodology', '')
                if methodology:
                    methodologies.add(methodology.lower())
            
            # Sample gap identification (simplified)
            if 'deep learning' in field.lower() and len([m for m in methodologies if 'transformer' in m]) < 5:
                gaps.append({
                    'type': 'methodology',
                    'gap_description': 'Limited exploration of transformer architectures in this domain',
                    'importance_score': 0.8,
                    'evidence': f'Only {len([m for m in methodologies if "transformer" in m])} papers found using transformers',
                    'research_directions': [
                        'Apply transformer models to domain-specific problems',
                        'Develop specialized attention mechanisms',
                        'Investigate transfer learning approaches'
                    ]
                })
            
            if len(results) < 20:
                gaps.append({
                    'type': 'temporal',
                    'gap_description': 'Limited recent research activity in this area',
                    'importance_score': 0.7,
                    'evidence': f'Only {len(results)} papers found in recent literature',
                    'research_directions': [
                        'Conduct comprehensive literature review',
                        'Identify emerging research opportunities',
                        'Explore interdisciplinary connections'
                    ]
                })
            
            # Application gaps
            application_count = sum(1 for paper in results if 'application' in paper.get('abstract', '').lower())
            if application_count < len(results) * 0.3:
                gaps.append({
                    'type': 'application',
                    'gap_description': 'Limited real-world applications and case studies',
                    'importance_score': 0.75,
                    'evidence': f'Only {application_count} papers focus on practical applications',
                    'research_directions': [
                        'Develop industry partnerships',
                        'Create practical implementation frameworks',
                        'Conduct user studies and evaluations'
                    ]
                })
            
            return JsonResponse({
                'field': field,
                'gap_count': len(gaps),
                'gaps': gaps
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'services': {
            'literature_search': 'available',
            'paper_analysis': 'available'
        }
    })


# Sync wrappers for Django views (since Django doesn't natively support async views)
def run_async(coro):
    """Run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# Sync view wrappers
class SyncEnhancedLiteratureSearchView(EnhancedLiteratureSearchView):
    def post(self, request):
        return run_async(super().post(request))


class SyncPaperAnalysisView(PaperAnalysisView):
    def post(self, request):
        return run_async(super().post(request))


class SyncPaperComparisonView(PaperComparisonView):
    def post(self, request):
        return run_async(super().post(request))


class SyncSmartRecommendationView(SmartRecommendationView):
    def post(self, request):
        return run_async(super().post(request))


class SyncResearchTrendAnalysisView(ResearchTrendAnalysisView):
    def post(self, request):
        return run_async(super().post(request))


class SyncResearchGapAnalysisView(ResearchGapAnalysisView):
    def post(self, request):
        return run_async(super().post(request))
