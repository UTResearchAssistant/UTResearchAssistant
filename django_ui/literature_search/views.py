"""API Views for Literature Search Service."""

import asyncio
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse

from .services import literature_search_service


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_literature(request):
    """
    Perform comprehensive literature search.
    
    POST data:
    {
        "query": "machine learning",
        "sources": ["arxiv", "semantic_scholar"],
        "max_results": 50,
        "filters": {
            "date_from": "2020-01-01",
            "date_to": "2024-12-31"
        }
    }
    """
    try:
        data = request.data
        query = data.get('query', '')
        sources = data.get('sources', ['arxiv', 'semantic_scholar'])
        max_results = data.get('max_results', 50)
        filters = data.get('filters', {})
        
        if not query:
            return Response(
                {'error': 'Query is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Run async search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                literature_search_service.search(
                    query=query,
                    sources=sources,
                    max_results=max_results,
                    filters=filters,
                    user=request.user
                )
            )
        finally:
            loop.close()
        
        return Response(results, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def search_history(request):
    """Get user's search history."""
    try:
        from core.models import SearchHistory
        
        history = SearchHistory.objects.filter(
            user=request.user
        ).order_by('-created_at')[:20]
        
        history_data = []
        for search in history:
            history_data.append({
                'id': search.id,
                'query': search.query,
                'search_type': search.search_type,
                'results_count': search.results_count,
                'execution_time': search.execution_time,
                'created_at': search.created_at
            })
        
        return Response(history_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
