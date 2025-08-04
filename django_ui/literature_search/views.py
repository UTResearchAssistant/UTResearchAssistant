"""Enhanced Literature Search Views."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from django.contrib import messages

from core.models import Paper, SearchHistory, PaperBookmark, UserPreferences

logger = logging.getLogger(__name__)


def search_literature(request):
    """Main literature search interface."""
    if request.method == 'GET':
        # Get user's recent searches and preferences
        recent_searches = []
        saved_papers = []
        
        if request.user.is_authenticated:
            recent_searches = SearchHistory.objects.filter(
                user=request.user
            ).order_by('-created_at')[:10]
            
            saved_papers = PaperBookmark.objects.filter(
                user=request.user
            ).select_related('paper').order_by('-created_at')[:5]
        
        context = {
            'recent_searches': recent_searches,
            'saved_papers': saved_papers,
            'total_papers': Paper.objects.count(),
        }
        
        return render(request, 'literature_search/search.html', context)
    
    elif request.method == 'POST':
        return perform_search(request)


@csrf_exempt
@require_http_methods(["POST"])
def perform_search(request):
    """Perform literature search with filters."""
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST
        
        query = data.get('query', '').strip()
        sources = data.getlist('sources') if hasattr(data, 'getlist') else data.get('sources', [])
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        max_results = int(data.get('max_results', 20))
        
        if not query:
            return JsonResponse({
                'success': False,
                'error': 'Search query is required'
            }, status=400)
        
        # Build search filters
        search_filters = Q()
        
        # Text search in title, abstract, keywords
        search_filters |= (
            Q(title__icontains=query) |
            Q(abstract__icontains=query) |
            Q(keywords__icontains=query) |
            Q(authors__icontains=query)
        )
        
        # Date filtering
        if date_from:
            search_filters &= Q(publication_date__gte=date_from)
        if date_to:
            search_filters &= Q(publication_date__lte=date_to)
        
        # Source filtering
        if sources:
            search_filters &= Q(source__in=sources)
        
        # Perform search
        papers = Paper.objects.filter(search_filters).order_by(
            '-relevance_score', '-citation_count', '-publication_date'
        )[:max_results]
        
        # Save search history for authenticated users
        if request.user.is_authenticated:
            SearchHistory.objects.create(
                user=request.user,
                query=query,
                filters=json.dumps({
                    'sources': sources,
                    'date_from': date_from,
                    'date_to': date_to,
                    'max_results': max_results
                }),
                result_count=papers.count()
            )
        
        # Format results
        results = []
        for paper in papers:
            authors_list = []
            try:
                authors_list = json.loads(paper.authors) if paper.authors else []
            except (json.JSONDecodeError, TypeError):
                authors_list = [paper.authors] if paper.authors else []
            
            results.append({
                'id': str(paper.id),
                'title': paper.title,
                'abstract': paper.abstract[:500] + '...' if len(paper.abstract) > 500 else paper.abstract,
                'authors': authors_list,
                'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
                'journal': paper.journal,
                'citation_count': paper.citation_count,
                'doi': paper.doi,
                'pdf_url': paper.pdf_url,
                'external_url': paper.external_url,
                'source': paper.source,
                'is_bookmarked': paper.bookmarks.filter(user=request.user).exists() if request.user.is_authenticated else False
            })
        
        return JsonResponse({
            'success': True,
            'results': results,
            'total_count': papers.count(),
            'query': query,
            'filters': {
                'sources': sources,
                'date_from': date_from,
                'date_to': date_to
            }
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during search'
        }, status=500)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def bookmark_paper(request):
    """Bookmark/unbookmark a paper."""
    try:
        data = json.loads(request.body)
        paper_id = data.get('paper_id')
        action = data.get('action', 'toggle')  # toggle, add, remove
        
        paper = Paper.objects.get(id=paper_id)
        bookmark, created = PaperBookmark.objects.get_or_create(
            user=request.user,
            paper=paper
        )
        
        if action == 'remove' or (action == 'toggle' and not created):
            bookmark.delete()
            bookmarked = False
        else:
            bookmarked = True
        
        return JsonResponse({
            'success': True,
            'bookmarked': bookmarked,
            'paper_id': paper_id
        })
        
    except Paper.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Paper not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Bookmark error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)


@login_required
def search_history(request):
    """Get user's search history."""
    searches = SearchHistory.objects.filter(
        user=request.user
    ).order_by('-created_at')
    
    paginator = Paginator(searches, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    if request.headers.get('Accept') == 'application/json':
        history_data = []
        for search in page_obj:
            filters = {}
            try:
                filters = json.loads(search.filters) if search.filters else {}
            except json.JSONDecodeError:
                pass
                
            history_data.append({
                'id': str(search.id),
                'query': search.query,
                'filters': filters,
                'result_count': search.result_count,
                'created_at': search.created_at.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'history': history_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'page_number': page_obj.number,
            'total_pages': paginator.num_pages
        })
    
    return render(request, 'literature_search/history.html', {
        'page_obj': page_obj
    })


@login_required
def bookmarks(request):
    """Get user's bookmarked papers."""
    bookmarks_qs = PaperBookmark.objects.filter(
        user=request.user
    ).select_related('paper').order_by('-created_at')
    
    paginator = Paginator(bookmarks_qs, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    if request.headers.get('Accept') == 'application/json':
        bookmarks_data = []
        for bookmark in page_obj:
            paper = bookmark.paper
            authors_list = []
            try:
                authors_list = json.loads(paper.authors) if paper.authors else []
            except (json.JSONDecodeError, TypeError):
                authors_list = [paper.authors] if paper.authors else []
            
            bookmarks_data.append({
                'id': str(bookmark.id),
                'paper': {
                    'id': str(paper.id),
                    'title': paper.title,
                    'authors': authors_list,
                    'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
                    'journal': paper.journal,
                    'citation_count': paper.citation_count,
                    'doi': paper.doi
                },
                'created_at': bookmark.created_at.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'bookmarks': bookmarks_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'page_number': page_obj.number,
            'total_pages': paginator.num_pages
        })
    
    return render(request, 'literature_search/bookmarks.html', {
        'page_obj': page_obj
    })


def paper_detail(request, paper_id):
    """Get detailed information about a specific paper."""
    try:
        paper = Paper.objects.get(id=paper_id)
        
        # Parse JSON fields
        authors_list = []
        keywords_list = []
        categories_list = []
        
        try:
            authors_list = json.loads(paper.authors) if paper.authors else []
        except (json.JSONDecodeError, TypeError):
            authors_list = [paper.authors] if paper.authors else []
        
        try:
            keywords_list = json.loads(paper.keywords) if paper.keywords else []
        except (json.JSONDecodeError, TypeError):
            keywords_list = paper.keywords.split(',') if paper.keywords else []
        
        try:
            categories_list = json.loads(paper.subject_categories) if paper.subject_categories else []
        except (json.JSONDecodeError, TypeError):
            categories_list = paper.subject_categories.split(',') if paper.subject_categories else []
        
        # Get related papers (simple implementation based on keywords)
        related_papers = []
        if keywords_list:
            related_papers = Paper.objects.filter(
                Q(keywords__icontains=keywords_list[0]) |
                Q(title__icontains=keywords_list[0])
            ).exclude(id=paper.id)[:5]
        
        paper_data = {
            'id': str(paper.id),
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': authors_list,
            'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
            'journal': paper.journal,
            'volume': paper.volume,
            'issue': paper.issue,
            'pages': paper.pages,
            'citation_count': paper.citation_count,
            'doi': paper.doi,
            'arxiv_id': paper.arxiv_id,
            'pubmed_id': paper.pubmed_id,
            'pdf_url': paper.pdf_url,
            'external_url': paper.external_url,
            'keywords': keywords_list,
            'subject_categories': categories_list,
            'language': paper.language,
            'is_open_access': paper.is_open_access,
            'source': paper.source,
            'quality_score': paper.quality_score,
            'summary': paper.summary,
            'is_bookmarked': paper.bookmarks.filter(user=request.user).exists() if request.user.is_authenticated else False
        }
        
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': True,
                'paper': paper_data,
                'related_papers': [
                    {
                        'id': str(p.id),
                        'title': p.title,
                        'authors': json.loads(p.authors) if p.authors else [],
                        'publication_date': p.publication_date.isoformat() if p.publication_date else None,
                        'citation_count': p.citation_count
                    } for p in related_papers
                ]
            })
        
        return render(request, 'literature_search/paper_detail.html', {
            'paper': paper_data,
            'related_papers': related_papers
        })
        
    except Paper.DoesNotExist:
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({
                'success': False,
                'error': 'Paper not found'
            }, status=404)
        
        messages.error(request, 'Paper not found')
        return redirect('literature_search:search')


@login_required
@csrf_exempt
@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def clear_search_history(request):
    """Clear user's search history."""
    try:
        deleted_count = SearchHistory.objects.filter(user=request.user).delete()[0]
        
        return JsonResponse({
            'success': True,
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)
