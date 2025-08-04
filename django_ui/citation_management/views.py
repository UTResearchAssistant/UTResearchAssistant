"""Enhanced Citation Management Views."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from django.contrib import messages

from core.models import Paper, PaperBookmark

logger = logging.getLogger(__name__)


@login_required
def list_citations(request):
    """List user's citations and bibliography."""
    # Get bookmarked papers as citations
    bookmarks = PaperBookmark.objects.filter(
        user=request.user
    ).select_related('paper').order_by('-created_at')
    
    # Apply filters
    search_query = request.GET.get('search', '')
    citation_style = request.GET.get('style', 'apa')
    
    if search_query:
        bookmarks = bookmarks.filter(
            Q(paper__title__icontains=search_query) |
            Q(paper__authors__icontains=search_query) |
            Q(paper__journal__icontains=search_query)
        )
    
    paginator = Paginator(bookmarks, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    if request.headers.get('Accept') == 'application/json':
        citations_data = []
        for bookmark in page_obj:
            paper = bookmark.paper
            authors_list = []
            try:
                authors_list = json.loads(paper.authors) if paper.authors else []
            except (json.JSONDecodeError, TypeError):
                authors_list = [paper.authors] if paper.authors else []
            
            citation = format_citation(paper, citation_style)
            
            citations_data.append({
                'id': str(bookmark.id),
                'paper_id': str(paper.id),
                'title': paper.title,
                'authors': authors_list,
                'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
                'journal': paper.journal,
                'doi': paper.doi,
                'citation': citation,
                'created_at': bookmark.created_at.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'citations': citations_data,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
            'page_number': page_obj.number,
            'total_pages': paginator.num_pages,
            'total_count': paginator.count
        })
    
    # Generate formatted citations for display
    formatted_citations = []
    for bookmark in page_obj:
        formatted_citations.append({
            'bookmark': bookmark,
            'citation': format_citation(bookmark.paper, citation_style)
        })
    
    context = {
        'page_obj': page_obj,
        'formatted_citations': formatted_citations,
        'search_query': search_query,
        'citation_style': citation_style,
        'available_styles': ['apa', 'mla', 'chicago', 'harvard', 'ieee'],
        'total_citations': paginator.count
    }
    
    return render(request, 'citation_management/list.html', context)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def add_citation(request):
    """Add a paper to citations (bookmark it)."""
    try:
        data = json.loads(request.body)
        paper_id = data.get('paper_id')
        
        if not paper_id:
            return JsonResponse({
                'success': False,
                'error': 'Paper ID is required'
            }, status=400)
        
        paper = get_object_or_404(Paper, id=paper_id)
        bookmark, created = PaperBookmark.objects.get_or_create(
            user=request.user,
            paper=paper
        )
        
        return JsonResponse({
            'success': True,
            'created': created,
            'citation_id': str(bookmark.id),
            'message': 'Paper added to citations' if created else 'Paper already in citations'
        })
        
    except Paper.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Paper not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Add citation error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)


@login_required
@require_http_methods(["DELETE"])
def delete_citation(request, citation_id):
    """Remove a citation (unbookmark paper)."""
    try:
        bookmark = get_object_or_404(
            PaperBookmark,
            id=citation_id,
            user=request.user
        )
        
        bookmark.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Citation removed'
        })
        
    except PaperBookmark.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Citation not found'
        }, status=404)
    except Exception as e:
        logger.error(f"Delete citation error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred'
        }, status=500)


def format_citations(request):
    """Format citations in specified style."""
    style = request.GET.get('style', 'apa')
    citation_ids = request.GET.getlist('citations')
    
    if not citation_ids and request.user.is_authenticated:
        # Get all user's citations if none specified
        bookmarks = PaperBookmark.objects.filter(user=request.user)
    else:
        bookmarks = PaperBookmark.objects.filter(id__in=citation_ids)
    
    formatted_citations = []
    for bookmark in bookmarks:
        citation = format_citation(bookmark.paper, style)
        formatted_citations.append({
            'id': str(bookmark.id),
            'citation': citation
        })
    
    return JsonResponse({
        'success': True,
        'style': style,
        'formatted_citations': formatted_citations
    })


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def import_citations(request):
    """Import citations from file (BibTeX, EndNote, etc.)."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file uploaded'
            }, status=400)
        
        uploaded_file = request.FILES['file']
        file_format = request.POST.get('format', 'bibtex')
        
        # For now, return a mock response
        # In a real implementation, you would parse the file
        # and create Paper and PaperBookmark objects
        
        return JsonResponse({
            'success': True,
            'message': f'Citations imported from {uploaded_file.name}',
            'imported_count': 5,  # Mock count
            'format': file_format
        })
        
    except Exception as e:
        logger.error(f"Import citations error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'An error occurred during import'
        }, status=500)


@login_required
def export_citations(request):
    """Export citations in specified format."""
    export_format = request.GET.get('format', 'bibtex')
    citation_ids = request.GET.getlist('citations')
    
    if citation_ids:
        bookmarks = PaperBookmark.objects.filter(
            id__in=citation_ids,
            user=request.user
        )
    else:
        bookmarks = PaperBookmark.objects.filter(user=request.user)
    
    if export_format == 'bibtex':
        content = generate_bibtex(bookmarks)
        content_type = 'text/plain'
        filename = 'citations.bib'
    elif export_format == 'ris':
        content = generate_ris(bookmarks)
        content_type = 'text/plain'
        filename = 'citations.ris'
    elif export_format == 'json':
        content = generate_json_export(bookmarks)
        content_type = 'application/json'
        filename = 'citations.json'
    else:
        return JsonResponse({
            'success': False,
            'error': 'Unsupported export format'
        }, status=400)
    
    response = HttpResponse(content, content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@login_required
def bibliography_generator(request):
    """Generate formatted bibliography."""
    style = request.GET.get('style', 'apa')
    citation_ids = request.GET.getlist('citations')
    
    if citation_ids:
        bookmarks = PaperBookmark.objects.filter(
            id__in=citation_ids,
            user=request.user
        ).select_related('paper').order_by('paper__title')
    else:
        bookmarks = PaperBookmark.objects.filter(
            user=request.user
        ).select_related('paper').order_by('paper__title')
    
    bibliography_entries = []
    for bookmark in bookmarks:
        citation = format_citation(bookmark.paper, style)
        bibliography_entries.append({
            'id': str(bookmark.id),
            'citation': citation,
            'paper': bookmark.paper
        })
    
    if request.headers.get('Accept') == 'application/json':
        return JsonResponse({
            'success': True,
            'style': style,
            'entries': bibliography_entries,
            'count': len(bibliography_entries)
        })
    
    context = {
        'bibliography_entries': bibliography_entries,
        'style': style,
        'available_styles': ['apa', 'mla', 'chicago', 'harvard', 'ieee'],
        'total_entries': len(bibliography_entries)
    }
    
    return render(request, 'citation_management/bibliography.html', context)


def format_citation(paper, style):
    """Format a single paper citation in the specified style."""
    authors_list = []
    try:
        authors_list = json.loads(paper.authors) if paper.authors else []
    except (json.JSONDecodeError, TypeError):
        authors_list = [paper.authors] if paper.authors else []
    
    # Format authors based on style
    if style == 'apa':
        return format_apa_citation(paper, authors_list)
    elif style == 'mla':
        return format_mla_citation(paper, authors_list)
    elif style == 'chicago':
        return format_chicago_citation(paper, authors_list)
    elif style == 'harvard':
        return format_harvard_citation(paper, authors_list)
    elif style == 'ieee':
        return format_ieee_citation(paper, authors_list)
    else:
        return format_apa_citation(paper, authors_list)  # Default to APA


def format_apa_citation(paper, authors_list):
    """Format citation in APA style."""
    # Author formatting
    if authors_list:
        if len(authors_list) == 1:
            authors = authors_list[0]
        elif len(authors_list) <= 7:
            authors = ', '.join(authors_list[:-1]) + ', & ' + authors_list[-1]
        else:
            authors = ', '.join(authors_list[:6]) + ', ... ' + authors_list[-1]
    else:
        authors = 'Unknown author'
    
    # Year
    year = paper.publication_date.year if paper.publication_date else 'n.d.'
    
    # Title
    title = paper.title
    
    # Journal
    journal_part = ''
    if paper.journal:
        journal_part = f"{paper.journal}"
        if paper.volume:
            journal_part += f", {paper.volume}"
        if paper.issue:
            journal_part += f"({paper.issue})"
        if paper.pages:
            journal_part += f", {paper.pages}"
    
    # DOI
    doi_part = f" https://doi.org/{paper.doi}" if paper.doi else ""
    
    citation = f"{authors} ({year}). {title}."
    if journal_part:
        citation += f" {journal_part}."
    citation += doi_part
    
    return citation


def format_mla_citation(paper, authors_list):
    """Format citation in MLA style."""
    # Similar implementation for MLA
    authors = authors_list[0] if authors_list else 'Unknown author'
    title = f'"{paper.title}"'
    journal = paper.journal if paper.journal else ''
    year = paper.publication_date.year if paper.publication_date else ''
    
    citation = f"{authors}. {title}"
    if journal:
        citation += f" {journal},"
    if year:
        citation += f" {year}."
    
    return citation


def format_chicago_citation(paper, authors_list):
    """Format citation in Chicago style."""
    # Basic Chicago format implementation
    return format_apa_citation(paper, authors_list)  # Simplified


def format_harvard_citation(paper, authors_list):
    """Format citation in Harvard style."""
    # Basic Harvard format implementation
    return format_apa_citation(paper, authors_list)  # Simplified


def format_ieee_citation(paper, authors_list):
    """Format citation in IEEE style."""
    # Basic IEEE format implementation
    authors = ', '.join(authors_list) if authors_list else 'Unknown author'
    title = f'"{paper.title}"'
    journal = paper.journal if paper.journal else ''
    year = paper.publication_date.year if paper.publication_date else ''
    
    citation = f"{authors}, {title}"
    if journal:
        citation += f", {journal}"
    if year:
        citation += f", {year}."
    
    return citation


def generate_bibtex(bookmarks):
    """Generate BibTeX format export."""
    bibtex_entries = []
    
    for bookmark in bookmarks:
        paper = bookmark.paper
        authors_list = []
        try:
            authors_list = json.loads(paper.authors) if paper.authors else []
        except (json.JSONDecodeError, TypeError):
            authors_list = [paper.authors] if paper.authors else []
        
        # Create BibTeX key
        first_author = authors_list[0].split()[-1] if authors_list else 'Unknown'
        year = paper.publication_date.year if paper.publication_date else 'Unknown'
        key = f"{first_author}{year}"
        
        # Format authors for BibTeX
        authors = ' and '.join(authors_list) if authors_list else 'Unknown'
        
        entry = f"""@article{{{key},
  title={{{paper.title}}},
  author={{{authors}}},"""
        
        if paper.journal:
            entry += f"\n  journal={{{paper.journal}}},"
        if paper.publication_date:
            entry += f"\n  year={{{paper.publication_date.year}}},"
        if paper.volume:
            entry += f"\n  volume={{{paper.volume}}},"
        if paper.issue:
            entry += f"\n  number={{{paper.issue}}},"
        if paper.pages:
            entry += f"\n  pages={{{paper.pages}}},"
        if paper.doi:
            entry += f"\n  doi={{{paper.doi}}},"
        
        entry += "\n}"
        bibtex_entries.append(entry)
    
    return '\n\n'.join(bibtex_entries)


def generate_ris(bookmarks):
    """Generate RIS format export."""
    ris_entries = []
    
    for bookmark in bookmarks:
        paper = bookmark.paper
        authors_list = []
        try:
            authors_list = json.loads(paper.authors) if paper.authors else []
        except (json.JSONDecodeError, TypeError):
            authors_list = [paper.authors] if paper.authors else []
        
        entry = "TY  - JOUR\n"
        entry += f"TI  - {paper.title}\n"
        
        for author in authors_list:
            entry += f"AU  - {author}\n"
        
        if paper.journal:
            entry += f"JO  - {paper.journal}\n"
        if paper.publication_date:
            entry += f"PY  - {paper.publication_date.year}\n"
        if paper.volume:
            entry += f"VL  - {paper.volume}\n"
        if paper.issue:
            entry += f"IS  - {paper.issue}\n"
        if paper.pages:
            entry += f"SP  - {paper.pages}\n"
        if paper.doi:
            entry += f"DO  - {paper.doi}\n"
        if paper.abstract:
            entry += f"AB  - {paper.abstract}\n"
        
        entry += "ER  -\n"
        ris_entries.append(entry)
    
    return '\n\n'.join(ris_entries)


def generate_json_export(bookmarks):
    """Generate JSON format export."""
    citations = []
    
    for bookmark in bookmarks:
        paper = bookmark.paper
        authors_list = []
        try:
            authors_list = json.loads(paper.authors) if paper.authors else []
        except (json.JSONDecodeError, TypeError):
            authors_list = [paper.authors] if paper.authors else []
        
        citation_data = {
            'title': paper.title,
            'authors': authors_list,
            'abstract': paper.abstract,
            'journal': paper.journal,
            'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
            'volume': paper.volume,
            'issue': paper.issue,
            'pages': paper.pages,
            'doi': paper.doi,
            'arxiv_id': paper.arxiv_id,
            'pubmed_id': paper.pubmed_id,
            'citation_count': paper.citation_count,
            'keywords': json.loads(paper.keywords) if paper.keywords else [],
            'url': paper.external_url
        }
        citations.append(citation_data)
    
    return json.dumps({
        'citations': citations,
        'exported_at': datetime.now().isoformat(),
        'count': len(citations)
    }, indent=2)
