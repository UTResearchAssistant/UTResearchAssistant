"""Crawler for research papers.

This module defines functions to discover and download new papers from
sources such as arXiv, conference websites, and academic databases.
Includes comprehensive search and filtering capabilities.
"""

import logging
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import time
import re

logger = logging.getLogger(__name__)


def crawl_arxiv(category: str, max_results: int = 100, days_back: int = 7) -> List[Dict]:
    """Fetch a list of new papers from arXiv for a given category.

    Parameters
    ----------
    category : str
        The arXiv category code (e.g. 'cs.AI', 'cs.LG').
    max_results : int
        Maximum number of papers to return
    days_back : int
        Number of days to look back for new papers

    Returns
    -------
    list[dict]
        A list of paper metadata dictionaries.
    """
    try:
        import arxiv
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Search arXiv
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in search.results():
            # Filter by date if needed
            if result.published >= start_date:
                paper_data = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "arxiv_id": result.get_short_id(),
                    "source": "arxiv",
                    "category": category
                }
                papers.append(paper_data)
        
        logger.info(f"Found {len(papers)} papers from arXiv category {category}")
        return papers
        
    except ImportError:
        logger.warning("arxiv package not available, using fallback")
        return _fallback_arxiv_crawl(category, max_results)
    except Exception as e:
        logger.error(f"Error crawling arXiv: {e}")
        return []


def crawl_papers(query: str, sources: List[str] = None, max_results: int = 50) -> List[Dict]:
    """Crawl multiple sources for papers matching a query.
    
    Parameters
    ----------
    query : str
        Search query for papers
    sources : list[str], optional
        List of sources to search. Defaults to ['arxiv', 'semantic_scholar']
    max_results : int
        Maximum total results to return
    
    Returns
    -------
    list[dict]
        List of paper metadata from all sources
    """
    if sources is None:
        sources = ['arxiv', 'semantic_scholar']
    
    all_papers = []
    results_per_source = max_results // len(sources)
    
    for source in sources:
        try:
            if source == 'arxiv':
                papers = crawl_arxiv_by_query(query, results_per_source)
            elif source == 'semantic_scholar':
                papers = crawl_semantic_scholar(query, results_per_source)
            elif source == 'pubmed':
                papers = crawl_pubmed(query, results_per_source)
            else:
                logger.warning(f"Unknown source: {source}")
                continue
            
            all_papers.extend(papers)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error crawling {source}: {e}")
    
    # Remove duplicates based on title similarity
    unique_papers = _remove_duplicate_papers(all_papers)
    
    return unique_papers[:max_results]


def crawl_arxiv_by_query(query: str, max_results: int = 50) -> List[Dict]:
    """Search arXiv with a specific query string.
    
    Parameters
    ----------
    query : str
        Search query
    max_results : int
        Maximum results to return
    
    Returns
    -------
    list[dict]
        List of paper metadata
    """
    try:
        import arxiv
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            paper_data = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "arxiv_id": result.get_short_id(),
                "source": "arxiv"
            }
            papers.append(paper_data)
        
        return papers
        
    except ImportError:
        return _fallback_arxiv_crawl("cs.AI", max_results)
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        return []


def crawl_semantic_scholar(query: str, max_results: int = 50) -> List[Dict]:
    """Search Semantic Scholar for papers.
    
    Parameters
    ----------
    query : str
        Search query
    max_results : int
        Maximum results to return
    
    Returns
    -------
    list[dict]
        List of paper metadata
    """
    try:
        import requests
        
        # Semantic Scholar API
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,abstract,year,url,citationCount,publicationDate"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        papers = []
        
        for paper in data.get('data', []):
            paper_data = {
                "title": paper.get('title', ''),
                "authors": [author.get('name', '') for author in paper.get('authors', [])],
                "summary": paper.get('abstract', ''),
                "published": paper.get('publicationDate', ''),
                "year": paper.get('year'),
                "url": paper.get('url', ''),
                "citation_count": paper.get('citationCount', 0),
                "source": "semantic_scholar",
                "paper_id": paper.get('paperId', '')
            }
            papers.append(paper_data)
        
        logger.info(f"Found {len(papers)} papers from Semantic Scholar")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        return []


def crawl_pubmed(query: str, max_results: int = 50) -> List[Dict]:
    """Search PubMed for papers.
    
    Parameters
    ----------
    query : str
        Search query
    max_results : int
        Maximum results to return
    
    Returns
    -------
    list[dict]
        List of paper metadata
    """
    try:
        import requests
        from xml.etree import ElementTree as ET
        
        # PubMed E-utilities
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 100),
            "retmode": "xml"
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        
        # Parse search results
        search_root = ET.fromstring(search_response.content)
        id_list = [id_elem.text for id_elem in search_root.findall('.//Id')]
        
        if not id_list:
            return []
        
        # Fetch paper details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse paper details
        fetch_root = ET.fromstring(fetch_response.content)
        papers = []
        
        for article in fetch_root.findall('.//PubmedArticle'):
            try:
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else ''
                
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                # Extract authors
                authors = []
                for author_elem in article.findall('.//Author'):
                    fname = author_elem.find('.//ForeName')
                    lname = author_elem.find('.//LastName')
                    if fname is not None and lname is not None:
                        authors.append(f"{fname.text} {lname.text}")
                
                # Extract publication date
                pub_date_elem = article.find('.//PubDate')
                pub_date = ''
                if pub_date_elem is not None:
                    year_elem = pub_date_elem.find('.//Year')
                    if year_elem is not None:
                        pub_date = year_elem.text
                
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ''
                
                paper_data = {
                    "title": title,
                    "authors": authors,
                    "summary": abstract,
                    "published": pub_date,
                    "pmid": pmid,
                    "source": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ''
                }
                papers.append(paper_data)
                
            except Exception as e:
                logger.warning(f"Error parsing PubMed article: {e}")
                continue
        
        logger.info(f"Found {len(papers)} papers from PubMed")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching PubMed: {e}")
        return []


def _fallback_arxiv_crawl(category: str, max_results: int) -> List[Dict]:
    """Fallback method when arxiv package is not available"""
    logger.info(f"Fallback: Would fetch {max_results} papers from arXiv category: {category}")
    
    # Return mock data for demonstration
    return [{
        "title": f"Mock Paper from {category}",
        "authors": ["Mock Author"],
        "summary": "This is a mock paper for demonstration purposes.",
        "published": datetime.now().isoformat(),
        "categories": [category],
        "pdf_url": "https://arxiv.org/pdf/mock.pdf",
        "source": "arxiv_fallback"
    }]


def _remove_duplicate_papers(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers based on title similarity.
    
    Parameters
    ----------
    papers : list[dict]
        List of papers to deduplicate
    
    Returns
    -------
    list[dict]
        Deduplicated list of papers
    """
    if not papers:
        return papers
    
    unique_papers = []
    seen_titles = set()
    
    for paper in papers:
        title = paper.get('title', '').lower().strip()
        
        # Simple title normalization for duplicate detection
        normalized_title = re.sub(r'[^\w\s]', '', title)
        normalized_title = re.sub(r'\s+', ' ', normalized_title)
        
        if normalized_title not in seen_titles and len(normalized_title) > 10:
            seen_titles.add(normalized_title)
            unique_papers.append(paper)
    
    return unique_papers


def get_trending_papers(days: int = 7, sources: List[str] = None) -> List[Dict]:
    """Get trending papers from the last few days.
    
    Parameters
    ----------
    days : int
        Number of days to look back
    sources : list[str], optional
        Sources to search
    
    Returns
    -------
    list[dict]
        Trending papers
    """
    if sources is None:
        sources = ['arxiv']
    
    trending_papers = []
    
    for source in sources:
        if source == 'arxiv':
            # Get papers from popular CS categories
            categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
            for category in categories:
                papers = crawl_arxiv(category, max_results=20, days_back=days)
                trending_papers.extend(papers)
    
    # Sort by publication date (most recent first)
    trending_papers.sort(key=lambda x: x.get('published', ''), reverse=True)
    
    return trending_papers[:50]
