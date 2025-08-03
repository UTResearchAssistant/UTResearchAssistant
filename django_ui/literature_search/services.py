"""Literature Search Service with AI-powered search capabilities.

This service provides comprehensive literature search across multiple academic
databases including arXiv, Google Scholar, Semantic Scholar, and PubMed.
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import re

import requests
import arxiv
from scholarly import scholarly
import openai
from django.conf import settings
from django.core.cache import cache

from core.models import Paper, SearchHistory

logger = logging.getLogger(__name__)


class LiteratureSearchService:
    """AI-powered literature search service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.search_sources = {
            'arxiv': self._search_arxiv,
            'semantic_scholar': self._search_semantic_scholar,
            'pubmed': self._search_pubmed,
            'google_scholar': self._search_google_scholar,
        }
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 50,
        filters: Dict = None,
        user=None
    ) -> Dict:
        """
        Perform comprehensive literature search across multiple sources.
        
        Args:
            query: Search query
            sources: List of sources to search (arxiv, semantic_scholar, etc.)
            max_results: Maximum number of results to return
            filters: Additional filters (date_range, authors, journals, etc.)
            user: User making the search request
            
        Returns:
            Dictionary containing search results and metadata
        """
        if sources is None:
            sources = ['arxiv', 'semantic_scholar']
        
        if filters is None:
            filters = {}
        
        # Enhance query with AI if needed
        enhanced_query = await self._enhance_query_with_ai(query)
        
        # Track search history
        search_start = datetime.now()
        
        # Search across all sources
        all_results = []
        source_stats = {}
        
        for source in sources:
            if source in self.search_sources:
                try:
                    logger.info(f"Searching {source} for: {enhanced_query}")
                    results = await self.search_sources[source](
                        enhanced_query, max_results // len(sources), filters
                    )
                    all_results.extend(results)
                    source_stats[source] = len(results)
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
                    source_stats[source] = 0
        
        # Remove duplicates and rank results
        unique_results = self._remove_duplicates(all_results)
        ranked_results = await self._rank_results_with_ai(unique_results, query)
        
        # Limit results
        final_results = ranked_results[:max_results]
        
        # Save papers to database
        saved_papers = []
        for result in final_results:
            paper = await self._save_paper_to_db(result)
            if paper:
                saved_papers.append(paper)
        
        # Track search history
        search_time = (datetime.now() - search_start).total_seconds()
        if user:
            SearchHistory.objects.create(
                user=user,
                query=query,
                search_type='literature',
                sources=json.dumps(sources),
                filters=json.dumps(filters),
                results_count=len(final_results),
                execution_time=search_time
            )
        
        return {
            'query': query,
            'enhanced_query': enhanced_query,
            'results': final_results,
            'total_results': len(final_results),
            'source_stats': source_stats,
            'execution_time': search_time,
            'papers_saved': len(saved_papers)
        }
    
    async def _enhance_query_with_ai(self, query: str) -> str:
        """Enhance search query using AI to improve search results."""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic search assistant. Enhance the given search query to improve academic literature search results. Add relevant synonyms, alternative terms, and academic keywords. Keep it concise but comprehensive."
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this academic search query: {query}"
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            enhanced = response.choices[0].message.content.strip()
            logger.info(f"Enhanced query: {query} -> {enhanced}")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing query with AI: {e}")
            return query
    
    async def _search_arxiv(self, query: str, max_results: int, filters: Dict) -> List[Dict]:
        """Search arXiv papers."""
        try:
            # Build arXiv search query
            search_query = query
            
            # Add date filters if specified
            if 'date_from' in filters:
                # arXiv doesn't support date filters in query, we'll filter later
                pass
            
            # Search arXiv
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # Apply date filter
                if 'date_from' in filters:
                    if paper.published < datetime.strptime(filters['date_from'], '%Y-%m-%d'):
                        continue
                
                results.append({
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [str(author) for author in paper.authors],
                    'arxiv_id': paper.entry_id.split('/')[-1],
                    'publication_date': paper.published.date(),
                    'pdf_url': paper.pdf_url,
                    'external_url': paper.entry_id,
                    'source': 'arxiv',
                    'categories': paper.categories,
                    'journal': getattr(paper, 'journal_ref', ''),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str, max_results: int, filters: Dict) -> List[Dict]:
        """Search Semantic Scholar papers."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': min(max_results, 100),  # API limit
                'fields': 'title,abstract,authors,year,journal,url,citationCount,externalIds'
            }
            
            # Add date filters
            if 'date_from' in filters:
                params['year'] = f"{filters['date_from'][:4]}-"
            
            response = await asyncio.to_thread(requests.get, url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for paper in data.get('data', []):
                if not paper.get('title'):
                    continue
                
                results.append({
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'publication_date': paper.get('year'),
                    'journal': paper.get('journal', {}).get('name', ''),
                    'external_url': paper.get('url', ''),
                    'citation_count': paper.get('citationCount', 0),
                    'source': 'semantic_scholar',
                    'doi': paper.get('externalIds', {}).get('DOI'),
                    'arxiv_id': paper.get('externalIds', {}).get('ArXiv'),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    async def _search_pubmed(self, query: str, max_results: int, filters: Dict) -> List[Dict]:
        """Search PubMed papers."""
        try:
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }
            
            # Add date filters
            if 'date_from' in filters:
                search_params['mindate'] = filters['date_from']
            if 'date_to' in filters:
                search_params['maxdate'] = filters['date_to']
            
            search_response = await asyncio.to_thread(
                requests.get, f"{base_url}esearch.fcgi", params=search_params
            )
            search_data = search_response.json()
            
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return []
            
            # Fetch paper details
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            fetch_response = await asyncio.to_thread(
                requests.get, f"{base_url}efetch.fcgi", params=fetch_params
            )
            
            # Parse XML response (simplified)
            # In a real implementation, you'd use xml.etree.ElementTree
            results = []
            for pmid in pmids:
                results.append({
                    'title': f"PubMed Paper {pmid}",  # Would extract from XML
                    'abstract': "Abstract would be extracted from XML",
                    'authors': ["Authors from XML"],
                    'pubmed_id': pmid,
                    'source': 'pubmed',
                    'external_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def _search_google_scholar(self, query: str, max_results: int, filters: Dict) -> List[Dict]:
        """Search Google Scholar papers."""
        try:
            # Use scholarly library (be careful with rate limiting)
            search_query = scholarly.search_pubs(query)
            
            results = []
            count = 0
            
            for paper in search_query:
                if count >= max_results:
                    break
                
                try:
                    filled_paper = scholarly.fill(paper)
                    
                    results.append({
                        'title': filled_paper.get('title', ''),
                        'abstract': filled_paper.get('abstract', ''),
                        'authors': filled_paper.get('author', []),
                        'publication_date': filled_paper.get('year'),
                        'journal': filled_paper.get('journal', ''),
                        'citation_count': filled_paper.get('num_citations', 0),
                        'external_url': filled_paper.get('pub_url', ''),
                        'source': 'google_scholar',
                    })
                    
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing Google Scholar paper: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []
    
    async def _rank_results_with_ai(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Rank search results using AI for relevance."""
        try:
            if not results:
                return results
            
            # Prepare papers for ranking
            papers_text = []
            for i, paper in enumerate(results):
                text = f"Paper {i}: {paper.get('title', '')} - {paper.get('abstract', '')[:200]}"
                papers_text.append(text)
            
            # Use AI to rank papers
            ranking_prompt = f"""
            Given the search query: "{original_query}"
            
            Rank the following papers by relevance (most relevant first).
            Return only a comma-separated list of paper numbers (0-{len(results)-1}).
            
            Papers:
            {chr(10).join(papers_text)}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert academic researcher. Rank papers by relevance to the given query."},
                    {"role": "user", "content": ranking_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            ranking_text = response.choices[0].message.content.strip()
            
            # Parse ranking
            try:
                rankings = [int(x.strip()) for x in ranking_text.split(',')]
                ranked_results = [results[i] for i in rankings if 0 <= i < len(results)]
                
                # Add any missing papers at the end
                included_indices = set(rankings)
                for i, paper in enumerate(results):
                    if i not in included_indices:
                        ranked_results.append(paper)
                
                return ranked_results
                
            except Exception as e:
                logger.error(f"Error parsing AI ranking: {e}")
                return results
            
        except Exception as e:
            logger.error(f"Error ranking results with AI: {e}")
            return results
    
    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title, DOI, or arXiv ID."""
        seen = set()
        unique_papers = []
        
        for paper in papers:
            # Create identifier for deduplication
            identifiers = []
            
            if paper.get('doi'):
                identifiers.append(('doi', paper['doi'].lower()))
            if paper.get('arxiv_id'):
                identifiers.append(('arxiv', paper['arxiv_id'].lower()))
            if paper.get('title'):
                # Normalize title
                title = re.sub(r'\s+', ' ', paper['title'].lower().strip())
                identifiers.append(('title', title))
            
            # Check if we've seen this paper before
            is_duplicate = False
            for id_type, id_value in identifiers:
                if (id_type, id_value) in seen:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Add all identifiers to seen set
                for id_type, id_value in identifiers:
                    seen.add((id_type, id_value))
                unique_papers.append(paper)
        
        return unique_papers
    
    async def _save_paper_to_db(self, paper_data: Dict) -> Optional[Paper]:
        """Save paper to database."""
        try:
            # Check if paper already exists
            existing_paper = None
            
            if paper_data.get('doi'):
                existing_paper = Paper.objects.filter(doi=paper_data['doi']).first()
            elif paper_data.get('arxiv_id'):
                existing_paper = Paper.objects.filter(arxiv_id=paper_data['arxiv_id']).first()
            
            if existing_paper:
                return existing_paper
            
            # Create new paper
            paper = Paper.objects.create(
                title=paper_data.get('title', '')[:500],
                abstract=paper_data.get('abstract', ''),
                authors=json.dumps(paper_data.get('authors', [])),
                doi=paper_data.get('doi'),
                arxiv_id=paper_data.get('arxiv_id'),
                pubmed_id=paper_data.get('pubmed_id'),
                publication_date=paper_data.get('publication_date'),
                journal=paper_data.get('journal', '')[:200],
                pdf_url=paper_data.get('pdf_url', ''),
                external_url=paper_data.get('external_url', ''),
                citation_count=paper_data.get('citation_count', 0),
                source=paper_data.get('source', 'unknown'),
                keywords=json.dumps(paper_data.get('keywords', [])),
                subject_categories=json.dumps(paper_data.get('categories', [])),
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error saving paper to database: {e}")
            return None


# Create singleton instance
literature_search_service = LiteratureSearchService()
