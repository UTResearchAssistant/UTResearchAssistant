"""
Enhanced Literature Search Service - Unified Multi-Database Search
Integrates arXiv, Semantic Scholar, Google Scholar, and PubMed
"""

import aiohttp
import asyncio
import arxiv
import requests
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging
from scholarly import scholarly
import datetime
import re

logger = logging.getLogger(__name__)


class EnhancedLiteratureSearchService:
    """
    Advanced literature search service that provides unified search across
    multiple academic databases with deduplication and filtering capabilities
    """
    
    def __init__(self):
        self.session = None
        self.arxiv_client = arxiv.Client()
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def unified_search(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Perform unified search across all academic databases
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with unified schema
        """
        try:
            # Search all databases concurrently
            tasks = [
                self._search_arxiv(query, max_results // 4),
                self._search_semantic_scholar(query, max_results // 4),
                self._search_google_scholar(query, max_results // 4),
                self._search_pubmed(query, max_results // 4)
            ]
            
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            all_papers = []
            for result in search_results:
                if isinstance(result, list):
                    all_papers.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Search error: {result}")
            
            # Remove duplicates and limit results
            unique_results = self._remove_duplicates(all_papers)
            return unique_results[:max_results]
            
        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return []
    
    async def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv database"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in self.arxiv_client.results(search):
                paper_data = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "publication_date": result.published.isoformat() if result.published else None,
                    "journal": "arXiv",
                    "doi": result.doi,
                    "url": result.entry_id,
                    "citation_count": 0,  # arXiv doesn't provide citation counts
                    "source": "arXiv",
                    "categories": [cat for cat in result.categories],
                    "pdf_url": result.pdf_url
                }
                papers.append(paper_data)
            
            return papers
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar database"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            url = f"{self.semantic_scholar_base}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,authors,abstract,year,venue,citationCount,url,externalIds"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = []
                    
                    for paper in data.get("data", []):
                        authors = [author.get("name", "") for author in paper.get("authors", [])]
                        
                        paper_data = {
                            "title": paper.get("title", ""),
                            "authors": authors,
                            "abstract": paper.get("abstract", ""),
                            "publication_date": f"{paper.get('year', '')}-01-01" if paper.get("year") else None,
                            "journal": paper.get("venue", ""),
                            "doi": paper.get("externalIds", {}).get("DOI"),
                            "url": paper.get("url", ""),
                            "citation_count": paper.get("citationCount", 0),
                            "source": "Semantic Scholar",
                            "semantic_scholar_id": paper.get("paperId")
                        }
                        papers.append(paper_data)
                    
                    return papers
                else:
                    logger.warning(f"Semantic Scholar API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    async def _search_google_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Google Scholar (using scholarly library)"""
        try:
            search_query = scholarly.search_pubs(query)
            papers = []
            
            count = 0
            for pub in search_query:
                if count >= limit:
                    break
                
                try:
                    # Fill publication details
                    filled_pub = scholarly.fill(pub)
                    
                    paper_data = {
                        "title": filled_pub.get("bib", {}).get("title", ""),
                        "authors": filled_pub.get("bib", {}).get("author", []),
                        "abstract": filled_pub.get("bib", {}).get("abstract", ""),
                        "publication_date": f"{filled_pub.get('bib', {}).get('pub_year', '')}-01-01" if filled_pub.get("bib", {}).get("pub_year") else None,
                        "journal": filled_pub.get("bib", {}).get("venue", ""),
                        "doi": None,  # Google Scholar doesn't always provide DOI
                        "url": filled_pub.get("pub_url", ""),
                        "citation_count": filled_pub.get("num_citations", 0),
                        "source": "Google Scholar",
                        "google_scholar_id": filled_pub.get("scholar_id")
                    }
                    papers.append(paper_data)
                    count += 1
                    
                except Exception as pub_error:
                    logger.warning(f"Error processing Google Scholar publication: {pub_error}")
                    continue
            
            return papers
            
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            return []
    
    async def _search_pubmed(self, query: str, limit: int) -> List[Dict]:
        """Search PubMed database"""
        try:
            # Using NCBI E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            
            # Search for paper IDs
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json"
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    return []
                
                search_data = await response.json()
                id_list = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not id_list:
                    return []
                
                # Fetch detailed information
                fetch_url = f"{base_url}efetch.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list[:limit]),
                    "retmode": "xml"
                }
                
                async with self.session.get(fetch_url, params=fetch_params) as fetch_response:
                    if fetch_response.status != 200:
                        return []
                    
                    # Parse XML response (simplified parsing)
                    xml_content = await fetch_response.text()
                    papers = self._parse_pubmed_xml(xml_content)
                    
                    return papers
                    
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """Parse PubMed XML response (simplified implementation)"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            papers = []
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Extract authors
                    authors = []
                    for author in article.findall(".//Author"):
                        last_name = author.find("LastName")
                        first_name = author.find("ForeName")
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")
                    
                    # Extract abstract
                    abstract_elem = article.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract publication date
                    pub_date = article.find(".//PubDate")
                    year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else ""
                    
                    # Extract journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Extract DOI
                    doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    paper_data = {
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "publication_date": f"{year}-01-01" if year else None,
                        "journal": journal,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.find('.//PMID').text}/" if article.find('.//PMID') is not None else "",
                        "citation_count": 0,  # PubMed doesn't provide citation counts
                        "source": "PubMed"
                    }
                    papers.append(paper_data)
                    
                except Exception as parse_error:
                    logger.warning(f"Error parsing PubMed article: {parse_error}")
                    continue
            
            return papers
            
        except Exception as e:
            logger.error(f"PubMed XML parsing error: {e}")
            return []
    
    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get("title", "").lower().strip()
            # Simple deduplication by title
            title_key = re.sub(r'[^\w\s]', '', title)
            title_key = ' '.join(title_key.split())
            
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers
    
    async def search_with_filters(self, query: str, filters: Dict[str, Any]) -> List[Dict]:
        """Search with advanced filters"""
        results = await self.unified_search(query)
        
        # Apply filters
        filtered_results = []
        for paper in results:
            if self._matches_filters(paper, filters):
                filtered_results.append(paper)
        
        return filtered_results
    
    def _matches_filters(self, paper: Dict, filters: Dict[str, Any]) -> bool:
        """Check if paper matches the given filters"""
        # Year filter
        if "year_from" in filters or "year_to" in filters:
            pub_date = paper.get("publication_date")
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        year = int(pub_date.split('-')[0])
                    elif hasattr(pub_date, 'year'):
                        year = pub_date.year
                    else:
                        year = int(pub_date)
                    
                    if "year_from" in filters and year < filters["year_from"]:
                        return False
                    if "year_to" in filters and year > filters["year_to"]:
                        return False
                except:
                    pass
        
        # Author filter
        if "author" in filters:
            authors = paper.get("authors", [])
            author_names = " ".join(authors).lower()
            if filters["author"].lower() not in author_names:
                return False
        
        # Journal filter
        if "journal" in filters:
            journal = paper.get("journal", "").lower()
            if filters["journal"].lower() not in journal:
                return False
        
        # Minimum citation count
        if "min_citations" in filters:
            citations = paper.get("citation_count", 0)
            if citations < filters["min_citations"]:
                return False
        
        return True


# Create global instance
literature_search_service = EnhancedLiteratureSearchService()
