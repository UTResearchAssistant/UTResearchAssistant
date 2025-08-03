"""Enhanced literature search service with advanced features."""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import arxiv
from scholarly import scholarly
import requests
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnhancedLiteratureSearchService:
    """Advanced literature search service with multilingual support and smart filtering."""

    def __init__(self, config=None):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.cache = {}  # Simple in-memory cache

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def unified_search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 50,
        filters: Dict = None,
    ) -> List[Dict]:
        """Perform unified search across multiple academic databases."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "pubmed", "google_scholar"]

        # Check cache first
        cache_key = f"{query}_{':'.join(sorted(sources))}_{max_results}"
        if cache_key in self.cache:
            logger.info(f"Returning cached results for query: {query}")
            return self.cache[cache_key]

        all_results = []
        tasks = []

        # Create search tasks for each source
        results_per_source = max(max_results // len(sources), 10)

        if "arxiv" in sources:
            tasks.append(self.search_arxiv(query, results_per_source))
        if "semantic_scholar" in sources:
            tasks.append(self.search_semantic_scholar(query, results_per_source))
        if "pubmed" in sources:
            tasks.append(self.search_pubmed(query, results_per_source))
        if "google_scholar" in sources:
            tasks.append(self.search_google_scholar(query, results_per_source))
        if "ieee" in sources:
            tasks.append(self.search_ieee(query, results_per_source))

        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search error: {result}")

        # Remove duplicates and apply filters
        unique_results = self._remove_duplicates(all_results)
        if filters:
            unique_results = self._apply_filters(unique_results, filters)

        # Sort by relevance and citation count
        unique_results = self._sort_results(unique_results)

        # Cache results
        final_results = unique_results[:max_results]
        self.cache[cache_key] = final_results

        return final_results

    async def search_arxiv(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search arXiv database."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for result in search.results():
                paper_data = {
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(author) for author in result.authors],
                    "doi": result.doi,
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "publication_date": result.published,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "source": "arxiv",
                    "url": result.entry_id,
                    "citation_count": 0,  # ArXiv doesn't provide citation counts
                    "relevance_score": 1.0,
                }
                results.append(paper_data)

            return results
        except Exception as e:
            logger.error(f"ArXiv search error: {str(e)}")
            return []

    async def search_semantic_scholar(
        self, query: str, max_results: int = 20
    ) -> List[Dict]:
        """Search Semantic Scholar API."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,authors,citationCount,publicationDate,doi,url,venue,year",
            }

            headers = {}
            if self.config and self.config.semantic_scholar_api_key:
                headers["x-api-key"] = self.config.semantic_scholar_api_key

            async with self.session.get(
                url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for paper in data.get("data", []):
                        if paper.get("title"):
                            paper_data = {
                                "id": paper.get("paperId"),
                                "title": paper.get("title"),
                                "abstract": paper.get("abstract", ""),
                                "authors": [
                                    author.get("name", "")
                                    for author in paper.get("authors", [])
                                ],
                                "doi": paper.get("doi"),
                                "publication_date": paper.get("publicationDate"),
                                "year": paper.get("year"),
                                "citation_count": paper.get("citationCount", 0),
                                "journal": (
                                    paper.get("venue", {}).get("name")
                                    if paper.get("venue")
                                    else None
                                ),
                                "url": paper.get("url"),
                                "source": "semantic_scholar",
                                "relevance_score": 0.9,
                            }
                            results.append(paper_data)

                    return results
                else:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {str(e)}")
            return []

    async def search_pubmed(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search PubMed database."""
        try:
            # PubMed E-utilities API
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
            }

            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    search_data = await response.json()
                    pmids = search_data.get("esearchresult", {}).get("idlist", [])

                    if not pmids:
                        return []

                    # Fetch details for PMIDs
                    fetch_url = (
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    )
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(pmids),
                        "retmode": "xml",
                    }

                    async with self.session.get(
                        fetch_url, params=fetch_params
                    ) as fetch_response:
                        if fetch_response.status == 200:
                            xml_data = await fetch_response.text()
                            return self._parse_pubmed_xml(xml_data)

            return []
        except Exception as e:
            logger.error(f"PubMed search error: {str(e)}")
            return []

    async def search_google_scholar(
        self, query: str, max_results: int = 20
    ) -> List[Dict]:
        """Search Google Scholar using scholarly library."""
        try:
            search_query = scholarly.search_pubs(query)
            results = []

            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break

                paper_data = {
                    "id": (
                        pub.get("pub_url", "").split("/")[-1]
                        if pub.get("pub_url")
                        else None
                    ),
                    "title": pub.get("bib", {}).get("title", ""),
                    "abstract": pub.get("bib", {}).get("abstract", ""),
                    "authors": pub.get("bib", {}).get("author", []),
                    "publication_date": pub.get("bib", {}).get("pub_year"),
                    "journal": pub.get("bib", {}).get("venue", ""),
                    "citation_count": pub.get("num_citations", 0),
                    "url": pub.get("pub_url"),
                    "source": "google_scholar",
                    "relevance_score": 0.8,
                }
                results.append(paper_data)

            return results
        except Exception as e:
            logger.error(f"Google Scholar search error: {str(e)}")
            return []

    async def search_ieee(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search IEEE Xplore database."""
        try:
            if not self.config or not self.config.ieee_api_key:
                logger.warning("IEEE API key not configured")
                return []

            url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
            params = {
                "querytext": query,
                "max_records": max_results,
                "start_record": 1,
                "sort_field": "article_number",
                "sort_order": "desc",
                "apikey": self.config.ieee_api_key,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for article in data.get("articles", []):
                        paper_data = {
                            "id": article.get("article_number"),
                            "title": article.get("title"),
                            "abstract": article.get("abstract"),
                            "authors": [
                                author.get("full_name", "")
                                for author in article.get("authors", {}).get(
                                    "authors", []
                                )
                            ],
                            "doi": article.get("doi"),
                            "publication_date": article.get("publication_date"),
                            "journal": article.get("publication_title"),
                            "citation_count": article.get("citing_paper_count", 0),
                            "url": article.get("html_url"),
                            "pdf_url": article.get("pdf_url"),
                            "source": "ieee",
                            "relevance_score": 0.85,
                        }
                        results.append(paper_data)

                    return results

            return []
        except Exception as e:
            logger.error(f"IEEE search error: {str(e)}")
            return []

    async def search_multilingual(
        self, query: str, languages: List[str], max_results: int = 50
    ) -> List[Dict]:
        """Search for papers in multiple languages."""
        all_results = []

        for lang in languages:
            # Translate query if needed
            translated_query = await self._translate_query(query, lang)

            # Search with translated query
            lang_results = await self.unified_search(
                translated_query, max_results=max_results // len(languages)
            )

            # Add language metadata
            for result in lang_results:
                result["search_language"] = lang
                result["translated_query"] = translated_query

            all_results.extend(lang_results)

        return self._remove_duplicates(all_results)

    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict]:
        """Parse PubMed XML response."""
        # Simplified XML parsing - in production, use proper XML parser
        results = []
        # This is a placeholder - implement proper XML parsing
        return results

    def _remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on DOI and title similarity."""
        unique_papers = {}

        for paper in papers:
            # Use DOI as primary key
            if paper.get("doi"):
                key = paper["doi"]
            else:
                # Use normalized title
                title = paper.get("title", "").lower().strip()
                key = re.sub(r"[^\w\s]", "", title)

            if key and key not in unique_papers:
                unique_papers[key] = paper
            elif key in unique_papers:
                # Keep paper with higher citation count
                existing = unique_papers[key]
                if paper.get("citation_count", 0) > existing.get("citation_count", 0):
                    unique_papers[key] = paper

        return list(unique_papers.values())

    def _apply_filters(self, papers: List[Dict], filters: Dict) -> List[Dict]:
        """Apply search filters to results."""
        filtered_papers = []

        for paper in papers:
            if self._matches_filters(paper, filters):
                filtered_papers.append(paper)

        return filtered_papers

    def _matches_filters(self, paper: Dict, filters: Dict) -> bool:
        """Check if paper matches given filters."""
        # Year filter
        if "year_from" in filters:
            pub_date = paper.get("publication_date") or paper.get("year")
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        year = int(pub_date[:4])
                    else:
                        year = (
                            pub_date.year
                            if hasattr(pub_date, "year")
                            else int(pub_date)
                        )

                    if year < filters["year_from"]:
                        return False
                except (ValueError, AttributeError):
                    pass

        # Citation count filter
        if "min_citations" in filters:
            citations = paper.get("citation_count", 0)
            if citations < filters["min_citations"]:
                return False

        # Source filter
        if "sources" in filters:
            if paper.get("source") not in filters["sources"]:
                return False

        # Author filter
        if "authors" in filters:
            paper_authors = [author.lower() for author in paper.get("authors", [])]
            required_authors = [author.lower() for author in filters["authors"]]
            if not any(
                req_author in paper_author
                for req_author in required_authors
                for paper_author in paper_authors
            ):
                return False

        return True

    def _sort_results(self, papers: List[Dict]) -> List[Dict]:
        """Sort papers by relevance and citation count."""

        def sort_key(paper):
            citation_count = paper.get("citation_count", 0)
            relevance_score = paper.get("relevance_score", 0.5)
            # Combine citation count and relevance
            return citation_count * 0.3 + relevance_score * 0.7

        return sorted(papers, key=sort_key, reverse=True)

    async def _translate_query(self, query: str, target_language: str) -> str:
        """Translate search query to target language."""
        # Placeholder for translation service
        # In production, integrate with Google Translate or similar service
        return query

    async def get_paper_details(self, paper_id: str, source: str) -> Optional[Dict]:
        """Get detailed information about a specific paper."""
        if source == "arxiv":
            return await self._get_arxiv_details(paper_id)
        elif source == "semantic_scholar":
            return await self._get_semantic_scholar_details(paper_id)
        elif source == "pubmed":
            return await self._get_pubmed_details(paper_id)

        return None

    async def _get_arxiv_details(self, arxiv_id: str) -> Optional[Dict]:
        """Get detailed ArXiv paper information."""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            for result in search.results():
                return {
                    "id": arxiv_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(author) for author in result.authors],
                    "doi": result.doi,
                    "publication_date": result.published,
                    "pdf_url": result.pdf_url,
                    "categories": result.categories,
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "source": "arxiv",
                }
        except Exception as e:
            logger.error(f"Error fetching ArXiv details: {e}")

        return None

    async def _get_semantic_scholar_details(self, paper_id: str) -> Optional[Dict]:
        """Get detailed Semantic Scholar paper information."""
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
            params = {
                "fields": "title,abstract,authors,citationCount,publicationDate,doi,url,venue,references,citations"
            }

            headers = {}
            if self.config and self.config.semantic_scholar_api_key:
                headers["x-api-key"] = self.config.semantic_scholar_api_key

            async with self.session.get(
                url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar details: {e}")

        return None

    async def _get_pubmed_details(self, pmid: str) -> Optional[Dict]:
        """Get detailed PubMed paper information."""
        # Placeholder for PubMed details fetching
        return None
