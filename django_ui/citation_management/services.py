"""Citation Management Service with AI-powered citation generation and formatting.

This service provides comprehensive citation management including automatic
citation generation, bibliography creation, and citation style formatting.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Optional, List, Any
from datetime import datetime

import openai
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class CitationManagementService:
    """AI-powered citation management service."""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=getattr(settings, 'OPENAI_API_KEY', None)
        )
        self.citation_styles = [
            'APA', 'MLA', 'Chicago', 'Harvard', 'IEEE', 'Vancouver', 'Nature'
        ]
        self.doi_api_url = "https://api.crossref.org/works"
        self.arxiv_api_url = "http://export.arxiv.org/api/query"
    
    async def generate_citation(
        self,
        identifier: str,
        style: str = 'APA',
        identifier_type: str = 'auto'
    ) -> Dict:
        """
        Generate citation from various identifiers (DOI, arXiv, URL, etc.).
        
        Args:
            identifier: DOI, arXiv ID, URL, or other identifier
            style: Citation style (APA, MLA, etc.)
            identifier_type: Type of identifier (auto-detect if 'auto')
            
        Returns:
            Dictionary containing citation information
        """
        try:
            logger.info(f"Generating {style} citation for: {identifier}")
            
            # Auto-detect identifier type if needed
            if identifier_type == 'auto':
                identifier_type = self._detect_identifier_type(identifier)
            
            # Get metadata based on identifier type
            metadata = await self._get_metadata(identifier, identifier_type)
            
            if not metadata:
                raise ValueError(f"Could not retrieve metadata for: {identifier}")
            
            # Generate citation in requested style
            citation = await self._format_citation(metadata, style)
            
            return {
                'success': True,
                'citation': citation,
                'metadata': metadata,
                'style': style,
                'identifier': identifier,
                'identifier_type': identifier_type
            }
            
        except Exception as e:
            logger.error(f"Error generating citation: {e}")
            return {
                'success': False,
                'error': str(e),
                'identifier': identifier,
                'style': style
            }
    
    async def generate_bibliography(
        self,
        citations: List[Dict],
        style: str = 'APA',
        sort_by: str = 'author'
    ) -> Dict:
        """
        Generate formatted bibliography from list of citations.
        
        Args:
            citations: List of citation metadata dictionaries
            style: Citation style
            sort_by: Sorting method (author, year, title)
            
        Returns:
            Dictionary containing formatted bibliography
        """
        try:
            logger.info(f"Generating {style} bibliography with {len(citations)} citations")
            
            # Sort citations
            sorted_citations = self._sort_citations(citations, sort_by)
            
            # Format each citation
            formatted_citations = []
            for citation_data in sorted_citations:
                formatted = await self._format_citation(citation_data, style)
                formatted_citations.append(formatted)
            
            # Create bibliography text
            bibliography_text = self._create_bibliography_text(formatted_citations, style)
            
            return {
                'success': True,
                'bibliography': bibliography_text,
                'formatted_citations': formatted_citations,
                'total_citations': len(citations),
                'style': style,
                'sort_by': sort_by
            }
            
        except Exception as e:
            logger.error(f"Error generating bibliography: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_citations': len(citations) if citations else 0
            }
    
    async def extract_citations_from_text(
        self,
        text: str,
        citation_style: str = 'auto'
    ) -> Dict:
        """
        Extract and parse citations from text.
        
        Args:
            text: Text containing citations
            citation_style: Expected citation style or 'auto' to detect
            
        Returns:
            Dictionary containing extracted citations
        """
        try:
            logger.info("Extracting citations from text")
            
            # Detect citation style if auto
            if citation_style == 'auto':
                citation_style = await self._detect_citation_style(text)
            
            # Extract citations using pattern matching
            raw_citations = self._extract_citation_patterns(text, citation_style)
            
            # Parse extracted citations with AI
            parsed_citations = await self._parse_citations_with_ai(raw_citations, citation_style)
            
            # Try to get full metadata for parsed citations
            enriched_citations = []
            for citation in parsed_citations:
                enriched = await self._enrich_citation_metadata(citation)
                enriched_citations.append(enriched)
            
            return {
                'success': True,
                'citations_found': len(raw_citations),
                'parsed_citations': parsed_citations,
                'enriched_citations': enriched_citations,
                'detected_style': citation_style
            }
            
        except Exception as e:
            logger.error(f"Error extracting citations: {e}")
            return {
                'success': False,
                'error': str(e),
                'citations_found': 0
            }
    
    async def convert_citation_style(
        self,
        citation: str,
        from_style: str,
        to_style: str
    ) -> Dict:
        """
        Convert citation from one style to another.
        
        Args:
            citation: Original citation text
            from_style: Original citation style
            to_style: Target citation style
            
        Returns:
            Dictionary containing converted citation
        """
        try:
            logger.info(f"Converting citation from {from_style} to {to_style}")
            
            # Parse original citation to extract metadata
            metadata = await self._parse_single_citation(citation, from_style)
            
            # Format in new style
            converted_citation = await self._format_citation(metadata, to_style)
            
            return {
                'success': True,
                'original_citation': citation,
                'converted_citation': converted_citation,
                'from_style': from_style,
                'to_style': to_style,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error converting citation style: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_citation': citation
            }
    
    async def validate_citations(
        self,
        citations: List[str],
        expected_style: str = 'APA'
    ) -> Dict:
        """
        Validate citations for formatting consistency and accuracy.
        
        Args:
            citations: List of citation strings
            expected_style: Expected citation style
            
        Returns:
            Dictionary containing validation results
        """
        try:
            logger.info(f"Validating {len(citations)} citations for {expected_style} style")
            
            validation_results = []
            
            for i, citation in enumerate(citations):
                result = await self._validate_single_citation(citation, expected_style)
                result['citation_index'] = i
                validation_results.append(result)
            
            # Calculate overall statistics
            valid_count = sum(1 for r in validation_results if r['is_valid'])
            error_count = sum(len(r['errors']) for r in validation_results)
            warning_count = sum(len(r['warnings']) for r in validation_results)
            
            return {
                'success': True,
                'total_citations': len(citations),
                'valid_citations': valid_count,
                'invalid_citations': len(citations) - valid_count,
                'total_errors': error_count,
                'total_warnings': warning_count,
                'validation_results': validation_results,
                'overall_valid': error_count == 0
            }
            
        except Exception as e:
            logger.error(f"Error validating citations: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_citations': len(citations) if citations else 0
            }
    
    def _detect_identifier_type(self, identifier: str) -> str:
        """Detect the type of identifier."""
        identifier = identifier.strip()
        
        # DOI patterns
        if re.match(r'^10\.\d+/', identifier) or identifier.startswith('doi:'):
            return 'doi'
        
        # arXiv patterns
        if re.match(r'^(\d{4}\.\d{4,5}(v\d+)?)|(\w+-\w+/\d{7}(v\d+)?)$', identifier):
            return 'arxiv'
        
        # URL patterns
        if identifier.startswith(('http://', 'https://')):
            return 'url'
        
        # ISBN patterns
        if re.match(r'^(97[89])?\d{9}[\dX]$', identifier.replace('-', '')):
            return 'isbn'
        
        # PubMed patterns
        if identifier.isdigit() and len(identifier) >= 6:
            return 'pmid'
        
        return 'unknown'
    
    async def _get_metadata(self, identifier: str, identifier_type: str) -> Optional[Dict]:
        """Get metadata for an identifier."""
        try:
            if identifier_type == 'doi':
                return await self._get_doi_metadata(identifier)
            elif identifier_type == 'arxiv':
                return await self._get_arxiv_metadata(identifier)
            elif identifier_type == 'url':
                return await self._get_url_metadata(identifier)
            else:
                # Use AI to try to parse any text into citation metadata
                return await self._parse_identifier_with_ai(identifier)
                
        except Exception as e:
            logger.error(f"Error getting metadata for {identifier}: {e}")
            return None
    
    async def _get_doi_metadata(self, doi: str) -> Optional[Dict]:
        """Get metadata from DOI using CrossRef API."""
        try:
            # Clean DOI
            doi = doi.replace('doi:', '').strip()
            
            url = f"{self.doi_api_url}/{doi}"
            response = await asyncio.to_thread(requests.get, url)
            response.raise_for_status()
            
            data = response.json()
            work = data['message']
            
            # Extract metadata
            metadata = {
                'title': work.get('title', [''])[0],
                'authors': [
                    f"{author.get('given', '')} {author.get('family', '')}"
                    for author in work.get('author', [])
                ],
                'journal': work.get('container-title', [''])[0],
                'year': str(work.get('published-print', work.get('published-online', {})).get('date-parts', [[None]])[0][0]),
                'volume': work.get('volume', ''),
                'issue': work.get('issue', ''),
                'pages': work.get('page', ''),
                'doi': work.get('DOI', ''),
                'url': work.get('URL', ''),
                'publisher': work.get('publisher', ''),
                'type': 'journal-article'
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting DOI metadata: {e}")
            return None
    
    async def _get_arxiv_metadata(self, arxiv_id: str) -> Optional[Dict]:
        """Get metadata from arXiv ID."""
        try:
            url = f"{self.arxiv_api_url}?id_list={arxiv_id}"
            response = await asyncio.to_thread(requests.get, url)
            response.raise_for_status()
            
            # Parse XML response (simplified - in production use xml.etree.ElementTree)
            content = response.text
            
            # Extract basic info (this is a simplified parser)
            title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
            title = title_match.group(1).strip() if title_match else ''
            
            # Extract authors (simplified)
            authors = re.findall(r'<name>(.*?)</name>', content)
            
            # Extract year from ID
            year = '20' + arxiv_id[:2] if arxiv_id[0].isdigit() else ''
            
            metadata = {
                'title': title,
                'authors': authors,
                'year': year,
                'arxiv_id': arxiv_id,
                'url': f"https://arxiv.org/abs/{arxiv_id}",
                'type': 'preprint'
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting arXiv metadata: {e}")
            return None
    
    async def _get_url_metadata(self, url: str) -> Optional[Dict]:
        """Get metadata from URL (simplified extraction)."""
        try:
            # In a real implementation, you would:
            # 1. Fetch the webpage
            # 2. Parse HTML for metadata (title, authors, etc.)
            # 3. Look for structured data (JSON-LD, microdata)
            # 4. Use content extraction libraries
            
            # For demo purposes, return basic metadata
            return {
                'title': 'Web Resource',
                'url': url,
                'type': 'webpage',
                'access_date': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            logger.error(f"Error getting URL metadata: {e}")
            return None
    
    async def _parse_identifier_with_ai(self, identifier: str) -> Optional[Dict]:
        """Use AI to parse unknown identifier into citation metadata."""
        try:
            prompt = f"""
            Parse this identifier/citation into structured metadata:
            
            Identifier: {identifier}
            
            Extract available information and format as JSON:
            {{
                "title": "title if available",
                "authors": ["author1", "author2"],
                "year": "year if available",
                "journal": "journal if available",
                "type": "article/book/webpage/etc"
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at parsing bibliographic information. Extract metadata from any citation format."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            try:
                metadata = json.loads(response.choices[0].message.content)
                return metadata
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            logger.error(f"Error parsing identifier with AI: {e}")
            return None
    
    async def _format_citation(self, metadata: Dict, style: str) -> str:
        """Format citation in specified style using AI."""
        try:
            # Create comprehensive prompt with metadata
            metadata_text = json.dumps(metadata, indent=2)
            
            prompt = f"""
            Format this citation metadata in {style} style:
            
            Metadata:
            {metadata_text}
            
            Provide only the formatted citation text, following {style} guidelines exactly.
            Include all available information in the correct order and format.
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert in {style} citation style. Format citations exactly according to {style} guidelines."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            citation = response.choices[0].message.content.strip()
            
            # Remove any quotation marks that might have been added
            citation = citation.strip('"\'')
            
            return citation
            
        except Exception as e:
            logger.error(f"Error formatting citation: {e}")
            # Fallback to basic formatting
            return self._basic_citation_format(metadata, style)
    
    def _basic_citation_format(self, metadata: Dict, style: str) -> str:
        """Basic citation formatting as fallback."""
        authors = ', '.join(metadata.get('authors', ['Unknown Author']))
        title = metadata.get('title', 'Unknown Title')
        year = metadata.get('year', 'n.d.')
        journal = metadata.get('journal', '')
        
        if style.upper() == 'APA':
            citation = f"{authors} ({year}). {title}."
            if journal:
                citation += f" {journal}."
        elif style.upper() == 'MLA':
            citation = f"{authors}. \"{title}.\""
            if journal:
                citation += f" {journal},"
            citation += f" {year}."
        else:
            citation = f"{authors}. {title}. {year}."
        
        return citation
    
    def _sort_citations(self, citations: List[Dict], sort_by: str) -> List[Dict]:
        """Sort citations by specified criteria."""
        try:
            if sort_by == 'author':
                return sorted(citations, key=lambda x: x.get('authors', [''])[0].split()[-1])
            elif sort_by == 'year':
                return sorted(citations, key=lambda x: x.get('year', '0'), reverse=True)
            elif sort_by == 'title':
                return sorted(citations, key=lambda x: x.get('title', ''))
            else:
                return citations
        except Exception:
            return citations
    
    def _create_bibliography_text(self, formatted_citations: List[str], style: str) -> str:
        """Create formatted bibliography text."""
        if style.upper() == 'APA':
            header = "References\n\n"
        elif style.upper() == 'MLA':
            header = "Works Cited\n\n"
        else:
            header = "Bibliography\n\n"
        
        bibliography = header + '\n\n'.join(formatted_citations)
        return bibliography
    
    async def _detect_citation_style(self, text: str) -> str:
        """Detect citation style from text using AI."""
        try:
            prompt = f"""
            Analyze this text and identify the citation style being used (APA, MLA, Chicago, etc.):
            
            Text sample:
            {text[:1000]}
            
            Respond with just the style name (e.g., "APA", "MLA", "Chicago").
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in citation styles. Identify the citation style from text samples."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            style = response.choices[0].message.content.strip()
            return style if style in self.citation_styles else 'APA'
            
        except Exception as e:
            logger.error(f"Error detecting citation style: {e}")
            return 'APA'  # Default fallback
    
    def _extract_citation_patterns(self, text: str, style: str) -> List[str]:
        """Extract citation patterns from text based on style."""
        citations = []
        
        if style.upper() == 'APA':
            # APA in-text citations: (Author, Year) or (Author et al., Year)
            pattern = r'\([^)]*\d{4}[^)]*\)'
            citations.extend(re.findall(pattern, text))
            
        elif style.upper() == 'MLA':
            # MLA in-text citations: (Author Page) or (Author Title Page)
            pattern = r'\([^)]*\d+\)'
            citations.extend(re.findall(pattern, text))
            
        else:
            # General patterns
            patterns = [
                r'\([^)]*\d{4}[^)]*\)',  # Parenthetical with year
                r'\[\d+\]',              # Numbered citations
                r'\([^)]*\d+\)',         # Parenthetical with numbers
            ]
            
            for pattern in patterns:
                citations.extend(re.findall(pattern, text))
        
        return list(set(citations))  # Remove duplicates
    
    async def _parse_citations_with_ai(
        self, citations: List[str], style: str
    ) -> List[Dict]:
        """Parse extracted citations using AI."""
        parsed_citations = []
        
        for citation in citations:
            try:
                prompt = f"""
                Parse this {style} citation and extract metadata:
                
                Citation: {citation}
                
                Extract available information as JSON:
                {{
                    "authors": ["author names"],
                    "year": "year if available",
                    "title": "title if available",
                    "page": "page numbers if available"
                }}
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert at parsing {style} citations. Extract all available metadata."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                
                try:
                    metadata = json.loads(response.choices[0].message.content)
                    metadata['original_citation'] = citation
                    parsed_citations.append(metadata)
                except json.JSONDecodeError:
                    # Fallback
                    parsed_citations.append({
                        'original_citation': citation,
                        'authors': [],
                        'year': '',
                        'title': ''
                    })
                    
            except Exception as e:
                logger.error(f"Error parsing citation {citation}: {e}")
                parsed_citations.append({
                    'original_citation': citation,
                    'error': str(e)
                })
        
        return parsed_citations
    
    async def _enrich_citation_metadata(self, citation: Dict) -> Dict:
        """Enrich citation with additional metadata if possible."""
        # Try to find full metadata if we have partial information
        enriched = citation.copy()
        
        # If we have author and year, try to find full citation
        if citation.get('authors') and citation.get('year'):
            try:
                # This would involve searching databases for full metadata
                # For now, just return the original citation
                pass
            except Exception:
                pass
        
        return enriched
    
    async def _parse_single_citation(self, citation: str, style: str) -> Dict:
        """Parse a single citation string into metadata."""
        try:
            prompt = f"""
            Parse this {style} citation into structured metadata:
            
            Citation: {citation}
            
            Extract all available information as JSON:
            {{
                "authors": ["author names"],
                "title": "title",
                "year": "year",
                "journal": "journal name",
                "volume": "volume",
                "issue": "issue",
                "pages": "page range",
                "publisher": "publisher",
                "doi": "DOI if available",
                "url": "URL if available"
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at parsing {style} citations. Extract all metadata accurately."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            try:
                metadata = json.loads(response.choices[0].message.content)
                return metadata
            except json.JSONDecodeError:
                return {'error': 'Could not parse citation'}
                
        except Exception as e:
            logger.error(f"Error parsing citation: {e}")
            return {'error': str(e)}
    
    async def _validate_single_citation(
        self, citation: str, expected_style: str
    ) -> Dict:
        """Validate a single citation for style compliance."""
        try:
            prompt = f"""
            Validate this citation for {expected_style} style compliance:
            
            Citation: {citation}
            
            Check for:
            - Correct formatting
            - Required elements
            - Style consistency
            - Common errors
            
            Provide response as JSON:
            {{
                "is_valid": true/false,
                "errors": ["error descriptions"],
                "warnings": ["warning descriptions"],
                "suggestions": ["improvement suggestions"]
            }}
            """
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert in {expected_style} citation style. Validate citations strictly according to {expected_style} guidelines."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            try:
                validation = json.loads(response.choices[0].message.content)
                validation['citation'] = citation
                return validation
            except json.JSONDecodeError:
                return {
                    'citation': citation,
                    'is_valid': False,
                    'errors': ['Could not validate citation'],
                    'warnings': [],
                    'suggestions': []
                }
                
        except Exception as e:
            logger.error(f"Error validating citation: {e}")
            return {
                'citation': citation,
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'suggestions': []
            }


# Create singleton instance
citation_management_service = CitationManagementService()
