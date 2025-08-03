"""Dataset and paper downloader service.

This module provides functions to download papers, datasets, and other
research materials from various sources including arXiv, academic databases,
and direct URLs.
"""

import logging
import requests
import tempfile
from pathlib import Path
from typing import Iterable, Dict, Optional, Union
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


def download_dataset(destination: Path, sources: Iterable[str]) -> None:
    """Download datasets from the given sources.

    Parameters
    ----------
    destination : pathlib.Path
        Directory where the downloaded files should be saved.
    sources : iterable of str
        List of dataset identifiers or URLs to download.  These could be
        arXiv category names, DOI links, or names of HuggingFace
        datasets.
    """
    destination.mkdir(parents=True, exist_ok=True)
    
    for src in sources:
        try:
            if src.startswith('http'):
                download_file(src, destination)
            elif src.startswith('arxiv:'):
                download_arxiv_paper(src.replace('arxiv:', ''), destination)
            else:
                logger.info(f"[Downloader] Processing dataset source: {src}")
        except Exception as e:
            logger.error(f"Failed to download from {src}: {e}")


def download_paper(url_or_id: str, destination: Optional[Path] = None) -> Dict:
    """Download a research paper from URL or ID.
    
    Parameters
    ----------
    url_or_id : str
        URL to the paper or paper ID (e.g., arXiv ID)
    destination : pathlib.Path, optional
        Directory to save the paper. If None, uses temp directory.
    
    Returns
    -------
    dict
        Download result with status, path, and metadata
    """
    if destination is None:
        destination = Path(tempfile.gettempdir()) / "research_papers"
    
    destination.mkdir(parents=True, exist_ok=True)
    
    try:
        if url_or_id.startswith('http'):
            return download_file(url_or_id, destination)
        elif 'arxiv' in url_or_id.lower() or url_or_id.replace('.', '').replace('v', '').isdigit():
            # Likely an arXiv ID
            arxiv_id = url_or_id.split('/')[-1]  # Extract ID from URL if needed
            return download_arxiv_paper(arxiv_id, destination)
        else:
            return {"status": "error", "message": f"Unsupported paper identifier: {url_or_id}"}
    except Exception as e:
        logger.error(f"Error downloading paper {url_or_id}: {e}")
        return {"status": "error", "message": str(e)}


def download_file(url: str, destination: Path, filename: Optional[str] = None) -> Dict:
    """Download a file from URL.
    
    Parameters
    ----------
    url : str
        URL to download
    destination : pathlib.Path
        Directory to save the file
    filename : str, optional
        Custom filename. If None, extracts from URL.
    
    Returns
    -------
    dict
        Download result
    """
    try:
        # Generate filename if not provided
        if filename is None:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or '.' not in filename:
                # Generate filename from URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"download_{url_hash}.pdf"
        
        file_path = destination / filename
        
        # Download with progress
        logger.info(f"Downloading {url} to {file_path}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        file_size = file_path.stat().st_size
        logger.info(f"Successfully downloaded {filename} ({file_size} bytes)")
        
        return {
            "status": "success",
            "path": str(file_path),
            "filename": filename,
            "size": file_size,
            "url": url
        }
        
    except requests.RequestException as e:
        logger.error(f"Network error downloading {url}: {e}")
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return {"status": "error", "message": str(e)}


def download_arxiv_paper(arxiv_id: str, destination: Path) -> Dict:
    """Download paper from arXiv.
    
    Parameters
    ----------
    arxiv_id : str
        arXiv paper ID (e.g., "2301.12345")
    destination : pathlib.Path
        Directory to save the paper
    
    Returns
    -------
    dict
        Download result
    """
    try:
        # Clean arXiv ID
        arxiv_id = arxiv_id.replace('arXiv:', '').replace('v', '').strip()
        
        # Construct arXiv PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        filename = f"arxiv_{arxiv_id}.pdf"
        
        logger.info(f"Downloading arXiv paper {arxiv_id}")
        
        result = download_file(pdf_url, destination, filename)
        
        if result["status"] == "success":
            # Add arXiv-specific metadata
            result["arxiv_id"] = arxiv_id
            result["source"] = "arxiv"
            
            # Try to get paper metadata
            try:
                import arxiv
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(search.results())
                
                result["metadata"] = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "categories": paper.categories,
                    "arxiv_url": paper.entry_id
                }
            except ImportError:
                logger.warning("arxiv package not available for metadata extraction")
            except Exception as e:
                logger.warning(f"Could not extract arXiv metadata: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error downloading arXiv paper {arxiv_id}: {e}")
        return {"status": "error", "message": str(e)}


def get_download_info(url_or_id: str) -> Dict:
    """Get information about a download without actually downloading.
    
    Parameters
    ----------
    url_or_id : str
        URL or ID to check
    
    Returns
    -------
    dict
        Information about the downloadable resource
    """
    try:
        if url_or_id.startswith('http'):
            response = requests.head(url_or_id, timeout=10)
            response.raise_for_status()
            
            return {
                "status": "available",
                "content_type": response.headers.get('content-type', 'unknown'),
                "content_length": response.headers.get('content-length', 'unknown'),
                "url": url_or_id
            }
        elif 'arxiv' in url_or_id.lower():
            arxiv_id = url_or_id.split('/')[-1].replace('arXiv:', '').replace('v', '')
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            response = requests.head(pdf_url, timeout=10)
            response.raise_for_status()
            
            return {
                "status": "available",
                "content_type": "application/pdf",
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url
            }
        else:
            return {"status": "unknown", "message": "Unsupported URL or ID format"}
            
    except requests.RequestException as e:
        return {"status": "unavailable", "message": f"Network error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
