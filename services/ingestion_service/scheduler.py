"""Scheduler for periodic ingestion tasks.

This module demonstrates how one might schedule recurring jobs to
crawl new papers and update embeddings. In practice you could use
APScheduler, Celery beat or a simple cron job to trigger the
pipeline. Here we provide both synchronous and asynchronous functions.
"""

import time
import logging
from typing import Iterable, Dict, List, Optional
from datetime import datetime, timedelta

from .crawler import crawl_arxiv, crawl_papers
from .parser import parse_pdf, parse_document
from .embedder import embed_document_chunks, generate_embeddings

logger = logging.getLogger(__name__)


def schedule_ingestion(
    sources: List[str] = None,
    interval_hours: int = 24,
    max_papers: int = 100
) -> Dict:
    """Schedule ingestion task for research papers.
    
    Parameters
    ----------
    sources : list[str], optional
        Sources to ingest from (e.g., ['arxiv', 'semantic_scholar'])
    interval_hours : int
        Hours between ingestion runs
    max_papers : int
        Maximum papers to process per run
    
    Returns
    -------
    dict
        Scheduling result
    """
    if sources is None:
        sources = ['arxiv']
    
    try:
        logger.info(f"Scheduling ingestion for sources: {sources}")
        
        # For now, this is a mock scheduler
        # In production, you'd use Celery, APScheduler, or similar
        next_run = datetime.now() + timedelta(hours=interval_hours)
        
        result = {
            "status": "scheduled",
            "sources": sources,
            "interval_hours": interval_hours,
            "max_papers": max_papers,
            "next_run": next_run.isoformat(),
            "message": f"Ingestion scheduled for {len(sources)} sources"
        }
        
        logger.info(f"Ingestion scheduled: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error scheduling ingestion: {e}")
        return {"status": "error", "message": str(e)}


def run_immediate_ingestion(
    query: str = "machine learning",
    sources: List[str] = None,
    max_papers: int = 10
) -> Dict:
    """Run immediate ingestion for testing purposes.
    
    Parameters
    ----------
    query : str
        Search query for papers
    sources : list[str], optional
        Sources to search
    max_papers : int
        Maximum papers to process
    
    Returns
    -------
    dict
        Ingestion result
    """
    if sources is None:
        sources = ['arxiv']
    
    try:
        logger.info(f"Starting immediate ingestion: {query}")
        
        # Crawl papers
        papers = crawl_papers(query, sources, max_papers)
        
        # Process papers (simplified)
        processed_count = 0
        for paper in papers:
            try:
                # Generate embeddings for abstract
                if paper.get('summary'):
                    embeddings = generate_embeddings([paper['summary']])
                    processed_count += 1
            except Exception as e:
                logger.warning(f"Error processing paper: {e}")
        
        result = {
            "status": "completed",
            "query": query,
            "sources": sources,
            "papers_found": len(papers),
            "papers_processed": processed_count,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Ingestion completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in immediate ingestion: {e}")
        return {"status": "error", "message": str(e)}


def get_ingestion_status() -> Dict:
    """Get current ingestion system status.
    
    Returns
    -------
    dict
        Status information
    """
    return {
        "status": "ready",
        "last_run": None,
        "next_scheduled": None,
        "total_papers_processed": 0,
        "active_sources": ["arxiv"],
        "timestamp": datetime.now().isoformat()
    }
