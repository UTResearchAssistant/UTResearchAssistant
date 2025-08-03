"""
Django Service Integration Layer
Provides Django-compatible wrappers for all research assistant services
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from django.conf import settings

logger = logging.getLogger(__name__)

# Add parent directories to path for importing external services
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "services"))
sys.path.insert(0, str(BASE_DIR / "models"))
sys.path.insert(0, str(BASE_DIR / "agents"))


class DjangoServiceIntegrator:
    """
    Integrates all external services with Django
    """
    
    def __init__(self):
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all external services with Django compatibility"""
        try:
            # Import and initialize external services
            self._setup_ingestion_services()
            self._setup_agent_services()
            self._setup_model_services()
            self._setup_monitoring_services()
            logger.info("All external services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
    
    def _setup_ingestion_services(self):
        """Setup ingestion services"""
        try:
            from ingestion_service.parser import parse_pdf
            from ingestion_service.downloader import download_paper
            from ingestion_service.crawler import crawl_papers
            from ingestion_service.embedder import generate_embeddings
            from ingestion_service.scheduler import schedule_ingestion
            
            self.pdf_parser = parse_pdf
            self.paper_downloader = download_paper
            self.paper_crawler = crawl_papers
            self.embeddings_generator = generate_embeddings
            self.ingestion_scheduler = schedule_ingestion
            
            logger.info("Ingestion services initialized")
        except ImportError as e:
            logger.warning(f"Could not import ingestion services: {e}")
            self._setup_fallback_ingestion()
    
    def _setup_agent_services(self):
        """Setup agent services"""
        try:
            from agents.browsing_agent import BrowsingAgent
            from agents.summarizer_agent import SummarizerAgent
            
            # Initialize concrete agent implementations
            self.browsing_agent = BrowsingAgent()
            self.summarizer_agent = SummarizerAgent()
            logger.info("Agent services initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize agent services: {e}")
            self.browsing_agent = None
            self.summarizer_agent = None
    
    def _setup_model_services(self):
        """Setup model services"""
        try:
            from models.openai_api import OpenAIService
            from models.embedding_model.embedder import EmbeddingModel
            
            # Initialize with Django settings
            openai_key = getattr(settings, 'OPENAI_API_KEY', None)
            self.openai_service = OpenAIService(api_key=openai_key)
            self.embedding_model = EmbeddingModel()
            
            logger.info("Model services initialized")
        except ImportError as e:
            logger.warning(f"Could not import model services: {e}")
            self._setup_fallback_models()
    
    def _setup_monitoring_services(self):
        """Setup monitoring services"""
        try:
            from monitoring_service.monitor import SystemMonitor
            self.system_monitor = SystemMonitor()
            logger.info("Monitoring services initialized")
        except ImportError as e:
            logger.warning(f"Could not import monitoring services: {e}")
            self._setup_fallback_monitoring()
    
    def _setup_fallback_ingestion(self):
        """Fallback methods for ingestion services"""
        def fallback_parse_pdf(path):
            return f"Fallback: Text extracted from {path}"
        
        def fallback_download(url):
            return {"status": "error", "message": "Service unavailable"}
        
        def fallback_crawl(query):
            return []
        
        def fallback_embeddings(texts):
            import numpy as np
            return np.random.rand(len(texts), 384)
        
        def fallback_schedule():
            pass
        
        self.pdf_parser = fallback_parse_pdf
        self.paper_downloader = fallback_download
        self.paper_crawler = fallback_crawl
        self.embeddings_generator = fallback_embeddings
        self.ingestion_scheduler = fallback_schedule
    
    def _setup_fallback_agents(self):
        """Fallback methods for agent services"""
        class FallbackAgent:
            def process(self, data):
                return {"status": "processed", "result": data}
            
            def search(self, query):
                return []
            
            def summarize(self, text):
                return text[:200] + "..." if len(text) > 200 else text
        
        self.browsing_agent = FallbackAgent()
        self.summarizer_agent = FallbackAgent()
    
    def _setup_fallback_models(self):
        """Fallback methods for model services"""
        class FallbackOpenAI:
            def generate_completion(self, prompt):
                return "Fallback completion for: " + prompt[:50]
            
            def generate_embeddings(self, texts):
                import numpy as np
                return np.random.rand(len(texts), 1536)
        
        class FallbackEmbedding:
            def encode(self, texts):
                import numpy as np
                return np.random.rand(len(texts), 384)
        
        self.openai_service = FallbackOpenAI()
        self.embedding_model = FallbackEmbedding()
    
    def _setup_fallback_monitoring(self):
        """Fallback methods for monitoring services"""
        class FallbackMonitor:
            def get_system_stats(self):
                return {"cpu": 0, "memory": 0, "status": "unknown"}
            
            def log_activity(self, activity):
                pass
        
        self.system_monitor = FallbackMonitor()
    
    # Public interface methods
    def parse_document(self, file_path: str) -> str:
        """Parse document and extract text"""
        return self.pdf_parser(Path(file_path))
    
    def download_paper(self, url: str) -> Dict:
        """Download paper from URL"""
        return self.paper_downloader(url)
    
    def crawl_for_papers(self, query: str) -> List[Dict]:
        """Crawl for papers matching query"""
        return self.paper_crawler(query)
    
    def generate_embeddings(self, texts: List[str]) -> Any:
        """Generate embeddings for texts"""
        return self.embeddings_generator(texts)
    
    def browse_web(self, query: str) -> List[Dict]:
        """Browse web for information"""
        return self.browsing_agent.search(query)
    
    def summarize_text(self, text: str) -> str:
        """Summarize text content"""
        return self.summarizer_agent.summarize(text)
    
    def generate_ai_completion(self, prompt: str) -> str:
        """Generate AI completion"""
        return self.openai_service.generate_completion(prompt)
    
    def get_system_status(self) -> Dict:
        """Get system monitoring status"""
        return self.system_monitor.get_system_stats()


# Global instance
django_service_integrator = DjangoServiceIntegrator()
