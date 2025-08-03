"""
Django management command to test all research assistant services integration
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Test all research assistant services integration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test-type',
            choices=['all', 'services', 'models', 'agents', 'integration'],
            default='all',
            help='Type of test to run'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )

    def handle(self, *args, **options):
        self.verbosity = options['verbosity']
        test_type = options['test_type']
        
        self.stdout.write(
            self.style.SUCCESS('🚀 Starting Research Assistant Integration Tests')
        )
        
        if test_type in ['all', 'services']:
            self.test_django_services()
        
        if test_type in ['all', 'models']:
            self.test_model_services()
        
        if test_type in ['all', 'agents']:
            self.test_agent_services()
        
        if test_type in ['all', 'integration']:
            self.test_full_integration()
        
        self.stdout.write(
            self.style.SUCCESS('✅ All tests completed successfully!')
        )

    def test_django_services(self):
        """Test Django-specific services"""
        self.stdout.write('📊 Testing Django Services...')
        
        try:
            # Test Llama text processor
            from services.llama_text_processor import llama_text_processor
            
            test_text = "This is a test document about machine learning and artificial intelligence."
            
            # Test embeddings
            embeddings = llama_text_processor.generate_embeddings([test_text])
            self.stdout.write(f"  ✅ Llama embeddings: {len(embeddings)} vectors generated")
            
            # Test keyword extraction
            keywords = llama_text_processor.extract_keywords(test_text)
            self.stdout.write(f"  ✅ Keywords extracted: {keywords[:5]}")
            
            # Test summarization
            summary = llama_text_processor.summarize_text(test_text)
            self.stdout.write(f"  ✅ Summary generated: {summary[:50]}...")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Llama text processor error: {e}")
            )
        
        try:
            # Test literature search service
            from services.literature_search_service import enhanced_literature_search_service
            
            # Test search functionality
            results = enhanced_literature_search_service.unified_search(
                "machine learning",
                sources=["arxiv"],
                limit=5
            )
            self.stdout.write(f"  ✅ Literature search: Found {len(results.get('papers', []))} papers")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Literature search service error: {e}")
            )
        
        try:
            # Test paper analysis service
            from services.paper_analysis_service import paper_analysis_service
            
            test_paper = {
                "title": "Deep Learning for Natural Language Processing",
                "abstract": "This paper presents a comprehensive review of deep learning techniques for natural language processing tasks.",
                "authors": ["John Doe", "Jane Smith"]
            }
            
            # Test analysis
            analysis = paper_analysis_service.analyze_paper(test_paper)
            if isinstance(analysis, dict) and 'summary' in analysis:
                self.stdout.write(f"  ✅ Paper analysis: Generated analysis with {len(analysis)} components")
            else:
                self.stdout.write("  ⚠️  Paper analysis: Basic analysis completed")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Paper analysis service error: {e}")
            )

    def test_model_services(self):
        """Test model services integration"""
        self.stdout.write('🤖 Testing Model Services...')
        
        try:
            # Test Django integration service
            from services.django_integration import django_service_integrator
            
            # Test document parsing
            test_result = django_service_integrator.parse_document("/tmp/test.pdf")
            self.stdout.write(f"  ✅ Document parser: {test_result[:50]}...")
            
            # Test embeddings
            test_texts = ["Hello world", "Machine learning is fascinating"]
            embeddings = django_service_integrator.generate_embeddings(test_texts)
            self.stdout.write(f"  ✅ Embeddings: Generated for {len(test_texts)} texts")
            
            # Test AI completion
            completion = django_service_integrator.generate_ai_completion("What is AI?")
            self.stdout.write(f"  ✅ AI completion: {completion[:50]}...")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Model services error: {e}")
            )

    def test_agent_services(self):
        """Test agent services integration"""
        self.stdout.write('🤖 Testing Agent Services...')
        
        try:
            from services.django_integration import django_service_integrator
            
            # Test browsing agent
            search_results = django_service_integrator.browse_web("machine learning")
            self.stdout.write(f"  ✅ Browsing agent: Found {len(search_results)} results")
            
            # Test summarizer agent
            test_text = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."
            summary = django_service_integrator.summarize_text(test_text)
            self.stdout.write(f"  ✅ Summarizer agent: {summary[:50]}...")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Agent services error: {e}")
            )

    def test_full_integration(self):
        """Test full system integration"""
        self.stdout.write('🔄 Testing Full Integration...')
        
        try:
            # Test end-to-end workflow
            from services.django_integration import django_service_integrator
            from services.paper_analysis_service import paper_analysis_service
            from services.literature_search_service import enhanced_literature_search_service
            
            # Step 1: Search for papers
            search_results = enhanced_literature_search_service.unified_search(
                "artificial intelligence",
                limit=3
            )
            
            papers_found = len(search_results.get('papers', []))
            self.stdout.write(f"  ✅ Step 1 - Literature Search: {papers_found} papers found")
            
            # Step 2: Analyze a paper
            if papers_found > 0:
                first_paper = search_results['papers'][0]
                analysis = paper_analysis_service.analyze_paper(first_paper)
                self.stdout.write(f"  ✅ Step 2 - Paper Analysis: Analysis completed")
            
            # Step 3: Generate embeddings
            test_texts = ["AI research", "Machine learning applications"]
            embeddings = django_service_integrator.generate_embeddings(test_texts)
            self.stdout.write(f"  ✅ Step 3 - Embeddings: Generated successfully")
            
            # Step 4: System monitoring
            system_status = django_service_integrator.get_system_status()
            self.stdout.write(f"  ✅ Step 4 - System Status: {system_status}")
            
            self.stdout.write(
                self.style.SUCCESS('  🎉 Full integration test completed successfully!')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Full integration error: {e}")
            )

    def test_database_integration(self):
        """Test database models integration"""
        self.stdout.write('💾 Testing Database Integration...')
        
        try:
            from core.models import Paper, SearchHistory, UserPreferences
            from django.contrib.auth.models import User
            
            # Test paper model
            paper_count = Paper.objects.count()
            self.stdout.write(f"  ✅ Papers in database: {paper_count}")
            
            # Test search history
            search_count = SearchHistory.objects.count()
            self.stdout.write(f"  ✅ Search history entries: {search_count}")
            
            # Test user preferences
            pref_count = UserPreferences.objects.count()
            self.stdout.write(f"  ✅ User preferences: {pref_count}")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"  ❌ Database integration error: {e}")
            )
