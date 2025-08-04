import pytest
import time
import asyncio
from django_ui.literature_search.services import literature_search_service
from django_ui.podcast_generation.services import podcast_generation_service
from django_ui.writing_assistance.services import writing_assistance_service


@pytest.mark.benchmark(group="search")
def test_literature_search_performance(benchmark):
    """Benchmark literature search performance"""
    
    def search_papers():
        return asyncio.run(literature_search_service.search_papers(
            query="machine learning",
            limit=10
        ))
    
    result = benchmark(search_papers)
    assert result['success'] is True
    assert len(result['results']) > 0


@pytest.mark.benchmark(group="ai_generation")
def test_podcast_generation_performance(benchmark):
    """Benchmark podcast generation performance"""
    
    def generate_podcast():
        return asyncio.run(podcast_generation_service.generate_podcast(
            paper_ids=["test1", "test2"],
            style="summary",
            duration="short"
        ))
    
    result = benchmark(generate_podcast)
    assert result['success'] is True


@pytest.mark.benchmark(group="ai_assistance")
def test_writing_assistance_performance(benchmark):
    """Benchmark writing assistance performance"""
    
    def assist_writing():
        return asyncio.run(writing_assistance_service.assist_writing(
            content="This is a test document for academic writing assistance.",
            task_type="abstract",
            tone="academic"
        ))
    
    result = benchmark(assist_writing)
    assert result['success'] is True


@pytest.mark.benchmark(group="database")
def test_database_query_performance(benchmark):
    """Benchmark database operations"""
    
    def query_database():
        from django_ui.core.models import ResearchProject
        return list(ResearchProject.objects.all()[:100])
    
    result = benchmark(query_database)
    assert isinstance(result, list)


def test_concurrent_requests():
    """Test system under concurrent load"""
    
    async def concurrent_search():
        tasks = []
        for i in range(10):
            task = literature_search_service.search_papers(
                query=f"test query {i}",
                limit=5
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    start_time = time.time()
    results = asyncio.run(concurrent_search())
    end_time = time.time()
    
    assert len(results) == 10
    assert end_time - start_time < 30  # Should complete within 30 seconds


def test_memory_usage():
    """Test memory usage under load"""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operations
    for i in range(100):
        asyncio.run(literature_search_service.search_papers(
            query="memory test",
            limit=1
        ))
        if i % 10 == 0:
            gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024
