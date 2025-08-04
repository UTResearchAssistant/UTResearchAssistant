import time
import psutil
from memory_profiler import profile
from django_ui.literature_search.services import literature_search_service
from django_ui.podcast_generation.services import podcast_generation_service
from django_ui.writing_assistance.services import writing_assistance_service
import asyncio


@profile
def test_literature_search_memory():
    """Profile memory usage during literature search"""
    
    async def search_multiple():
        for i in range(20):
            result = await literature_search_service.search_papers(
                query=f"test query {i}",
                limit=10
            )
            print(f"Search {i} completed")
    
    asyncio.run(search_multiple())


@profile
def test_ai_generation_memory():
    """Profile memory usage during AI content generation"""
    
    async def generate_content():
        for i in range(10):
            result = await writing_assistance_service.assist_writing(
                content=f"This is test content number {i} for memory profiling.",
                task_type="abstract",
                tone="academic"
            )
            print(f"Generation {i} completed")
    
    asyncio.run(generate_content())


def monitor_system_resources():
    """Monitor system resources during test execution"""
    
    def log_resources():
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"CPU: {cpu_percent}%")
        print(f"Memory: {memory.percent}% ({memory.used / 1024 / 1024:.1f}MB used)")
        print(f"Disk: {disk.percent}% ({disk.used / 1024 / 1024 / 1024:.1f}GB used)")
        print("---")
    
    # Monitor for 60 seconds
    for i in range(12):
        log_resources()
        time.sleep(5)


if __name__ == "__main__":
    print("Starting memory profiling...")
    
    print("\n1. Testing literature search memory usage:")
    test_literature_search_memory()
    
    print("\n2. Testing AI generation memory usage:")
    test_ai_generation_memory()
    
    print("\n3. Monitoring system resources:")
    monitor_system_resources()
    
    print("Memory profiling completed.")
