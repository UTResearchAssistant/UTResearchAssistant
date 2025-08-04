from locust import HttpUser, task, between
import json
import random


class ResearchAssistantUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login or setup session
        self.client.post("/api/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
    
    @task(3)
    def search_literature(self):
        """Test literature search functionality"""
        queries = [
            "machine learning",
            "artificial intelligence",
            "neural networks",
            "deep learning",
            "natural language processing"
        ]
        query = random.choice(queries)
        
        response = self.client.post("/api/literature/search", json={
            "query": query,
            "limit": 10
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                # Follow up with paper details
                paper_id = data['results'][0]['id']
                self.client.get(f"/api/literature/paper/{paper_id}")
    
    @task(2)
    def generate_podcast(self):
        """Test podcast generation"""
        self.client.post("/api/podcast/generate", json={
            "paper_ids": ["123", "456"],
            "style": "summary",
            "duration": "short"
        })
    
    @task(1)
    def analyze_video(self):
        """Test video analysis"""
        self.client.post("/api/video/analyze", json={
            "video_url": "https://example.com/test.mp4",
            "analysis_type": "content"
        })
    
    @task(2)
    def writing_assistance(self):
        """Test writing assistance"""
        content = "This is a sample academic text that needs improvement."
        
        self.client.post("/api/writing/assist", json={
            "content": content,
            "task_type": "abstract",
            "tone": "academic"
        })
    
    @task(1)
    def check_academic_integrity(self):
        """Test academic integrity checking"""
        content = "Sample research content for plagiarism checking."
        
        self.client.post("/api/integrity/check", json={
            "content": content,
            "check_types": ["plagiarism", "citation"]
        })
    
    @task(1)
    def manage_citations(self):
        """Test citation management"""
        self.client.post("/api/citations/format", json={
            "citations": [
                {
                    "type": "article",
                    "title": "Test Article",
                    "authors": ["Author, A."],
                    "year": 2023
                }
            ],
            "style": "apa"
        })
    
    @task(1)
    def dashboard_metrics(self):
        """Test dashboard and analytics"""
        self.client.get("/api/analytics/dashboard")
        self.client.get("/api/analytics/usage")


class AdminUser(HttpUser):
    wait_time = between(2, 5)
    weight = 1  # Lower weight means fewer admin users
    
    def on_start(self):
        # Admin login
        self.client.post("/api/auth/login", json={
            "username": "admin_user",
            "password": "admin_password"
        })
    
    @task
    def view_system_metrics(self):
        """Admin views system metrics"""
        self.client.get("/api/admin/metrics")
        self.client.get("/api/admin/logs")
    
    @task
    def manage_users(self):
        """Admin manages users"""
        self.client.get("/api/admin/users")
        self.client.get("/api/admin/usage-stats")
