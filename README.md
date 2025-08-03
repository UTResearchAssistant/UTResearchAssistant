# Enhanced Research Assistant v2.0 🎓✨

A comprehensive AI-powered research assistant that helps researchers discover, analyze, and synthesize academic content with cutting-edge features including podcast generation, video analysis, multilingual search, and collaborative tools.

## 🌟 Features

### Core Research Capabilities

- **🔍 Advanced Literature Search**: Multi-source academic search across arXiv, Google Scholar, Semantic Scholar, and CrossRef
- **📊 Intelligent Summarization**: AI-powered paper summarization with key insights extraction
- **🌐 Multilingual Support**: Search and analyze papers in multiple languages with automatic translation
- **📚 Citation Network Analysis**: Discover related papers and build comprehensive bibliographies

### Content Generation & Analysis

- **🎙️ Podcast Generation**: Transform research papers into engaging podcasts
  - Summary-style podcasts for quick paper overviews
  - Interview-style discussions between virtual experts
  - Debate formats exploring different perspectives
- **🎥 Video Analysis**: Extract insights from research videos and lectures
  - Automatic transcription and content analysis
  - Key concept extraction and timeline generation
  - Integration with academic video platforms

### Research Enhancement Tools

- **✍️ Writing Assistant**: AI-powered academic writing support
  - Literature review generation
  - Citation formatting and management
  - Writing style improvement suggestions
- **🔍 Academic Integrity Checker**: Ensure originality and proper attribution
  - Plagiarism detection
  - Citation verification
  - Academic standards compliance

### Collaboration & Organization

- **👥 Research Collaboration**: Connect and collaborate with other researchers
  - Shared research projects
  - Collaborative annotations and notes
  - Team paper recommendations
- **🔔 Research Alerts**: Stay updated with the latest developments
  - Keyword-based paper alerts
  - Author follow notifications
  - Conference and journal updates

### Data Management

- **📂 Dataset Management**: Organize and process research datasets
- **🏷️ Smart Tagging**: Automatic categorization and metadata extraction
- **📈 Research Analytics**: Track your research progress and impact

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- PostgreSQL 12+
- Redis 6+

### One-Command Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:

1. Set up Python virtual environment
2. Install all dependencies
3. Configure database and Redis
4. Create environment files
5. Build and start all services

### Manual Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd research-assistant
```

2. **Backend Setup**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**

```bash
cd frontend
npm install
```

4. **Environment Configuration**

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Database Setup**

```bash
docker run -d \
  --name research_postgres \
  -e POSTGRES_USER=research_user \
  -e POSTGRES_PASSWORD=research_password \
  -e POSTGRES_DB=research_assistant \
  -p 5432:5432 \
  postgres:15
```

6. **Start Services**

```bash
# Development mode
./start_dev.sh

# Or individually
./start_backend.sh  # Backend only
./start_frontend.sh # Frontend only
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://research_user:research_password@localhost:5432/research_assistant

# OpenAI (Required for AI features)
OPENAI_API_KEY=your_openai_api_key_here

# Google Services (Optional)
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here

# Academic APIs (Optional)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
CROSSREF_API_KEY=your_crossref_api_key_here

# Email Configuration (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Feature Flags
ENABLE_PODCAST_GENERATION=true
ENABLE_VIDEO_ANALYSIS=true
ENABLE_MULTILINGUAL_SEARCH=true
ENABLE_RESEARCH_ALERTS=true
ENABLE_WRITING_ASSISTANT=true
ENABLE_COLLABORATION=true
```

### API Keys Setup

1. **OpenAI API Key** (Required)

   - Sign up at [OpenAI](https://platform.openai.com/)
   - Generate an API key
   - Add to `.env` file

2. **Google Translate API** (Optional)

   - Enable Google Translate API in Google Cloud Console
   - Create credentials and download JSON key
   - Add API key to `.env` file

3. **Academic APIs** (Optional)
   - Semantic Scholar: [API Documentation](https://api.semanticscholar.org/)
   - CrossRef: [API Documentation](https://github.com/CrossRef/rest-api-doc)

## 📁 Project Structure

```
research-assistant/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/v1/         # API endpoints
│   │   ├── core/           # Configuration
│   │   ├── db/             # Database models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities
│   ├── enhanced_*.py       # Enhanced v2.0 features
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Next.js pages
│   │   ├── services/       # API services
│   │   └── utils/          # Utilities
│   └── package.json        # Node.js dependencies
├── services/               # Background services
│   ├── ingestion_service/  # Data ingestion
│   ├── monitoring_service/ # System monitoring
│   └── training_service/   # ML training
├── datasets/               # Data storage
├── models/                 # ML models
├── docs/                   # Documentation
└── docker-compose.yml      # Container orchestration
```

## 🎯 Usage Examples

### Basic Search

```python
from backend.enhanced_literature_service import EnhancedLiteratureSearchService

# Initialize service
search_service = EnhancedLiteratureSearchService()

# Search for papers
results = await search_service.unified_search(
    query="machine learning in healthcare",
    sources=["arxiv", "semantic_scholar"],
    limit=10
)
```

### Podcast Generation

```python
from backend.podcast_service import PodcastGenerationService

# Generate a podcast from a paper
podcast_service = PodcastGenerationService()

podcast = await podcast_service.generate_paper_summary_podcast(
    paper_content="Paper content here...",
    style="conversational",
    duration=300  # 5 minutes
)
```

### Video Analysis

```python
from backend.video_analysis_service import VideoAnalysisService

# Analyze a research video
video_service = VideoAnalysisService()

analysis = await video_service.analyze_research_video(
    video_url="https://example.com/video.mp4",
    analysis_type="comprehensive"
)
```

## 🌐 API Documentation

Once the backend is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Key Endpoints

- `GET /api/v2/search/unified` - Multi-source literature search
- `POST /api/v2/podcasts/generate` - Generate podcasts from papers
- `POST /api/v2/videos/analyze` - Analyze research videos
- `GET /api/v2/alerts/` - Research alerts management
- `POST /api/v2/writing/assist` - Writing assistance tools
- `GET /api/v2/collaboration/requests` - Collaboration features

## 🧪 Testing

Run the test suite to verify your setup:

```bash
# Test entire setup
./test_setup.sh

# Run backend tests
cd backend
source venv/bin/activate
pytest

# Run frontend tests
cd frontend
npm test
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and deploy with Docker Compose
docker-compose up -d --build

# Scale services
docker-compose up -d --scale backend=3
```

### Production Considerations

- Use environment-specific configuration files
- Set up SSL/TLS certificates
- Configure proper database backups
- Monitor system resources and performance
- Set up logging and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT and Whisper APIs
- The research community for inspiration
- All contributors to open-source academic tools

## 📞 Support

- 📧 Email: support@research-assistant.com
- 💬 Discord: [Research Assistant Community](https://discord.gg/research-assistant)
- 📖 Documentation: [Full Documentation](https://docs.research-assistant.com)

---

**Happy Researching!** 🎓✨
