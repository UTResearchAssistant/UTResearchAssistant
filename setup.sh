#!/bin/bash

# Enhanced Research Assistant Setup Script
# This script sets up the complete research assistant with all enhanced features

set -e

echo "🚀 Setting up Enhanced Research Assistant v2.0..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_warning "This script is optimized for macOS. Some commands may need adjustment for other systems."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

if ! command_exists node; then
    print_error "Node.js is required but not installed."
    exit 1
fi

if ! command_exists docker; then
    print_error "Docker is required but not installed."
    exit 1
fi

print_success "All prerequisites found!"

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Python dependencies installed"

# Go back to project root
cd ..

# Setup frontend dependencies
print_status "Setting up frontend dependencies..."
cd frontend
npm install
print_success "Frontend dependencies installed"

# Go back to project root
cd ..

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p datasets/metadata
mkdir -p datasets/papers_processed
mkdir -p datasets/papers_raw
mkdir -p datasets/training_data
mkdir -p datasets/user_uploads
mkdir -p backend/logs
mkdir -p backend/temp
mkdir -p backend/audio_output
mkdir -p backend/video_output
print_success "Directories created"

# Create environment configuration
print_status "Creating environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# Database Configuration
DATABASE_URL=postgresql://research_user:research_password@localhost:5432/research_assistant
POSTGRES_USER=research_user
POSTGRES_PASSWORD=research_password
POSTGRES_DB=research_assistant

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Translate API
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here

# Academic APIs (optional)
ARXIV_API_KEY=
CROSSREF_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=

# Email Configuration (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Settings
SECRET_KEY=your_secret_key_here
ENVIRONMENT=development
DEBUG=true

# Feature Flags
ENABLE_PODCAST_GENERATION=true
ENABLE_VIDEO_ANALYSIS=true
ENABLE_MULTILINGUAL_SEARCH=true
ENABLE_RESEARCH_ALERTS=true
ENABLE_WRITING_ASSISTANT=true
ENABLE_COLLABORATION=true
ENABLE_INTEGRITY_CHECKING=true

# Audio/Video Processing
MAX_AUDIO_DURATION=3600
MAX_VIDEO_SIZE=500MB
AUDIO_OUTPUT_FORMAT=mp3
VIDEO_OUTPUT_FORMAT=mp4
EOL
    print_success "Environment file created (.env)"
    print_warning "Please update the .env file with your API keys and configuration"
else
    print_warning ".env file already exists"
fi

# Create Docker environment file
if [ ! -f ".env.docker" ]; then
    cp .env .env.docker
    print_success "Docker environment file created (.env.docker)"
fi

# Setup database with Docker
print_status "Setting up database..."
if ! docker ps | grep -q research_postgres; then
    docker run -d \
        --name research_postgres \
        -e POSTGRES_USER=research_user \
        -e POSTGRES_PASSWORD=research_password \
        -e POSTGRES_DB=research_assistant \
        -p 5432:5432 \
        -v research_postgres_data:/var/lib/postgresql/data \
        postgres:15
    print_success "PostgreSQL database started"
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
else
    print_warning "PostgreSQL database already running"
fi

# Setup Redis
print_status "Setting up Redis..."
if ! docker ps | grep -q research_redis; then
    docker run -d \
        --name research_redis \
        -p 6379:6379 \
        redis:7-alpine
    print_success "Redis started"
else
    print_warning "Redis already running"
fi

# Initialize database schema
print_status "Initializing database schema..."
cd backend
source venv/bin/activate

# Create database migration
python -c "
from app.db.models import Base
from app.core.config import get_settings
from sqlalchemy import create_engine

settings = get_settings()
engine = create_engine(settings.DATABASE_URL)
Base.metadata.create_all(bind=engine)
print('Database schema created successfully')
"

cd ..
print_success "Database schema initialized"

# Build and start services with Docker Compose
print_status "Building and starting all services..."
docker-compose build
docker-compose up -d

print_success "All services started!"

# Create launch scripts
print_status "Creating launch scripts..."

# Backend launch script
cat > start_backend.sh << 'EOL'
#!/bin/bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOL
chmod +x start_backend.sh

# Frontend launch script
cat > start_frontend.sh << 'EOL'
#!/bin/bash
cd frontend
npm run dev
EOL
chmod +x start_frontend.sh

# Development launch script
cat > start_dev.sh << 'EOL'
#!/bin/bash
echo "Starting Enhanced Research Assistant in development mode..."

# Start backend in background
echo "Starting backend..."
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend in background
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Services started!"
echo "📊 Backend API: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap 'echo "Stopping services..."; kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait
EOL
chmod +x start_dev.sh

print_success "Launch scripts created"

# Create test script
cat > test_setup.sh << 'EOL'
#!/bin/bash
echo "🧪 Testing Enhanced Research Assistant setup..."

# Test backend health
echo "Testing backend health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$response" = "200" ]; then
    echo "✅ Backend is running"
else
    echo "❌ Backend not responding (HTTP $response)"
fi

# Test frontend
echo "Testing frontend..."
frontend_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$frontend_response" = "200" ]; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend not responding (HTTP $frontend_response)"
fi

# Test database connection
echo "Testing database connection..."
cd backend
source venv/bin/activate
python -c "
from app.db.session import get_db
try:
    next(get_db())
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
cd ..

echo "🏁 Testing complete!"
EOL
chmod +x test_setup.sh

print_success "Test script created"

echo ""
echo "=================================================="
print_success "Enhanced Research Assistant v2.0 Setup Complete!"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo "1. Update .env file with your API keys"
echo "2. Run: ./start_dev.sh to start in development mode"
echo "3. Open http://localhost:3000 to access the application"
echo "4. Check http://localhost:8000/docs for API documentation"
echo ""
echo "🔧 Available Commands:"
echo "  ./start_backend.sh  - Start backend only"
echo "  ./start_frontend.sh - Start frontend only"
echo "  ./start_dev.sh      - Start both services"
echo "  ./test_setup.sh     - Test the setup"
echo ""
echo "📚 Services:"
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Database:  localhost:5432"
echo "  Redis:     localhost:6379"
echo ""
print_warning "Remember to configure your API keys in the .env file!"
print_success "Happy researching! 🎓✨"
