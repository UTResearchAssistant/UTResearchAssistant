#!/bin/bash

# Enhanced Research Assistant Django Setup Script
# This script sets up the complete Django-based research assistant

set -e

echo "🚀 Setting up Enhanced Research Assistant v2.0 (Django Backend)..."
echo "=================================================================="

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

if ! command_exists docker; then
    print_error "Docker is required but not installed."
    exit 1
fi

print_success "All prerequisites found!"

# Setup Python virtual environment for Django
print_status "Setting up Django Python virtual environment..."
cd django_ui
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
print_status "Installing Django dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Django dependencies installed"

# Go back to project root
cd ..

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p datasets/metadata
mkdir -p datasets/papers_processed
mkdir -p datasets/papers_raw
mkdir -p datasets/training_data
mkdir -p datasets/user_uploads
mkdir -p django_ui/logs
mkdir -p django_ui/media
mkdir -p django_ui/media/podcasts
mkdir -p django_ui/media/profiles
mkdir -p django_ui/media/temp
mkdir -p django_ui/staticfiles
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
DATABASE_HOST=localhost
DATABASE_PORT=5432

# Django Configuration
DJANGO_SECRET_KEY=your_django_secret_key_here_$(openssl rand -base64 32)
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Google Services (Optional)
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here

# Academic APIs (Optional)
ARXIV_API_KEY=
CROSSREF_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=

# Email Configuration (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
DEFAULT_FROM_EMAIL=research-assistant@example.com

# Redis Configuration
REDIS_URL=redis://localhost:6379

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

# Application Settings
MAX_SEARCH_RESULTS=100
MAX_PODCAST_DURATION=3600
EOL
    print_success "Environment file created (.env)"
    print_warning "Please update the .env file with your API keys and configuration"
else
    print_warning ".env file already exists"
fi

# Setup database with Docker
print_status "Setting up PostgreSQL database..."
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

# Initialize Django application
print_status "Initializing Django application..."
cd django_ui
source venv/bin/activate

# Run Django migrations
print_status "Running Django migrations..."
python manage.py makemigrations
python manage.py migrate
print_success "Django migrations completed"

# Create superuser
print_status "Setting up Django admin..."
python manage.py setup_research_assistant --create-superuser --load-sample-data
print_success "Django admin setup completed"

# Collect static files
print_status "Collecting static files..."
python manage.py collectstatic --noinput
print_success "Static files collected"

cd ..

# Create launch scripts
print_status "Creating launch scripts..."

# Django launch script
cat > start_django.sh << 'EOL'
#!/bin/bash
cd django_ui
source venv/bin/activate
python manage.py runserver 0.0.0.0:8000
EOL
chmod +x start_django.sh

# Development launch script
cat > start_django_dev.sh << 'EOL'
#!/bin/bash
echo "Starting Enhanced Research Assistant (Django Backend)..."

# Start Django in development mode
echo "Starting Django backend..."
cd django_ui
source venv/bin/activate
python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!
cd ..

echo "✅ Django backend started!"
echo "🌐 Admin Interface: http://localhost:8000/admin"
echo "🏠 Application: http://localhost:8000"
echo "📚 Default Admin Login:"
echo "   Email: admin@research-assistant.com"
echo "   Password: admin123"
echo ""
echo "Press Ctrl+C to stop the service"

# Wait for Ctrl+C
trap 'echo "Stopping Django..."; kill $DJANGO_PID; exit' INT
wait
EOL
chmod +x start_django_dev.sh

# Production launch script
cat > start_django_prod.sh << 'EOL'
#!/bin/bash
echo "Starting Enhanced Research Assistant (Production Mode)..."

cd django_ui
source venv/bin/activate

# Collect static files
python manage.py collectstatic --noinput

# Run migrations
python manage.py migrate

# Start with Gunicorn
gunicorn django_ui.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class sync \
    --timeout 300 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --log-level info \
    --access-logfile - \
    --error-logfile -
EOL
chmod +x start_django_prod.sh

# Create test script
cat > test_django_setup.sh << 'EOL'
#!/bin/bash
echo "🧪 Testing Enhanced Research Assistant Django setup..."

# Test Django health
echo "Testing Django backend..."
cd django_ui
source venv/bin/activate

# Start Django in background for testing
python manage.py runserver 0.0.0.0:8000 &
DJANGO_PID=$!
sleep 5

# Test health endpoint
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
if [ "$response" = "200" ]; then
    echo "✅ Django backend is running"
else
    echo "❌ Django backend not responding (HTTP $response)"
fi

# Test admin interface
admin_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/admin/)
if [ "$admin_response" = "200" ]; then
    echo "✅ Django admin interface is accessible"
else
    echo "❌ Django admin not responding (HTTP $admin_response)"
fi

# Test database connection
python manage.py check --database default
if [ $? -eq 0 ]; then
    echo "✅ Database connection successful"
else
    echo "❌ Database connection failed"
fi

# Stop test server
kill $DJANGO_PID
cd ..

echo "🏁 Testing complete!"
EOL
chmod +x test_django_setup.sh

print_success "Launch scripts created"

# Create Docker Compose for Django
print_status "Creating Docker Compose configuration..."
cat > docker-compose.django.yml << 'EOL'
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: research_assistant
      POSTGRES_USER: research_user
      POSTGRES_PASSWORD: research_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U research_user -d research_assistant"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  django:
    build:
      context: ./django_ui
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - DATABASE_URL=postgresql://research_user:research_password@db:5432/research_assistant
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./django_ui:/app
      - media_files:/app/media
      - static_files:/app/staticfiles
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: >
      sh -c "python manage.py migrate &&
             python manage.py collectstatic --noinput &&
             gunicorn django_ui.wsgi:application --bind 0.0.0.0:8000"

volumes:
  postgres_data:
  media_files:
  static_files:
EOL
print_success "Docker Compose configuration created"

# Create Django Dockerfile
print_status "Creating Django Dockerfile..."
cat > django_ui/Dockerfile << 'EOL'
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/media /app/staticfiles /app/logs

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "django_ui.wsgi:application", "--bind", "0.0.0.0:8000"]
EOL
print_success "Django Dockerfile created"

echo ""
echo "=================================================================="
print_success "Enhanced Research Assistant v2.0 (Django) Setup Complete!"
echo "=================================================================="
echo ""
echo "📋 Next Steps:"
echo "1. Update .env file with your API keys (especially OPENAI_API_KEY)"
echo "2. Run: ./start_django_dev.sh to start in development mode"
echo "3. Open http://localhost:8000 to access the application"
echo "4. Open http://localhost:8000/admin for Django admin interface"
echo ""
echo "🔧 Available Commands:"
echo "  ./start_django.sh       - Start Django backend only"
echo "  ./start_django_dev.sh   - Start in development mode"
echo "  ./start_django_prod.sh  - Start in production mode"
echo "  ./test_django_setup.sh  - Test the setup"
echo ""
echo "📚 Services:"
echo "  Django App:    http://localhost:8000"
echo "  Django Admin:  http://localhost:8000/admin"
echo "  Database:      localhost:5432"
echo "  Redis:         localhost:6379"
echo ""
echo "🔐 Default Admin Account:"
echo "  Email:    admin@research-assistant.com"
echo "  Password: admin123"
echo ""
echo "🐳 Docker Commands:"
echo "  docker-compose -f docker-compose.django.yml up -d  # Start all services"
echo "  docker-compose -f docker-compose.django.yml down   # Stop all services"
echo ""
print_warning "Remember to configure your API keys in the .env file!"
print_success "Happy researching with Django! 🎓✨"
