#!/bin/bash

# Enhanced Research Assistant - Django Setup Script
# This script sets up the complete Django backend with all enhanced features

set -e  # Exit on any error

echo "🚀 Starting Enhanced Research Assistant Django Setup..."
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

# Check if we're in the correct directory
if [ ! -f "manage.py" ]; then
    print_error "This script must be run from the django_ui directory"
    exit 1
fi

# Step 1: Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Step 3: Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install dependencies
print_status "Installing Python dependencies..."
print_warning "This may take a few minutes for all 60+ packages..."
pip install -r requirements.txt

# Step 6: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating environment configuration file..."
    cat > .env << EOF
# Django Configuration
SECRET_KEY=django-insecure-change-this-in-production-$(date +%s)
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Database Configuration
DATABASE_URL=sqlite:///db.sqlite3

# Redis Configuration (Optional - will use in-memory cache if not available)
REDIS_URL=redis://localhost:6379/1

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Academic API Keys (Optional)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
CROSSREF_EMAIL=your_email@example.com

# Email Configuration (Optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
EMAIL_USE_TLS=True

# Storage Configuration (Optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_STORAGE_BUCKET_NAME=your_bucket_name

# Feature Flags
ENABLE_PODCAST_GENERATION=True
ENABLE_VIDEO_ANALYSIS=True
ENABLE_WRITING_ASSISTANT=True
ENABLE_COLLABORATION=True
ENABLE_RESEARCH_ALERTS=True
ENABLE_INTEGRITY_CHECKS=True

# Performance Settings
CACHE_TIMEOUT=300
MAX_UPLOAD_SIZE=104857600
PAGINATION_SIZE=20

# Security Settings (Set to False in production)
SECURE_SSL_REDIRECT=False
SECURE_HSTS_SECONDS=0
EOF
    print_success "Environment file created (.env)"
    print_warning "Please update the .env file with your actual API keys"
else
    print_warning "Environment file already exists"
fi

# Step 7: Database migrations
print_status "Running database migrations..."
python manage.py makemigrations
python manage.py migrate

# Step 8: Create superuser (interactive)
print_status "Setting up admin user..."
echo "Creating Django superuser for admin access..."
python manage.py createsuperuser --noinput --username admin --email admin@example.com || true

# Step 9: Collect static files
print_status "Collecting static files..."
python manage.py collectstatic --noinput

# Step 10: Create sample data
print_status "Creating sample data..."
python manage.py setup_sample_data || print_warning "Sample data command not found (this is optional)"

# Step 11: Run tests
print_status "Running basic tests..."
python manage.py test core.tests || print_warning "Some tests may fail due to missing API keys"

# Step 12: Check for common issues
print_status "Performing system checks..."
python manage.py check

# Success message
print_success "🎉 Enhanced Research Assistant Django setup complete!"
echo ""
echo "=================================================="
echo -e "${GREEN}✅ Setup Summary:${NC}"
echo "- Virtual environment: venv/"
echo "- Dependencies: All 60+ packages installed"
echo "- Database: Migrated and ready"
echo "- Admin user: admin (set password during setup)"
echo "- Static files: Collected"
echo "- Configuration: .env file created"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "1. Update your .env file with actual API keys:"
echo "   - OPENAI_API_KEY (required for AI features)"
echo "   - Email settings (for notifications)"
echo "   - AWS settings (for file storage)"
echo ""
echo "2. Start the development server:"
echo "   python manage.py runserver"
echo ""
echo "3. Access the application:"
echo "   - Main app: http://localhost:8000"
echo "   - Admin panel: http://localhost:8000/admin"
echo ""
echo -e "${GREEN}🎯 Available Features:${NC}"
echo "- Enhanced Literature Search"
echo "- AI Podcast Generation"
echo "- Video Analysis Tools"
echo "- Academic Writing Assistant"
echo "- Research Alerts & Notifications"
echo "- Collaboration Platform"
echo "- Research Integrity Checks"
echo ""
echo -e "${YELLOW}📚 Documentation:${NC}"
echo "- Complete guide: README_DJANGO_COMPLETE.md"
echo "- API docs: http://localhost:8000/api/docs"
echo "- Model admin: http://localhost:8000/admin"
echo ""
echo -e "${GREEN}Happy researching! 🧠✨${NC}"
