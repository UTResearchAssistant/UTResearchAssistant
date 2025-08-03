#!/bin/bash

# Research Assistant - Comprehensive Setup and Installation Script
# This script ensures all modules are compatible and working with Django

echo "🚀 Research Assistant - Complete Setup"
echo "======================================"

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

# Check if we're in the right directory
if [ ! -f "manage.py" ]; then
    print_error "Please run this script from the django_ui directory"
    exit 1
fi

print_status "Checking Python virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment found"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
print_status "Installing core dependencies..."
pip install Django>=4.2,\<5.0
pip install djangorestframework>=3.14.0
pip install django-cors-headers>=4.3.0
pip install requests>=2.31.0
pip install numpy>=1.24.3
pip install pandas>=2.0.3

# Install AI/ML dependencies
print_status "Installing AI/ML dependencies..."
pip install scikit-learn>=1.3.0
pip install transformers>=4.35.2
pip install torch>=2.1.1
pip install accelerate>=0.25.0

# Install optional dependencies with fallback
print_status "Installing optional dependencies..."

# Try to install spacy
if pip install spacy>=3.7.2; then
    print_success "SpaCy installed"
    # Try to download English model
    if python -m spacy download en_core_web_sm; then
        print_success "SpaCy English model downloaded"
    else
        print_warning "Could not download SpaCy model (optional)"
    fi
else
    print_warning "Could not install SpaCy (optional dependency)"
fi

# Install academic search dependencies
print_status "Installing academic search dependencies..."
pip install arxiv>=1.4.8
pip install scholarly>=1.7.11

# Install document processing dependencies
print_status "Installing document processing dependencies..."
pip install PyPDF2>=3.0.1
pip install python-docx>=1.1.0
pip install beautifulsoup4>=4.12.2
pip install lxml>=4.9.3

# Install async dependencies
print_status "Installing async dependencies..."
pip install aiohttp>=3.9.0

# Install all requirements from file
print_status "Installing remaining requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "All requirements installed"
else
    print_warning "requirements.txt not found"
fi

# Django setup
print_status "Setting up Django..."

# Collect static files
print_status "Collecting static files..."
python manage.py collectstatic --noinput

# Run migrations
print_status "Running database migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser if needed
print_status "Checking for superuser..."
if python manage.py shell -c "from django.contrib.auth.models import User; print('exists' if User.objects.filter(is_superuser=True).exists() else 'none')"; then
    print_success "Superuser exists"
else
    print_status "Creating superuser (admin/admin123)..."
    python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Admin user already exists')
"
fi

# Test integration
print_status "Testing service integration..."
python manage.py test_integration --test-type=services

# Check system status
print_status "Checking system status..."
python -c "
import sys
import pkg_resources

print('\\n📦 Installed Packages:')
print('=' * 40)

packages = [
    'Django', 'numpy', 'pandas', 'scikit-learn', 
    'transformers', 'torch', 'requests', 'aiohttp',
    'arxiv', 'scholarly', 'PyPDF2', 'beautifulsoup4'
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f'✅ {package}: {version}')
    except pkg_resources.DistributionNotFound:
        print(f'❌ {package}: Not installed')

print('\\n🔧 System Information:')
print('=' * 40)
print(f'Python Version: {sys.version}')
print(f'Virtual Environment: Active')
"

# Final success message
echo ""
print_success "🎉 Research Assistant setup completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Start the Django server: python manage.py runserver"
echo "  2. Access the application at: http://127.0.0.1:8000/"
echo "  3. Admin panel at: http://127.0.0.1:8000/admin/ (admin/admin123)"
echo "  4. Run integration tests: python manage.py test_integration"
echo ""
print_status "For Llama 3.2 3B access:"
echo "  1. Visit: https://huggingface.co/meta-llama/Llama-3.2-3B"
echo "  2. Request access to the model"
echo "  3. Login with: huggingface-cli login"
echo "  4. The system will automatically use fallback methods if unavailable"
echo ""
print_success "Setup complete! 🚀"
