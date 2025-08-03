"""Django management command to set up the enhanced research assistant."""

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from django.db import connection
from django.conf import settings
import os
import sys


class Command(BaseCommand):
    help = "Set up the enhanced research assistant with initial data and configuration"

    def add_arguments(self, parser):
        parser.add_argument(
            "--create-superuser",
            action="store_true",
            help="Create a superuser account",
        )
        parser.add_argument(
            "--superuser-email",
            type=str,
            default="admin@research-assistant.com",
            help="Email for the superuser account",
        )
        parser.add_argument(
            "--superuser-password",
            type=str,
            default="admin123",
            help="Password for the superuser account",
        )
        parser.add_argument(
            "--load-sample-data",
            action="store_true",
            help="Load sample research data",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS("🚀 Setting up Enhanced Research Assistant...")
        )

        # Check database connection
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            self.stdout.write(self.style.SUCCESS("✅ Database connection successful"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Database connection failed: {e}"))
            return

        # Run migrations
        self.stdout.write("📦 Running database migrations...")
        try:
            from django.core.management import execute_from_command_line

            execute_from_command_line(["manage.py", "migrate"])
            self.stdout.write(self.style.SUCCESS("✅ Database migrations completed"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Migration failed: {e}"))
            return

        # Create superuser if requested
        if options["create_superuser"]:
            self.create_superuser(
                options["superuser_email"], options["superuser_password"]
            )

        # Load sample data if requested
        if options["load_sample_data"]:
            self.load_sample_data()

        # Check required settings
        self.check_settings()

        # Create necessary directories
        self.create_directories()

        self.stdout.write(
            self.style.SUCCESS("🎉 Enhanced Research Assistant setup complete!")
        )
        self.stdout.write(
            self.style.SUCCESS("🌐 Start the server with: python manage.py runserver")
        )

    def create_superuser(self, email, password):
        """Create a superuser account."""
        self.stdout.write("👤 Creating superuser account...")

        try:
            if User.objects.filter(email=email).exists():
                self.stdout.write(
                    self.style.WARNING(f"⚠️ User with email {email} already exists")
                )
                return

            user = User.objects.create_superuser(
                username="admin", email=email, password=password
            )
            user.first_name = "Research"
            user.last_name = "Admin"
            user.save()

            self.stdout.write(self.style.SUCCESS(f"✅ Superuser created: {email}"))
            self.stdout.write(self.style.SUCCESS(f"🔐 Password: {password}"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Failed to create superuser: {e}"))

    def load_sample_data(self):
        """Load sample research data."""
        self.stdout.write("📚 Loading sample research data...")

        try:
            from core.models import Paper, ResearchProject

            # Create sample papers
            sample_papers = [
                {
                    "title": "Machine Learning in Healthcare: A Comprehensive Review",
                    "abstract": "This paper reviews the current state of machine learning applications in healthcare, covering diagnostic imaging, drug discovery, and personalized treatment.",
                    "authors": '["Dr. Jane Smith", "Prof. John Doe", "Dr. Alice Johnson"]',
                    "journal": "Journal of Medical AI",
                    "publication_date": "2024-01-15",
                    "keywords": '["machine learning", "healthcare", "artificial intelligence"]',
                    "language": "en",
                    "is_open_access": True,
                    "citation_count": 42,
                    "source": "sample",
                },
                {
                    "title": "Climate Change Impact on Biodiversity: Global Perspectives",
                    "abstract": "An analysis of how climate change affects biodiversity across different ecosystems and geographical regions.",
                    "authors": '["Dr. Environmental Scientist", "Prof. Climate Researcher"]',
                    "journal": "Nature Climate Change",
                    "publication_date": "2024-02-20",
                    "keywords": '["climate change", "biodiversity", "ecosystem"]',
                    "language": "en",
                    "is_open_access": False,
                    "citation_count": 28,
                    "source": "sample",
                },
                {
                    "title": "Quantum Computing: From Theory to Practice",
                    "abstract": "This review examines the transition of quantum computing from theoretical concepts to practical implementations.",
                    "authors": '["Dr. Quantum Physicist", "Prof. Computer Scientist"]',
                    "journal": "Quantum Information Processing",
                    "publication_date": "2024-03-10",
                    "keywords": '["quantum computing", "quantum mechanics", "algorithms"]',
                    "language": "en",
                    "is_open_access": True,
                    "citation_count": 35,
                    "source": "sample",
                },
            ]

            for paper_data in sample_papers:
                paper, created = Paper.objects.get_or_create(
                    title=paper_data["title"], defaults=paper_data
                )
                if created:
                    self.stdout.write(f"📄 Created paper: {paper.title}")

            self.stdout.write(self.style.SUCCESS("✅ Sample data loaded successfully"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Failed to load sample data: {e}"))

    def check_settings(self):
        """Check required settings and configuration."""
        self.stdout.write("🔧 Checking configuration...")

        # Check API keys
        required_settings = {
            "OPENAI_API_KEY": "OpenAI API (required for AI features)",
            "GOOGLE_TRANSLATE_API_KEY": "Google Translate API (optional)",
            "SEMANTIC_SCHOLAR_API_KEY": "Semantic Scholar API (optional)",
        }

        missing_settings = []
        for setting, description in required_settings.items():
            value = getattr(settings, setting, None)
            if not value:
                missing_settings.append(f"{setting}: {description}")
                self.stdout.write(self.style.WARNING(f"⚠️ Missing: {setting}"))
            else:
                self.stdout.write(self.style.SUCCESS(f"✅ Found: {setting}"))

        if missing_settings:
            self.stdout.write(
                self.style.WARNING(
                    "⚠️ Some API keys are missing. Update your .env file:"
                )
            )
            for setting in missing_settings:
                self.stdout.write(f"   {setting}")
        else:
            self.stdout.write(self.style.SUCCESS("✅ All required settings configured"))

    def create_directories(self):
        """Create necessary directories for the application."""
        self.stdout.write("📁 Creating necessary directories...")

        directories = [
            settings.MEDIA_ROOT,
            settings.MEDIA_ROOT / "podcasts",
            settings.MEDIA_ROOT / "profiles",
            settings.MEDIA_ROOT / "temp",
            settings.BASE_DIR / "logs",
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.stdout.write(f"📁 Created: {directory}")
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"❌ Failed to create {directory}: {e}")
                )
