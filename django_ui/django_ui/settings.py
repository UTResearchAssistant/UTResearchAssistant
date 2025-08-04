"""Enhanced Django settings for the Research Assistant project.

This configuration includes all necessary settings for the enhanced features
including database, authentication, media files, and third-party integrations.
"""

from pathlib import Path
import os
import sys
import environ as django_environ

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# Add the parent directory to Python path so we can import from agents/ and services/
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Environment variables
env = django_environ.Env(
    DEBUG=(bool, True),
    ENABLE_PODCAST_GENERATION=(bool, True),
    ENABLE_VIDEO_ANALYSIS=(bool, True),
    ENABLE_MULTILINGUAL_SEARCH=(bool, True),
)

# Read environment file
django_environ.Env.read_env(BASE_DIR.parent / ".env")

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env("SECRET_KEY", default="django-insecure-change-me-in-production")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG", default=True)

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["*"])

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "self_explanation.apps.SelfExplanationConfig",
    "corsheaders",
    "core",
    "literature_search",
    "podcast_generation", 
    "video_analysis",
    "writing_assistance",
    "academic_integrity",
    "citation_management",
    "collaboration",
    "alerts",
    "dataset_models",
    "training_config",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "django_ui.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "django_ui.wsgi.application"

# Database
# Use DATABASE_URL if provided, otherwise fallback to individual settings
try:
    DATABASES = {"default": env.db_url("DATABASE_URL")}
except:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("POSTGRES_DB", default="research_assistant"),
            "USER": env("POSTGRES_USER", default="research_user"),
            "PASSWORD": env("POSTGRES_PASSWORD", default="research_password"),
            "HOST": env("DATABASE_HOST", default="localhost"),
            "PORT": env("DATABASE_PORT", default="5432"),
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Authentication URLs
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

# Django REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
}

# CORS settings
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# API Keys and External Services
OPENAI_API_KEY = env("OPENAI_API_KEY", default="")
GOOGLE_TRANSLATE_API_KEY = env("GOOGLE_TRANSLATE_API_KEY", default="")
SEMANTIC_SCHOLAR_API_KEY = env("SEMANTIC_SCHOLAR_API_KEY", default="")
CROSSREF_API_KEY = env("CROSSREF_API_KEY", default="")

# Feature Flags
ENABLE_PODCAST_GENERATION = env("ENABLE_PODCAST_GENERATION", default=True)
ENABLE_VIDEO_ANALYSIS = env("ENABLE_VIDEO_ANALYSIS", default=True)
ENABLE_MULTILINGUAL_SEARCH = env("ENABLE_MULTILINGUAL_SEARCH", default=True)
ENABLE_RESEARCH_ALERTS = env("ENABLE_RESEARCH_ALERTS", default=True)
ENABLE_WRITING_ASSISTANT = env("ENABLE_WRITING_ASSISTANT", default=True)
ENABLE_COLLABORATION = env("ENABLE_COLLABORATION", default=True)
ENABLE_INTEGRITY_CHECKING = env("ENABLE_INTEGRITY_CHECKING", default=True)

# Email configuration
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = env("SMTP_HOST", default="smtp.gmail.com")
EMAIL_PORT = env.int("SMTP_PORT", default=587)
EMAIL_USE_TLS = True
EMAIL_HOST_USER = env("SMTP_USERNAME", default="")
EMAIL_HOST_PASSWORD = env("SMTP_PASSWORD", default="")
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", default="research-assistant@example.com")

# Celery Configuration (for background tasks)
CELERY_BROKER_URL = env("REDIS_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("REDIS_URL", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE

# Logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "django.log",
            "formatter": "verbose",
        },
        "console": {
            "level": "DEBUG" if DEBUG else "INFO",
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "core": {
            "handlers": ["console", "file"],
            "level": "DEBUG" if DEBUG else "INFO",
            "propagate": False,
        },
    },
}

# Create logs directory if it doesn't exist
os.makedirs(BASE_DIR / "logs", exist_ok=True)

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_REDIRECT_EXEMPT = []
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

# Session settings
SESSION_ENGINE = "django.contrib.sessions.backends.db"
SESSION_COOKIE_AGE = 86400  # 1 day
SESSION_EXPIRE_AT_BROWSER_CLOSE = False

# Cache configuration
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": env("REDIS_URL", default="redis://127.0.0.1:6379/1"),
    }
}

# Message framework
from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.DEBUG: "debug",
    messages.INFO: "info",
    messages.SUCCESS: "success",
    messages.WARNING: "warning",
    messages.ERROR: "danger",
}

# Application-specific settings
MAX_SEARCH_RESULTS = 100
MAX_PODCAST_DURATION = 3600  # 1 hour
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
AUDIO_OUTPUT_FORMAT = "mp3"
VIDEO_OUTPUT_FORMAT = "mp4"

# Default user preferences
DEFAULT_USER_PREFERENCES = {
    "max_search_results": 20,
    "preferred_languages": ["en"],
    "email_notifications": True,
    "web_notifications": True,
    "notification_frequency": "weekly",
    "preferred_podcast_style": "summary",
    "preferred_voice": "alloy",
    "theme": "light",
    "items_per_page": 25,
}
