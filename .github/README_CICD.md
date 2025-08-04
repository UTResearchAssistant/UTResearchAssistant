# AI Research Assistant - GitHub Actions CI/CD Setup

## Overview

This project includes comprehensive GitHub Actions workflows for continuous integration, deployment, and monitoring of the AI Research Assistant system.

## Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Triggers:** Push to main/develop, Pull requests to main
**Features:**

- **Backend Testing:** Django tests with PostgreSQL and Redis
- **Frontend Testing:** Next.js build and test suite
- **AI Services Testing:** Validation of all AI service integrations
- **Security Scanning:** Bandit, Safety, and Semgrep security analysis
- **Docker Building:** Multi-stage builds for all services
- **Deployment:** Automated staging and production deployments

### 2. Model Training Pipeline (`.github/workflows/model-training.yml`)

**Triggers:** Weekly schedule, Manual dispatch
**Features:**

- **Data Preparation:** Automated training data preparation
- **Model Training:** Support for embedding, classification, and summarization models
- **Evaluation:** Comprehensive model performance evaluation
- **Deployment:** Automated model registry updates

### 3. Notebook Validation (`.github/workflows/notebook-validation.yml`)

**Triggers:** Changes to notebook/, services/, or agents/ directories
**Features:**

- **Notebook Testing:** Automated Jupyter notebook execution and validation
- **AI Integration Testing:** End-to-end AI service integration tests
- **Agent Coordination Testing:** Multi-agent workflow validation

### 4. Performance Testing (`.github/workflows/performance-testing.yml`)

**Triggers:** Daily schedule, Manual dispatch
**Features:**

- **Load Testing:** Locust-based load testing with configurable parameters
- **Benchmark Testing:** Performance regression detection
- **Memory Profiling:** Memory usage analysis and optimization

### 5. Data Pipeline (`.github/workflows/data-pipeline.yml`)

**Triggers:** Daily schedule, Manual dispatch
**Features:**

- **Data Ingestion:** Automated data collection from academic sources
- **Data Processing:** Embedding generation and index updates
- **Quality Assurance:** Data quality checks and validation

### 6. Code Quality (`.github/workflows/code-quality.yml`)

**Triggers:** Push to main/develop, Pull requests
**Features:**

- **Code Formatting:** Black, isort validation
- **Linting:** Flake8, Pylint analysis
- **Type Checking:** MyPy static analysis
- **Security:** Bandit, Safety vulnerability scanning
- **SonarCloud:** Code quality metrics and technical debt analysis
- **CodeQL:** GitHub's semantic code analysis

## Setup Requirements

### Repository Secrets

Add these secrets to your GitHub repository:

```bash
# Required secrets
OPENAI_API_KEY          # OpenAI API key for AI services
SONAR_TOKEN            # SonarCloud token for code analysis
SLACK_WEBHOOK          # Slack webhook for notifications
WANDB_API_KEY          # Weights & Biases for model tracking

# Optional secrets for advanced features
DOCKER_REGISTRY_TOKEN  # Docker registry authentication
STAGING_SSH_KEY        # SSH key for staging deployment
PRODUCTION_SSH_KEY     # SSH key for production deployment
```

### Environment Configuration

Create environment-specific configurations:

#### Staging Environment

- **Name:** staging
- **Protection Rules:** None
- **Secrets:** Staging-specific configuration

#### Production Environment

- **Name:** production
- **Protection Rules:** Required reviewers, deployment branches
- **Secrets:** Production configuration and credentials

## Docker Configuration

### Multi-Service Architecture

The system includes optimized Dockerfiles for:

1. **Django Backend** (`django_ui/Dockerfile`)

   - Python 3.13 slim base
   - Production-ready with Gunicorn
   - Non-root user for security
   - Static file collection

2. **FastAPI Service** (`backend/Dockerfile`)

   - Async-ready Python environment
   - Uvicorn ASGI server
   - Multi-worker configuration

3. **Next.js Frontend** (`frontend/Dockerfile`)
   - Multi-stage build for optimization
   - Nginx serving with caching
   - API proxy configuration

### Docker Compose for CI

Use `docker-compose.ci.yml` for testing environments:

- PostgreSQL with health checks
- Redis for caching
- Proper service dependencies
- Volume management

## Performance Testing

### Load Testing with Locust

The system includes comprehensive load testing scenarios:

```python
# Example usage
locust --host=http://localhost:8000 \
       --users=50 \
       --spawn-rate=10 \
       --run-time=10m \
       --headless \
       -f services/tests/locustfile.py
```

### Test Scenarios

- **Literature Search:** Multi-source academic paper searching
- **AI Generation:** Podcast and content generation
- **Writing Assistance:** Academic writing improvement
- **Citation Management:** Reference formatting and validation
- **Academic Integrity:** Plagiarism detection workflows

### Benchmark Testing

Automated benchmark tests track:

- **Response Times:** API endpoint performance
- **Memory Usage:** Memory efficiency under load
- **Database Performance:** Query optimization
- **AI Model Latency:** Model inference speed

## Security and Quality

### Automated Security Scanning

- **Dependency Scanning:** Automated vulnerability detection
- **Code Analysis:** Static analysis for security issues
- **Container Scanning:** Docker image vulnerability assessment
- **Secrets Detection:** Prevention of credential leaks

### Code Quality Standards

- **Formatting:** Black code formatter with 127-char line limit
- **Import Sorting:** isort for consistent import organization
- **Linting:** Flake8 and Pylint for code quality
- **Type Checking:** MyPy for static type analysis

### Quality Gates

Pull requests must pass:

- All automated tests
- Security scans
- Code quality checks
- Performance benchmarks (no regression)

## Deployment Strategy

### Staging Deployment

- **Trigger:** Push to develop branch
- **Environment:** staging.research-assistant.com
- **Database:** Staging PostgreSQL instance
- **Monitoring:** Basic health checks

### Production Deployment

- **Trigger:** Push to main branch
- **Environment:** research-assistant.com
- **Database:** Production PostgreSQL cluster
- **Monitoring:** Full observability stack
- **Rollback:** Automated rollback on failure

### Blue-Green Deployment

For zero-downtime deployments:

1. Deploy to green environment
2. Run health checks
3. Switch traffic from blue to green
4. Monitor for issues
5. Keep blue environment for rollback

## Monitoring and Notifications

### Slack Integration

Automated notifications for:

- Deployment status
- Test failures
- Security alerts
- Performance degradation

### Metrics Collection

- **Application Metrics:** Custom business metrics
- **Infrastructure Metrics:** System resource usage
- **Security Metrics:** Vulnerability trends
- **Quality Metrics:** Code coverage, technical debt

## Usage

### Manual Workflow Dispatch

Trigger workflows manually with custom parameters:

```bash
# Model training with specific configuration
gh workflow run model-training.yml \
  -f model_type=embedding \
  -f dataset_path=datasets/custom_data

# Performance testing with custom load
gh workflow run performance-testing.yml \
  -f test_duration=30 \
  -f concurrent_users=100

# Data pipeline for specific sources
gh workflow run data-pipeline.yml \
  -f data_source=arxiv \
  -f full_refresh=true
```

### Local Development

Run the same tests locally:

```bash
# Code quality checks
black --check .
flake8 .
mypy .
bandit -r .

# Run tests
python -m pytest django_ui/tests/
python -m pytest services/tests/

# Build Docker images
docker-compose -f docker-compose.ci.yml build
```

## Troubleshooting

### Common Issues

1. **Test Failures:** Check logs in Actions tab
2. **Docker Build Failures:** Verify Dockerfile syntax
3. **Deployment Failures:** Check environment secrets
4. **Performance Regressions:** Review benchmark reports

### Debug Commands

```bash
# View workflow logs
gh run list --workflow=ci-cd.yml
gh run view <run-id> --log

# Local debugging
docker-compose -f docker-compose.ci.yml up --build
docker-compose logs <service-name>
```

This comprehensive CI/CD setup ensures reliable, secure, and performant deployment of your AI Research Assistant system with full automation and monitoring capabilities.
