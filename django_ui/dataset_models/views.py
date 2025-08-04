from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Q
from django.views.decorators.http import require_http_methods
import json

from .models import Dataset, MLModel, ResearchField, SearchHistory


def dataset_search(request):
    """Search for datasets with filtering options"""
    query = request.GET.get('q', '')
    research_field = request.GET.get('field', '')
    dataset_type = request.GET.get('type', '')
    license_type = request.GET.get('license', '')
    min_samples = request.GET.get('min_samples', '')
    max_size = request.GET.get('max_size', '')
    
    # Start with all active datasets
    datasets = Dataset.objects.filter(is_active=True)
    
    # Apply text search
    if query:
        datasets = datasets.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(tags__icontains=query) |
            Q(keywords__icontains=query)
        )
    
    # Apply filters
    if research_field:
        datasets = datasets.filter(research_fields__name__icontains=research_field)
    
    if dataset_type:
        datasets = datasets.filter(dataset_type=dataset_type)
    
    if license_type:
        datasets = datasets.filter(license_type=license_type)
    
    if min_samples:
        try:
            datasets = datasets.filter(num_samples__gte=int(min_samples))
        except ValueError:
            pass
    
    if max_size:
        try:
            datasets = datasets.filter(size_mb__lte=float(max_size))
        except ValueError:
            pass
    
    # Distinct and order results
    datasets = datasets.distinct().order_by('-citation_count', '-download_count', 'name')
    
    # Save search history
    if query or research_field:
        search_field = None
        if research_field:
            search_field = ResearchField.objects.filter(name__icontains=research_field).first()
        
        SearchHistory.objects.create(
            search_type='dataset',
            query=query or research_field,
            research_field=search_field,
            filters={
                'dataset_type': dataset_type,
                'license_type': license_type,
                'min_samples': min_samples,
                'max_size': max_size,
            },
            results_count=datasets.count()
        )
    
    # Pagination
    paginator = Paginator(datasets, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get filter options
    research_fields = ResearchField.objects.all()
    dataset_types = Dataset.DATASET_TYPES
    license_types = Dataset.LICENSE_TYPES
    
    context = {
        'datasets': page_obj,
        'query': query,
        'research_fields': research_fields,
        'dataset_types': dataset_types,
        'license_types': license_types,
        'selected_field': research_field,
        'selected_type': dataset_type,
        'selected_license': license_type,
        'min_samples': min_samples,
        'max_size': max_size,
    }
    
    return render(request, 'dataset_models/dataset_search.html', context)


def model_search(request):
    """Search for ML models with filtering options"""
    query = request.GET.get('q', '')
    research_field = request.GET.get('field', '')
    model_type = request.GET.get('type', '')
    framework = request.GET.get('framework', '')
    pretrained = request.GET.get('pretrained', '')
    min_accuracy = request.GET.get('min_accuracy', '')
    max_size = request.GET.get('max_size', '')
    
    # Start with all active models
    models = MLModel.objects.filter(is_active=True)
    
    # Apply text search
    if query:
        models = models.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(architecture__icontains=query) |
            Q(tags__icontains=query) |
            Q(keywords__icontains=query)
        )
    
    # Apply filters
    if research_field:
        models = models.filter(research_fields__name__icontains=research_field)
    
    if model_type:
        models = models.filter(model_type=model_type)
    
    if framework:
        models = models.filter(framework=framework)
    
    if pretrained == 'true':
        models = models.filter(pretrained=True)
    elif pretrained == 'false':
        models = models.filter(pretrained=False)
    
    if min_accuracy:
        try:
            models = models.filter(accuracy__gte=float(min_accuracy))
        except ValueError:
            pass
    
    if max_size:
        try:
            models = models.filter(model_size_mb__lte=float(max_size))
        except ValueError:
            pass
    
    # Distinct and order results
    models = models.distinct().order_by('-citation_count', '-download_count', 'name')
    
    # Save search history
    if query or research_field:
        search_field = None
        if research_field:
            search_field = ResearchField.objects.filter(name__icontains=research_field).first()
        
        SearchHistory.objects.create(
            search_type='model',
            query=query or research_field,
            research_field=search_field,
            filters={
                'model_type': model_type,
                'framework': framework,
                'pretrained': pretrained,
                'min_accuracy': min_accuracy,
                'max_size': max_size,
            },
            results_count=models.count()
        )
    
    # Pagination
    paginator = Paginator(models, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get filter options
    research_fields = ResearchField.objects.all()
    model_types = MLModel.MODEL_TYPES
    frameworks = MLModel.FRAMEWORKS
    
    context = {
        'models': page_obj,
        'query': query,
        'research_fields': research_fields,
        'model_types': model_types,
        'frameworks': frameworks,
        'selected_field': research_field,
        'selected_type': model_type,
        'selected_framework': framework,
        'pretrained': pretrained,
        'min_accuracy': min_accuracy,
        'max_size': max_size,
    }
    
    return render(request, 'dataset_models/model_search.html', context)


def combined_search(request):
    """Combined search for both datasets and models"""
    query = request.GET.get('q', '')
    research_field = request.GET.get('field', '')
    
    # Search datasets
    datasets = Dataset.objects.filter(is_active=True)
    if query:
        datasets = datasets.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(tags__icontains=query) |
            Q(keywords__icontains=query)
        )
    if research_field:
        datasets = datasets.filter(research_fields__name__icontains=research_field)
    
    # Search models
    models = MLModel.objects.filter(is_active=True)
    if query:
        models = models.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(architecture__icontains=query) |
            Q(tags__icontains=query) |
            Q(keywords__icontains=query)
        )
    if research_field:
        models = models.filter(research_fields__name__icontains=research_field)
    
    # Limit results for combined view
    datasets = datasets.distinct().order_by('-citation_count')[:6]
    models = models.distinct().order_by('-citation_count')[:6]
    
    # Save search history
    if query or research_field:
        search_field = None
        if research_field:
            search_field = ResearchField.objects.filter(name__icontains=research_field).first()
        
        SearchHistory.objects.create(
            search_type='combined',
            query=query or research_field,
            research_field=search_field,
            results_count=len(datasets) + len(models)
        )
    
    # Get filter options
    research_fields = ResearchField.objects.all()
    
    context = {
        'datasets': datasets,
        'models': models,
        'query': query,
        'research_fields': research_fields,
        'selected_field': research_field,
    }
    
    return render(request, 'dataset_models/combined_search.html', context)


def dataset_detail(request, dataset_id):
    """Dataset detail view"""
    dataset = get_object_or_404(Dataset, id=dataset_id, is_active=True)
    
    # Get related models trained on this dataset
    related_models = dataset.trained_models.filter(is_active=True)[:5]
    
    context = {
        'dataset': dataset,
        'related_models': related_models,
    }
    
    return render(request, 'dataset_models/dataset_detail.html', context)


def model_detail(request, model_id):
    """Model detail view"""
    model = get_object_or_404(MLModel, id=model_id, is_active=True)
    
    # Get training datasets
    training_datasets = model.training_datasets.filter(is_active=True)
    
    context = {
        'model': model,
        'training_datasets': training_datasets,
    }
    
    return render(request, 'dataset_models/model_detail.html', context)


@require_http_methods(["GET"])
def api_search_suggestions(request):
    """API endpoint for search suggestions"""
    query = request.GET.get('q', '').strip()
    search_type = request.GET.get('type', 'all')  # 'dataset', 'model', or 'all'
    
    suggestions = []
    
    if len(query) >= 2:
        if search_type in ['dataset', 'all']:
            # Dataset suggestions
            dataset_suggestions = Dataset.objects.filter(
                Q(name__icontains=query) | Q(tags__icontains=query),
                is_active=True
            ).values_list('name', flat=True)[:5]
            suggestions.extend([{'type': 'dataset', 'text': name} for name in dataset_suggestions])
        
        if search_type in ['model', 'all']:
            # Model suggestions
            model_suggestions = MLModel.objects.filter(
                Q(name__icontains=query) | Q(architecture__icontains=query),
                is_active=True
            ).values_list('name', flat=True)[:5]
            suggestions.extend([{'type': 'model', 'text': name} for name in model_suggestions])
        
        # Research field suggestions
        field_suggestions = ResearchField.objects.filter(
            name__icontains=query
        ).values_list('name', flat=True)[:3]
        suggestions.extend([{'type': 'field', 'text': name} for name in field_suggestions])
    
    return JsonResponse({'suggestions': suggestions})


@require_http_methods(["POST"])
def api_generate_dataset_analysis(request):
    """API endpoint to generate AI analysis for a dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        dataset_name = data.get('dataset_name', 'Unknown Dataset')
        prompt = data.get('prompt', '')
        
        # Get the dataset
        dataset = get_object_or_404(Dataset, id=dataset_id, is_active=True)
        
        # For now, return a simulated analysis
        # In a real implementation, you would call OpenAI API here
        analysis = f"""
**Dataset Analysis: {dataset.name}**

**1. Dataset Characteristics and Structure**
• Type: {dataset.get_dataset_type_display()}
• Scale: {dataset.num_samples or 'Unknown'} samples
• Size: {dataset.size_mb or 'Unknown'} MB
• License: {dataset.get_license_type_display()}

**2. Potential Applications and Use Cases**
Based on the dataset type and research fields, this dataset appears suitable for:
• Research in {', '.join([field.name for field in dataset.research_fields.all()]) if dataset.research_fields.exists() else 'various domains'}
• Training and evaluation of machine learning models
• Benchmarking and comparative studies

**3. Quality Assessment**
• Publication year: {dataset.publication_year or 'Not specified'}
• Citation count: {dataset.citation_count} citations
• Download popularity: {dataset.download_count} downloads
• Community engagement: {dataset.star_count} stars

**4. Potential Considerations**
• License implications: Review {dataset.get_license_type_display()} terms for intended use
• Data privacy: Ensure compliance with relevant regulations
• Bias considerations: Consider dataset origin and collection methodology

**5. Best Practices for Usage**
• Review original paper and documentation if available
• Validate dataset quality for your specific use case
• Consider preprocessing requirements
• Follow ethical guidelines for data usage

**6. Recommendations**
• Check for updates or newer versions of the dataset
• Consider combining with complementary datasets
• Implement proper data validation and quality checks
• Document your usage and preprocessing steps

*Note: This analysis is generated automatically and should be supplemented with manual review of the dataset documentation and your specific requirements.*
        """
        
        return JsonResponse({
            'success': True,
            'analysis': analysis.strip()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def dataset_creation_wizard(request):
    """Interactive dataset creation wizard with AI assistance"""
    if request.method == 'POST':
        try:
            # Handle dataset creation
            dataset_data = {
                'name': request.POST.get('name'),
                'description': request.POST.get('description'),
                'dataset_type': request.POST.get('dataset_type'),
                'size_mb': float(request.POST.get('size_mb', 0)) if request.POST.get('size_mb') else None,
                'num_samples': int(request.POST.get('num_samples', 0)) if request.POST.get('num_samples') else None,
                'num_classes': int(request.POST.get('num_classes', 0)) if request.POST.get('num_classes') else None,
                'license_type': request.POST.get('license_type', 'public'),
                'publication_year': int(request.POST.get('publication_year', 0)) if request.POST.get('publication_year') else None,
                'source_url': request.POST.get('source_url', ''),
                'download_url': request.POST.get('download_url', ''),
                'paper_url': request.POST.get('paper_url', ''),
                'github_url': request.POST.get('github_url', ''),
            }
            
            # Create the dataset
            dataset = Dataset.objects.create(**dataset_data)
            
            # Handle tags and keywords
            tags = request.POST.get('tags', '').split(',')
            keywords = request.POST.get('keywords', '').split(',')
            authors = request.POST.get('authors', '').split(',')
            
            dataset.tags = [tag.strip() for tag in tags if tag.strip()]
            dataset.keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
            dataset.authors = [author.strip() for author in authors if author.strip()]
            
            # Handle research fields
            research_field_ids = request.POST.getlist('research_fields')
            if research_field_ids:
                dataset.research_fields.set(research_field_ids)
            
            dataset.save()
            
            # Generate automatic analysis if requested
            generate_analysis = request.POST.get('generate_analysis') == 'true'
            
            return JsonResponse({
                'success': True,
                'dataset_id': str(dataset.id),
                'message': 'Dataset created successfully!',
                'redirect_url': f'/datasets/datasets/{dataset.id}/',
                'generate_analysis': generate_analysis
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    # GET request - show the creation form
    research_fields = ResearchField.objects.all().order_by('name')
    dataset_types = Dataset.DATASET_TYPES
    license_types = Dataset.LICENSE_TYPES
    
    context = {
        'research_fields': research_fields,
        'dataset_types': dataset_types,
        'license_types': license_types,
    }
    
    return render(request, 'dataset_models/dataset_creation_wizard.html', context)


@require_http_methods(["POST"])
def api_generate_dataset_suggestions(request):
    """Generate AI suggestions for dataset creation"""
    try:
        data = json.loads(request.body)
        dataset_concept = data.get('concept', '')
        research_field = data.get('research_field', '')
        dataset_type = data.get('dataset_type', '')
        
        # Generate AI suggestions based on the concept
        suggestions = {
            'name_suggestions': [
                f"{dataset_concept}_dataset",
                f"{research_field}_{dataset_concept}",
                f"{dataset_concept}_benchmark",
                f"comprehensive_{dataset_concept}_collection"
            ],
            'description_template': f"""
A comprehensive {dataset_type.lower()} dataset focused on {dataset_concept} in the {research_field} domain. 
This dataset is designed to support research and development in {dataset_concept} applications.

Key Features:
• High-quality {dataset_type.lower()} data
• Diverse samples representing various scenarios
• Well-structured and preprocessed format
• Suitable for machine learning and research applications

Applications:
• Training and evaluation of {dataset_concept} models
• Benchmarking algorithm performance
• Research in {research_field}
• Educational and demonstration purposes
            """.strip(),
            'suggested_tags': [
                dataset_concept.lower(),
                research_field.lower(),
                dataset_type.lower(),
                'machine-learning',
                'research',
                'benchmark'
            ],
            'suggested_keywords': [
                dataset_concept,
                research_field,
                'artificial intelligence',
                'machine learning',
                'data science',
                'research dataset'
            ],
            'quality_checklist': [
                'Ensure data quality and consistency',
                'Verify proper licensing and permissions',
                'Include comprehensive documentation',
                'Provide clear usage guidelines',
                'Consider privacy and ethical implications',
                'Validate data accuracy and completeness'
            ],
            'recommended_size': _get_recommended_size(dataset_type, dataset_concept),
            'best_practices': [
                'Use consistent naming conventions',
                'Provide metadata for all samples',
                'Include train/validation/test splits',
                'Document preprocessing steps',
                'Consider data augmentation possibilities',
                'Ensure reproducibility'
            ]
        }
        
        return JsonResponse({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def _get_recommended_size(dataset_type, concept):
    """Helper function to recommend dataset size based on type and concept"""
    size_recommendations = {
        'image': {'small': '100-1K images', 'medium': '1K-10K images', 'large': '10K+ images'},
        'text': {'small': '1K-10K samples', 'medium': '10K-100K samples', 'large': '100K+ samples'},
        'audio': {'small': '100-1K clips', 'medium': '1K-10K clips', 'large': '10K+ clips'},
        'video': {'small': '50-500 videos', 'medium': '500-5K videos', 'large': '5K+ videos'},
        'tabular': {'small': '1K-10K rows', 'medium': '10K-1M rows', 'large': '1M+ rows'},
    }
    
    return size_recommendations.get(dataset_type.lower(), {
        'small': '1K-10K samples',
        'medium': '10K-100K samples', 
        'large': '100K+ samples'
    })
