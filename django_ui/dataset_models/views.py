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
