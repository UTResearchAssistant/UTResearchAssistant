from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import matplotlib.pyplot as plt
import io
import base64
import time
import requests
import asyncio
from .models import (
    HardwareProfile, BaseModelProfile, DatasetProfile, AdapterConfiguration,
    TrainingConfiguration, TrainingExperiment, ConfigurationComparison,
    CloudProvider, CloudTrainingJob, LiveTrainingMetrics, TrainingNotebook
)
from .services import CloudTrainingService, NotebookGenerator


@login_required
def training_dashboard(request):
    """Main training configuration dashboard"""
    # Get user's recent configurations
    recent_configs = TrainingConfiguration.objects.filter(user=request.user)[:5]
    
    # Get summary statistics
    total_configs = TrainingConfiguration.objects.filter(user=request.user).count()
    running_configs = TrainingConfiguration.objects.filter(user=request.user, status='RUNNING').count()
    completed_configs = TrainingConfiguration.objects.filter(user=request.user, status='COMPLETED').count()
    
    context = {
        'recent_configs': recent_configs,
        'total_configs': total_configs,
        'running_configs': running_configs,
        'completed_configs': completed_configs,
    }
    return render(request, 'training_config/dashboard.html', context)


@login_required
def configuration_builder(request):
    """Interactive configuration builder"""
    if request.method == 'POST':
        # Handle configuration creation
        try:
            config = TrainingConfiguration(
                name=request.POST.get('name'),
                user=request.user,
                dataset_id=request.POST.get('dataset'),
                base_model_id=request.POST.get('base_model'),
                adapter_id=request.POST.get('adapter'),
                hardware_id=request.POST.get('hardware'),
                learning_rate=float(request.POST.get('learning_rate', 5e-5)),
                batch_size=int(request.POST.get('batch_size', 32)),
                epochs=int(request.POST.get('epochs', 3)),
                gradient_accumulation_steps=int(request.POST.get('gradient_accumulation_steps', 1)),
                weight_decay=float(request.POST.get('weight_decay', 0.01)),
                warmup_steps=int(request.POST.get('warmup_steps', 100)),
                optimizer=request.POST.get('optimizer', 'ADAMW'),
                precision=request.POST.get('precision', 'FP16'),
            )
            
            # Calculate estimations
            metrics = config.calculate_estimated_metrics()
            config.save()
            
            messages.success(request, f'Configuration "{config.name}" created successfully!')
            return redirect('training_config:configuration_detail', config_id=config.id)
            
        except Exception as e:
            messages.error(request, f'Error creating configuration: {str(e)}')
    
    # Get available options
    datasets = DatasetProfile.objects.filter(is_active=True)
    models = BaseModelProfile.objects.filter(is_active=True)
    adapters = AdapterConfiguration.objects.filter(is_active=True)
    hardware = HardwareProfile.objects.filter(is_active=True)
    
    context = {
        'datasets': datasets,
        'models': models,
        'adapters': adapters,
        'hardware': hardware,
    }
    return render(request, 'training_config/configuration_builder.html', context)


@login_required
def configuration_list(request):
    """List all user configurations with filtering"""
    configurations = TrainingConfiguration.objects.filter(user=request.user)
    
    # Apply filters
    status_filter = request.GET.get('status')
    if status_filter:
        configurations = configurations.filter(status=status_filter)
    
    search_query = request.GET.get('search')
    if search_query:
        configurations = configurations.filter(
            Q(name__icontains=search_query) |
            Q(dataset__name__icontains=search_query) |
            Q(base_model__name__icontains=search_query)
        )
    
    # Pagination
    paginator = Paginator(configurations, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_choices': TrainingConfiguration.STATUS_CHOICES,
        'current_status': status_filter,
        'search_query': search_query,
    }
    return render(request, 'training_config/configuration_list.html', context)


@login_required
def configuration_detail(request, config_id):
    """Detailed view of a training configuration"""
    config = get_object_or_404(TrainingConfiguration, id=config_id, user=request.user)
    
    # Get related experiments
    experiments = TrainingExperiment.objects.filter(configuration=config)
    
    context = {
        'config': config,
        'experiments': experiments,
    }
    return render(request, 'training_config/configuration_detail.html', context)


@login_required
def estimate_configuration(request):
    """AJAX endpoint to estimate training metrics"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Create temporary configuration for estimation
            temp_config = TrainingConfiguration(
                dataset_id=data.get('dataset_id'),
                base_model_id=data.get('base_model_id'),
                adapter_id=data.get('adapter_id'),
                hardware_id=data.get('hardware_id'),
                learning_rate=float(data.get('learning_rate', 5e-5)),
                batch_size=int(data.get('batch_size', 32)),
                epochs=int(data.get('epochs', 3)),
                gradient_accumulation_steps=int(data.get('gradient_accumulation_steps', 1)),
                precision=data.get('precision', 'FP16'),
            )
            
            # Get related objects
            temp_config.dataset = DatasetProfile.objects.get(id=data.get('dataset_id'))
            temp_config.base_model = BaseModelProfile.objects.get(id=data.get('base_model_id'))
            temp_config.adapter = AdapterConfiguration.objects.get(id=data.get('adapter_id'))
            temp_config.hardware = HardwareProfile.objects.get(id=data.get('hardware_id'))
            
            # Calculate estimates
            metrics = temp_config.calculate_estimated_metrics()
            
            return JsonResponse({
                'success': True,
                'metrics': {
                    'time_hours': round(metrics['time_hours'], 2),
                    'cost_usd': round(metrics['cost_usd'], 2),
                    'gpu_hours': round(metrics['gpu_hours'], 2),
                    'total_flops': metrics['total_flops'],
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def configuration_comparison(request):
    """Compare multiple configurations"""
    if request.method == 'POST':
        config_ids = request.POST.getlist('configurations')
        if len(config_ids) < 2:
            messages.error(request, 'Please select at least 2 configurations to compare.')
            return redirect('training_config:configuration_list')
        
        # Create comparison
        comparison = ConfigurationComparison.objects.create(
            name=request.POST.get('name', f'Comparison {len(config_ids)} configs'),
            user=request.user
        )
        
        configs = TrainingConfiguration.objects.filter(id__in=config_ids, user=request.user)
        comparison.configurations.set(configs)
        
        return redirect('training_config:comparison_detail', comparison_id=comparison.id)
    
    # Show comparison setup
    configurations = TrainingConfiguration.objects.filter(user=request.user, status='ESTIMATED')
    return render(request, 'training_config/comparison_setup.html', {'configurations': configurations})


@login_required
def comparison_detail(request, comparison_id):
    """Detailed comparison view"""
    comparison = get_object_or_404(ConfigurationComparison, id=comparison_id, user=request.user)
    configurations = comparison.configurations.all()
    
    # Generate comparison data
    comparison_data = []
    for config in configurations:
        comparison_data.append({
            'name': config.name,
            'dataset': config.dataset.name,
            'model': config.base_model.name,
            'adapter': config.adapter.name,
            'estimated_time': config.estimated_time_hours,
            'estimated_cost': float(config.estimated_cost_usd) if config.estimated_cost_usd else 0,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
        })
    
    context = {
        'comparison': comparison,
        'configurations': configurations,
        'comparison_data': comparison_data,
    }
    return render(request, 'training_config/comparison_detail.html', context)


@login_required
def start_training(request, config_id):
    """Start training for a configuration"""
    config = get_object_or_404(TrainingConfiguration, id=config_id, user=request.user)
    
    if config.status == 'RUNNING':
        messages.warning(request, 'Training is already running for this configuration.')
        return redirect('training_config:configuration_detail', config_id=config.id)
    
    try:
        # Create training experiment
        experiment = TrainingExperiment.objects.create(configuration=config)
        
        # Update configuration status
        config.status = 'RUNNING'
        config.save()
        
        # TODO: Implement actual training logic here
        # This would integrate with your training infrastructure
        
        messages.success(request, f'Training started for configuration "{config.name}"!')
        
    except Exception as e:
        messages.error(request, f'Error starting training: {str(e)}')
    
    return redirect('training_config:configuration_detail', config_id=config.id)


@login_required
def experiment_detail(request, experiment_id):
    """Detailed view of a training experiment"""
    experiment = get_object_or_404(TrainingExperiment, id=experiment_id)
    
    # Generate loss curve visualization if data exists
    loss_chart = None
    if experiment.training_logs and 'losses' in experiment.training_logs:
        loss_chart = generate_loss_chart(experiment.training_logs)
    
    context = {
        'experiment': experiment,
        'loss_chart': loss_chart,
    }
    return render(request, 'training_config/experiment_detail.html', context)


def generate_loss_chart(training_logs):
    """Generate base64 encoded loss chart"""
    plt.figure(figsize=(10, 6))
    
    if 'losses' in training_logs:
        epochs = list(range(len(training_logs['losses'])))
        plt.plot(epochs, training_logs['losses'], label='Training Loss', color='blue')
    
    if 'val_losses' in training_logs:
        epochs = list(range(len(training_logs['val_losses'])))
        plt.plot(epochs, training_logs['val_losses'], label='Validation Loss', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64


@login_required
def hardware_profiles(request):
    """Manage hardware profiles"""
    profiles = HardwareProfile.objects.filter(is_active=True)
    return render(request, 'training_config/hardware_profiles.html', {'profiles': profiles})


@login_required
def model_profiles(request):
    """Manage model profiles"""
    profiles = BaseModelProfile.objects.filter(is_active=True)
    return render(request, 'training_config/model_profiles.html', {'profiles': profiles})


@login_required
def dataset_profiles(request):
    """Manage dataset profiles"""
    profiles = DatasetProfile.objects.filter(is_active=True)
    return render(request, 'training_config/dataset_profiles.html', {'profiles': profiles})


# ================================
# CLOUD TRAINING VIEWS
# ================================

@login_required
def cloud_training_dashboard(request):
    """Cloud training dashboard with live monitoring"""
    # Get user's cloud jobs
    cloud_jobs = CloudTrainingJob.objects.filter(
        configuration__user=request.user
    ).select_related('configuration', 'cloud_provider')[:10]
    
    # Get active jobs
    active_jobs = cloud_jobs.filter(status__in=['PENDING', 'STARTING', 'RUNNING'])
    
    # Get available cloud providers
    cloud_providers = CloudProvider.objects.filter(is_active=True)
    
    context = {
        'cloud_jobs': cloud_jobs,
        'active_jobs': active_jobs,
        'cloud_providers': cloud_providers,
    }
    return render(request, 'training_config/cloud_dashboard.html', context)


@login_required
def start_cloud_training(request, config_id):
    """Start training on a cloud platform"""
    config = get_object_or_404(TrainingConfiguration, id=config_id, user=request.user)
    
    if request.method == 'POST':
        cloud_provider_id = request.POST.get('cloud_provider')
        cloud_provider = get_object_or_404(CloudProvider, id=cloud_provider_id)
        
        try:
            # Create cloud training job
            cloud_job = CloudTrainingJob.objects.create(
                configuration=config,
                cloud_provider=cloud_provider,
                total_epochs=config.epochs,
                status='PENDING'
            )
            
            # TODO: Implement actual cloud job submission
            # This would integrate with cloud provider APIs
            cloud_job.external_job_id = f"job_{cloud_job.id.hex[:8]}"
            cloud_job.status = 'STARTING'
            cloud_job.save()
            
            messages.success(request, f'Cloud training job started on {cloud_provider.name}!')
            return redirect('training_config:cloud_job_detail', job_id=cloud_job.id)
            
        except Exception as e:
            messages.error(request, f'Error starting cloud training: {str(e)}')
    
    cloud_providers = CloudProvider.objects.filter(is_active=True)
    context = {
        'config': config,
        'cloud_providers': cloud_providers,
    }
    return render(request, 'training_config/start_cloud_training.html', context)


@login_required
def cloud_job_detail(request, job_id):
    """Detailed view of cloud training job with live metrics"""
    job = get_object_or_404(CloudTrainingJob, id=job_id, configuration__user=request.user)
    
    # Get recent metrics for visualization
    recent_metrics = LiveTrainingMetrics.objects.filter(cloud_job=job).order_by('epoch', 'step')[:1000]
    
    context = {
        'job': job,
        'recent_metrics': recent_metrics,
    }
    return render(request, 'training_config/cloud_job_detail.html', context)


@login_required
def live_metrics_stream(request, job_id):
    """Server-sent events stream for live training metrics"""
    job = get_object_or_404(CloudTrainingJob, id=job_id, configuration__user=request.user)
    
    def event_stream():
        while True:
            # Get latest metrics
            latest_metrics = LiveTrainingMetrics.objects.filter(
                cloud_job=job
            ).order_by('-timestamp').first()
            
            if latest_metrics:
                data = {
                    'epoch': latest_metrics.epoch,
                    'step': latest_metrics.step,
                    'train_loss': latest_metrics.train_loss,
                    'validation_loss': latest_metrics.validation_loss,
                    'learning_rate': latest_metrics.learning_rate,
                    'gpu_utilization': latest_metrics.gpu_utilization,
                    'memory_usage_gb': latest_metrics.memory_usage_gb,
                    'timestamp': latest_metrics.timestamp.isoformat(),
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            time.sleep(2)  # Update every 2 seconds
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Connection'] = 'keep-alive'
    return response


@csrf_exempt
def webhook_training_update(request, job_id):
    """Webhook endpoint for cloud providers to send training updates"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            job = CloudTrainingJob.objects.get(id=job_id)
            
            # Update job status
            if 'status' in data:
                job.status = data['status']
                job.save()
            
            # Add metrics if provided
            if 'metrics' in data:
                metrics_data = data['metrics']
                LiveTrainingMetrics.objects.create(
                    cloud_job=job,
                    epoch=metrics_data.get('epoch', 0),
                    step=metrics_data.get('step', 0),
                    train_loss=metrics_data.get('train_loss', 0.0),
                    validation_loss=metrics_data.get('validation_loss'),
                    learning_rate=metrics_data.get('learning_rate'),
                    gpu_utilization=metrics_data.get('gpu_utilization'),
                    memory_usage_gb=metrics_data.get('memory_usage_gb'),
                    custom_metrics=metrics_data.get('custom_metrics', {})
                )
            
            return JsonResponse({'status': 'success'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
def generate_colab_notebook(request, config_id):
    """Generate Google Colab notebook for training configuration"""
    config = get_object_or_404(TrainingConfiguration, id=config_id, user=request.user)
    
    # Generate notebook content
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Training Configuration: {config.name}\n",
                    f"**Dataset:** {config.dataset.name}\n",
                    f"**Model:** {config.base_model.name}\n",
                    f"**Adapter:** {config.adapter.name}\n",
                    f"**Hardware:** {config.hardware.name}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
                    "!pip install transformers datasets accelerate peft\n",
                    "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
                    "!pip install wandb  # For experiment tracking\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import json\n",
                    "import requests\n",
                    "from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer\n",
                    "from datasets import load_dataset\n",
                    "from peft import LoraConfig, get_peft_model, TaskType\n",
                    "import wandb\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Configuration\n",
                    f"CONFIG = {{\n",
                    f"    'dataset_name': '{config.dataset.hf_dataset_name}',\n",
                    f"    'model_name': '{config.base_model.hf_model_name}',\n",
                    f"    'learning_rate': {config.learning_rate},\n",
                    f"    'batch_size': {config.batch_size},\n",
                    f"    'epochs': {config.epochs},\n",
                    f"    'lora_rank': {config.adapter.lora_rank if config.adapter.lora_rank else 'None'},\n",
                    f"    'lora_alpha': {config.adapter.lora_alpha if config.adapter.lora_alpha else 'None'},\n",
                    f"    'webhook_url': 'https://your-django-app.com/training/webhook/{config.id}/',\n",
                    f"}}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Webhook function to send updates back to Django\n",
                    "def send_training_update(status, metrics=None):\n",
                    "    data = {'status': status}\n",
                    "    if metrics:\n",
                    "        data['metrics'] = metrics\n",
                    "    \n",
                    "    try:\n",
                    "        response = requests.post(CONFIG['webhook_url'], json=data)\n",
                    "        print(f'Webhook sent: {response.status_code}')\n",
                    "    except Exception as e:\n",
                    "        print(f'Webhook error: {e}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load dataset and model\n",
                    "send_training_update('STARTING')\n",
                    "\n",
                    "dataset = load_dataset(CONFIG['dataset_name'])\n",
                    "tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])\n",
                    "model = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'])\n",
                    "\n",
                    "if tokenizer.pad_token is None:\n",
                    "    tokenizer.pad_token = tokenizer.eos_token\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup LoRA if specified\n",
                    f"if CONFIG['lora_rank'] is not None:\n",
                    f"    lora_config = LoraConfig(\n",
                    f"        task_type=TaskType.CAUSAL_LM,\n",
                    f"        r=CONFIG['lora_rank'],\n",
                    f"        lora_alpha=CONFIG['lora_alpha'],\n",
                    f"        lora_dropout=0.1,\n",
                    f"        target_modules=['q_proj', 'v_proj']\n",
                    f"    )\n",
                    f"    model = get_peft_model(model, lora_config)\n",
                    f"    print('LoRA configuration applied')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Custom Trainer class with webhook updates\n",
                    "class WebhookTrainer(Trainer):\n",
                    "    def log(self, logs):\n",
                    "        super().log(logs)\n",
                    "        \n",
                    "        # Send metrics to webhook\n",
                    "        if 'train_loss' in logs:\n",
                    "            metrics = {\n",
                    "                'epoch': logs.get('epoch', 0),\n",
                    "                'step': logs.get('step', 0),\n",
                    "                'train_loss': logs.get('train_loss', 0),\n",
                    "                'learning_rate': logs.get('learning_rate', 0),\n",
                    "            }\n",
                    "            send_training_update('RUNNING', metrics)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Training setup and execution\n",
                    "training_args = TrainingArguments(\n",
                    f"    output_dir='./results',\n",
                    f"    num_train_epochs={config.epochs},\n",
                    f"    per_device_train_batch_size={config.batch_size},\n",
                    f"    learning_rate={config.learning_rate},\n",
                    f"    logging_steps=10,\n",
                    f"    save_steps=500,\n",
                    f"    evaluation_strategy='epoch',\n",
                    f")\n",
                    "\n",
                    "trainer = WebhookTrainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=dataset['train'],\n",
                    "    tokenizer=tokenizer,\n",
                    ")\n",
                    "\n",
                    "# Start training\n",
                    "trainer.train()\n",
                    "send_training_update('COMPLETED')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    notebook = TrainingNotebook.objects.create(
        configuration=config,
        notebook_type='COLAB',
        notebook_content=notebook_content,
        requirements_txt="transformers\ndatasets\naccelerate\npeft\ntorch\nwandb\nrequests"
    )
    
    # Return as downloadable file
    response = JsonResponse(notebook_content)
    response['Content-Disposition'] = f'attachment; filename="{config.name}_training.ipynb"'
    return response


@login_required
def live_training_monitor(request):
    """Live training monitoring dashboard"""
    # Get all active cloud jobs for the user
    active_jobs = CloudTrainingJob.objects.filter(
        configuration__user=request.user,
        status__in=['RUNNING', 'STARTING']
    ).select_related('configuration', 'cloud_provider')
    
    context = {
        'active_jobs': active_jobs,
    }
    return render(request, 'training_config/live_monitor.html', context)


@login_required
def training_metrics_api(request, job_id):
    """API endpoint for training metrics (for AJAX calls)"""
    job = get_object_or_404(CloudTrainingJob, id=job_id, configuration__user=request.user)
    
    # Get metrics based on time range
    limit = int(request.GET.get('limit', 100))
    metrics = LiveTrainingMetrics.objects.filter(
        cloud_job=job
    ).order_by('-timestamp')[:limit]
    
    data = {
        'job_status': job.status,
        'progress': job.progress_percentage,
        'current_epoch': job.current_epoch,
        'total_epochs': job.total_epochs,
        'metrics': [
            {
                'timestamp': m.timestamp.isoformat(),
                'epoch': m.epoch,
                'step': m.step,
                'train_loss': m.train_loss,
                'validation_loss': m.validation_loss,
                'learning_rate': m.learning_rate,
                'gpu_utilization': m.gpu_utilization,
                'memory_usage_gb': m.memory_usage_gb,
            }
            for m in reversed(metrics)
        ]
    }
    
    return JsonResponse(data)


@login_required
def dataset_detail(request, dataset_id):
    """Display detailed information about a specific dataset"""
    dataset = get_object_or_404(DatasetProfile, id=dataset_id)
    return render(request, 'training_config/dataset_detail.html', {'dataset': dataset})
