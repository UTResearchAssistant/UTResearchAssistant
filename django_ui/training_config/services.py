"""
Cloud training services for managing ML training jobs across different providers.
Includes Google Colab, Kaggle, AWS SageMaker, Azure ML, and other cloud platforms.
"""

import json
import requests
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
from .models import CloudTrainingJob, CloudProvider, TrainingConfiguration, LiveTrainingMetrics


class CloudTrainingService:
    """Service for managing cloud-based ML training jobs."""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize cloud provider handlers."""
        self.providers = {
            'COLAB': ColabProvider(),
            'KAGGLE': KaggleProvider(),
            'AWS_SAGEMAKER': AWSProvider(),
            'AZURE_ML': AzureProvider(),
            'GCP_VERTEX': GCPProvider(),
            'PAPERSPACE': PaperspaceProvider(),
        }
    
    def start_training_job(self, configuration: TrainingConfiguration, 
                          cloud_provider: CloudProvider) -> CloudTrainingJob:
        """Start a new training job on the specified cloud provider."""
        
        # Create job record
        job = CloudTrainingJob.objects.create(
            configuration=configuration,
            cloud_provider=cloud_provider,
            external_job_id=str(uuid.uuid4()),
            status='PENDING',
            started_at=timezone.now()
        )
        
        try:
            # Get provider handler
            provider_handler = self.providers.get(cloud_provider.provider_type)
            if not provider_handler:
                raise ValueError(f"Unsupported provider type: {cloud_provider.provider_type}")
            
            # Submit job to cloud provider
            external_job_id = provider_handler.submit_job(configuration, cloud_provider, job)
            
            # Update job with external ID
            job.external_job_id = external_job_id
            job.status = 'RUNNING'
            job.save()
            
            return job
            
        except Exception as e:
            job.status = 'FAILED'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
            raise
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get current status of a training job."""
        provider_handler = self.providers.get(job.cloud_provider.provider_type)
        if not provider_handler:
            return {'status': 'unknown', 'error': 'Unsupported provider'}
        
        return provider_handler.get_job_status(job)
    
    def stop_training_job(self, job: CloudTrainingJob) -> bool:
        """Stop a running training job."""
        provider_handler = self.providers.get(job.cloud_provider.provider_type)
        if not provider_handler:
            return False
        
        success = provider_handler.stop_job(job)
        if success:
            job.status = 'CANCELLED'
            job.completed_at = timezone.now()
            job.save()
        
        return success
    
    def _calculate_estimated_cost(self, configuration: TrainingConfiguration, 
                                 cloud_provider: CloudProvider) -> float:
        """Calculate estimated cost for training job."""
        # Basic cost calculation based on time estimate and provider type
        estimated_hours = self._estimate_training_time(configuration) / 3600
        
        # Cost estimates per hour by provider type
        cost_rates = {
            'COLAB': 9.99,
            'KAGGLE': 0.00,
            'PAPERSPACE': 0.45,
            'AWS_SAGEMAKER': 3.06,
            'AZURE_ML': 2.88,
            'GCP_VERTEX': 2.48,
        }
        
        hourly_rate = cost_rates.get(cloud_provider.provider_type, 1.00)
        return estimated_hours * hourly_rate
    
    def _estimate_training_time(self, configuration: TrainingConfiguration) -> float:
        """Estimate training time in seconds."""
        # Simplified estimation - can be enhanced with more sophisticated models
        base_time = 3600  # 1 hour base
        epoch_multiplier = configuration.max_epochs
        batch_multiplier = max(1, 32 / configuration.batch_size)
        
        return base_time * epoch_multiplier * batch_multiplier


class BaseCloudProvider:
    """Base class for cloud provider implementations."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit a training job to the cloud provider."""
        raise NotImplementedError
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get job status from the cloud provider."""
        raise NotImplementedError
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop a running job."""
        raise NotImplementedError


class ColabProvider(BaseCloudProvider):
    """Google Colab provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Generate and submit Colab notebook."""
        # For Colab, we generate a notebook and provide instructions
        # Since Colab doesn't have a direct API for job submission,
        # we'll simulate with a placeholder implementation
        
        notebook_content = NotebookGenerator.generate_colab_notebook(configuration)
        
        # In a real implementation, you might:
        # 1. Upload notebook to Google Drive
        # 2. Use Colab Pro API (when available)
        # 3. Provide manual execution instructions
        
        # For now, return a simulated job ID
        return f"colab_{uuid.uuid4().hex[:8]}"
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Simulate job status for Colab."""
        # In a real implementation, this would check actual job status
        return {
            'status': 'running',
            'progress': 45.0,
            'current_epoch': 2,
            'loss': 0.234,
            'accuracy': 0.876
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop Colab job (simulated)."""
        return True


class KaggleProvider(BaseCloudProvider):
    """Kaggle Kernels provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit job to Kaggle Kernels."""
        # Use Kaggle API to create and run a kernel
        # This is a simplified implementation
        
        try:
            # In a real implementation, use kaggle.api
            kernel_metadata = {
                'title': f"Training {configuration.name}",
                'code_file': 'train.py',
                'language': 'python',
                'kernel_type': 'script',
                'is_private': True,
                'enable_gpu': True,
                'enable_internet': True
            }
            
            # Simulated kernel ID
            return f"kaggle_{uuid.uuid4().hex[:8]}"
            
        except Exception as e:
            raise Exception(f"Failed to submit Kaggle job: {str(e)}")
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get Kaggle kernel status."""
        return {
            'status': 'running',
            'progress': 30.0,
            'logs': 'Training in progress...'
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop Kaggle kernel."""
        return True


class AWSProvider(BaseCloudProvider):
    """AWS SageMaker provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit job to AWS SageMaker."""
        # Use boto3 to create SageMaker training job
        # This is a placeholder implementation
        
        job_name = f"training-{uuid.uuid4().hex[:8]}"
        
        # In a real implementation:
        # import boto3
        # sagemaker = boto3.client('sagemaker')
        # response = sagemaker.create_training_job(...)
        
        return job_name
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get SageMaker job status."""
        return {
            'status': 'InProgress',
            'progress': 60.0
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop SageMaker job."""
        return True


class AzureProvider(BaseCloudProvider):
    """Azure ML provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit job to Azure ML."""
        # Use Azure ML SDK
        # This is a placeholder implementation
        
        return f"azure_{uuid.uuid4().hex[:8]}"
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get Azure ML job status."""
        return {
            'status': 'Running',
            'progress': 75.0
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop Azure ML job."""
        return True


class GCPProvider(BaseCloudProvider):
    """Google Cloud Vertex AI provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit job to Vertex AI."""
        return f"gcp_{uuid.uuid4().hex[:8]}"
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get Vertex AI job status."""
        return {
            'status': 'RUNNING',
            'progress': 50.0
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop Vertex AI job."""
        return True


class PaperspaceProvider(BaseCloudProvider):
    """Paperspace Gradient provider implementation."""
    
    def submit_job(self, configuration: TrainingConfiguration, 
                   cloud_provider: CloudProvider, job: CloudTrainingJob) -> str:
        """Submit job to Paperspace Gradient."""
        return f"paperspace_{uuid.uuid4().hex[:8]}"
    
    def get_job_status(self, job: CloudTrainingJob) -> Dict[str, Any]:
        """Get Paperspace job status."""
        return {
            'status': 'running',
            'progress': 40.0
        }
    
    def stop_job(self, job: CloudTrainingJob) -> bool:
        """Stop Paperspace job."""
        return True


class NotebookGenerator:
    """Service for generating training notebooks for different platforms."""
    
    @staticmethod
    def generate_colab_notebook(configuration: TrainingConfiguration) -> str:
        """Generate a Google Colab notebook for the training configuration."""
        
        notebook_content = f"""{{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {{
    "colab": {{
      "provenance": [],
      "gpuType": "T4"
    }},
    "kernelspec": {{
      "name": "python3",
      "display_name": "Python 3"
    }},
    "language_info": {{
      "name": "python"
    }},
    "accelerator": "GPU"
  }},
  "cells": [
    {{
      "cell_type": "markdown",
      "metadata": {{}},
      "source": [
        "# Training Configuration: {configuration.name}\\n",
        "\\n",
        "**Model:** {configuration.base_model.name}\\n",
        "**Dataset:** {configuration.dataset.name}\\n",
        "**Training Type:** {configuration.get_training_type_display()}\\n",
        "**Learning Rate:** {configuration.learning_rate}\\n",
        "**Batch Size:** {configuration.batch_size}\\n",
        "**Max Epochs:** {configuration.max_epochs}\\n"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Install required packages\\n",
        "!pip install torch transformers datasets accelerate peft wandb\\n",
        "\\n",
        "# Import libraries\\n",
        "import torch\\n",
        "import transformers\\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer\\n",
        "from datasets import load_dataset\\n",
        "from peft import LoraConfig, get_peft_model\\n",
        "import wandb\\n",
        "import json\\n",
        "from datetime import datetime"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Configuration\\n",
        "CONFIG = {{\\n",
        "    'model_name': '{configuration.base_model.name}',\\n",
        "    'dataset_name': '{configuration.dataset.name}',\\n",
        "    'learning_rate': {configuration.learning_rate},\\n",
        "    'batch_size': {configuration.batch_size},\\n",
        "    'max_epochs': {configuration.max_epochs},\\n",
        "    'max_length': 512,\\n",
        "    'lora_rank': {configuration.adapter_config.lora_rank if configuration.adapter_config else 8},\\n",
        "    'lora_alpha': {configuration.adapter_config.lora_alpha if configuration.adapter_config else 32},\\n",
        "    'output_dir': './results'\\n",
        "}}\\n",
        "\\n",
        "print(f'Training configuration: {{json.dumps(CONFIG, indent=2)}}')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Load dataset\\n",
        "dataset = load_dataset(CONFIG['dataset_name'])\\n",
        "print(f'Dataset loaded: {{dataset}}')\\n",
        "\\n",
        "# Load tokenizer and model\\n",
        "tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])\\n",
        "if tokenizer.pad_token is None:\\n",
        "    tokenizer.pad_token = tokenizer.eos_token\\n",
        "\\n",
        "model = AutoModelForCausalLM.from_pretrained(\\n",
        "    CONFIG['model_name'],\\n",
        "    torch_dtype=torch.float16,\\n",
        "    device_map='auto'\\n",
        ")\\n",
        "\\n",
        "print(f'Model loaded: {{model}}')\\n",
        "print(f'Model parameters: {{model.num_parameters():,}}')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Setup LoRA\\n",
        "lora_config = LoraConfig(\\n",
        "    r=CONFIG['lora_rank'],\\n",
        "    lora_alpha=CONFIG['lora_alpha'],\\n",
        "    target_modules=['q_proj', 'v_proj'],\\n",
        "    lora_dropout=0.1,\\n",
        "    bias='none',\\n",
        "    task_type='CAUSAL_LM'\\n",
        ")\\n",
        "\\n",
        "model = get_peft_model(model, lora_config)\\n",
        "model.print_trainable_parameters()"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Data preprocessing\\n",
        "def tokenize_function(examples):\\n",
        "    return tokenizer(\\n",
        "        examples['text'] if 'text' in examples else examples['input'],\\n",
        "        truncation=True,\\n",
        "        padding='max_length',\\n",
        "        max_length=CONFIG['max_length']\\n",
        "    )\\n",
        "\\n",
        "tokenized_dataset = dataset.map(tokenize_function, batched=True)\\n",
        "print(f'Tokenized dataset: {{tokenized_dataset}}')\\n",
        "\\n",
        "# Split for training\\n",
        "train_dataset = tokenized_dataset['train']\\n",
        "eval_dataset = tokenized_dataset.get('validation', tokenized_dataset.get('test'))\\n",
        "\\n",
        "print(f'Train samples: {{len(train_dataset)}}')\\n",
        "if eval_dataset:\\n",
        "    print(f'Eval samples: {{len(eval_dataset)}}')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Initialize Weights & Biases\\n",
        "wandb.init(\\n",
        "    project='research-training',\\n",
        "    config=CONFIG,\\n",
        "    name=f'colab-training-{{datetime.now().strftime(\\\"%Y%m%d-%H%M%S\\\")}}'\\n",
        ")\\n",
        "\\n",
        "# Training arguments\\n",
        "training_args = TrainingArguments(\\n",
        "    output_dir=CONFIG['output_dir'],\\n",
        "    num_train_epochs=CONFIG['max_epochs'],\\n",
        "    per_device_train_batch_size=CONFIG['batch_size'],\\n",
        "    per_device_eval_batch_size=CONFIG['batch_size'],\\n",
        "    learning_rate=CONFIG['learning_rate'],\\n",
        "    warmup_steps=100,\\n",
        "    logging_steps=10,\\n",
        "    evaluation_strategy='steps' if eval_dataset else 'no',\\n",
        "    eval_steps=50 if eval_dataset else None,\\n",
        "    save_strategy='steps',\\n",
        "    save_steps=100,\\n",
        "    load_best_model_at_end=True if eval_dataset else False,\\n",
        "    metric_for_best_model='eval_loss' if eval_dataset else None,\\n",
        "    report_to='wandb',\\n",
        "    dataloader_pin_memory=False,\\n",
        "    gradient_checkpointing=True,\\n",
        "    fp16=True\\n",
        ")"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Initialize trainer\\n",
        "trainer = Trainer(\\n",
        "    model=model,\\n",
        "    args=training_args,\\n",
        "    train_dataset=train_dataset,\\n",
        "    eval_dataset=eval_dataset,\\n",
        "    tokenizer=tokenizer\\n",
        ")\\n",
        "\\n",
        "print('Trainer initialized. Starting training...')\\n",
        "\\n",
        "# Start training\\n",
        "trainer.train()"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Save the model\\n",
        "trainer.save_model('./final_model')\\n",
        "tokenizer.save_pretrained('./final_model')\\n",
        "\\n",
        "print('Training completed! Model saved to ./final_model')\\n",
        "\\n",
        "# Log final metrics\\n",
        "if eval_dataset:\\n",
        "    eval_results = trainer.evaluate()\\n",
        "    print(f'Final evaluation results: {{eval_results}}')\\n",
        "    wandb.log(eval_results)\\n",
        "\\n",
        "# Finish W&B run\\n",
        "wandb.finish()"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{}},
      "source": [
        "# Test the trained model\\n",
        "from transformers import pipeline\\n",
        "\\n",
        "# Load the trained model\\n",
        "generator = pipeline(\\n",
        "    'text-generation',\\n",
        "    model='./final_model',\\n",
        "    tokenizer='./final_model',\\n",
        "    device=0 if torch.cuda.is_available() else -1\\n",
        ")\\n",
        "\\n",
        "# Test generation\\n",
        "test_prompt = 'Hello, this is a test of the trained model'\\n",
        "result = generator(test_prompt, max_length=100, num_return_sequences=1)\\n",
        "print(f'Generated text: {{result[0][\\\"generated_text\\\"]}}')"
      ]
    }}
  ]
}}"""
        
        return notebook_content
    
    @staticmethod
    def generate_kaggle_script(configuration: TrainingConfiguration) -> str:
        """Generate a Kaggle kernel script."""
        
        script_content = f"""#!/usr/bin/env python3
# Training Configuration: {configuration.name}
# Generated for Kaggle Kernels

import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb
import json
from datetime import datetime

# Configuration
CONFIG = {{
    'model_name': '{configuration.base_model.name}',
    'dataset_name': '{configuration.dataset.name}',
    'learning_rate': {configuration.learning_rate},
    'batch_size': {configuration.batch_size},
    'max_epochs': {configuration.max_epochs},
    'max_length': 512,
    'lora_rank': {configuration.adapter_config.lora_rank if configuration.adapter_config else 8},
    'lora_alpha': {configuration.adapter_config.lora_alpha if configuration.adapter_config else 32},
    'output_dir': '/kaggle/working/results'
}}

def main():
    print(f'Training configuration: {{json.dumps(CONFIG, indent=2)}}')
    
    # Set up CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {{device}}')
    
    # Load dataset
    dataset = load_dataset(CONFIG['dataset_name'])
    print(f'Dataset loaded: {{dataset}}')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    print(f'Model loaded: {{model}}')
    print(f'Model parameters: {{model.num_parameters():,}}')
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=CONFIG['lora_rank'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data preprocessing
    def tokenize_function(examples):
        return tokenizer(
            examples['text'] if 'text' in examples else examples['input'],
            truncation=True,
            padding='max_length',
            max_length=CONFIG['max_length']
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(f'Tokenized dataset: {{tokenized_dataset}}')
    
    # Split for training
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset.get('validation', tokenized_dataset.get('test'))
    
    print(f'Train samples: {{len(train_dataset)}}')
    if eval_dataset:
        print(f'Eval samples: {{len(eval_dataset)}}')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['max_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy='steps' if eval_dataset else 'no',
        eval_steps=50 if eval_dataset else None,
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model='eval_loss' if eval_dataset else None,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    print('Trainer initialized. Starting training...')
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model('/kaggle/working/final_model')
    tokenizer.save_pretrained('/kaggle/working/final_model')
    
    print('Training completed! Model saved to /kaggle/working/final_model')
    
    # Log final metrics
    if eval_dataset:
        eval_results = trainer.evaluate()
        print(f'Final evaluation results: {{eval_results}}')

if __name__ == '__main__':
    main()
"""
        
        return script_content


class MetricsCollector:
    """Service for collecting and processing training metrics."""
    
    @staticmethod
    def process_webhook_data(job: CloudTrainingJob, webhook_data: Dict[str, Any]):
        """Process incoming webhook data from cloud providers."""
        
        # Extract metrics from webhook data
        metrics_data = {
            'epoch': webhook_data.get('epoch', 0),
            'step': webhook_data.get('step', 0),
            'train_loss': webhook_data.get('loss', webhook_data.get('train_loss')),
            'validation_loss': webhook_data.get('validation_loss', webhook_data.get('val_loss')),
            'learning_rate': webhook_data.get('learning_rate'),
            'train_accuracy': webhook_data.get('accuracy', webhook_data.get('train_accuracy')),
            'cloud_job': job
        }
        
        # Create metrics record if we have required fields
        if metrics_data['train_loss'] is not None:
            LiveTrainingMetrics.objects.create(
                **{k: v for k, v in metrics_data.items() if v is not None}
            )
        
        # Update job status
        if 'status' in webhook_data:
            # Map status values to our model choices
            status_mapping = {
                'running': 'RUNNING',
                'completed': 'COMPLETED',
                'failed': 'FAILED',
                'cancelled': 'CANCELLED'
            }
            job.status = status_mapping.get(webhook_data['status'], webhook_data['status'])
        
        if 'progress_percentage' in webhook_data:
            job.progress_percentage = webhook_data['progress_percentage']
        
        if 'current_epoch' in webhook_data:
            job.current_epoch = webhook_data['current_epoch']
        
        job.save()
    
    @staticmethod
    def get_live_metrics(job: CloudTrainingJob, limit: int = 100) -> List[LiveTrainingMetrics]:
        """Get recent live metrics for a job."""
        return LiveTrainingMetrics.objects.filter(
            cloud_job=job
        ).order_by('-timestamp')[:limit]
