from django.core.management.base import BaseCommand
from training_config.models import CloudProvider, HardwareProfile, BaseModelProfile, DatasetProfile
import json


class Command(BaseCommand):
    help = 'Populate sample cloud providers and training data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Delete existing cloud providers before creating new ones',
        )

    def handle(self, *args, **options):
        if options['reset']:
            self.stdout.write('Deleting existing cloud providers...')
            CloudProvider.objects.all().delete()

        # Create cloud providers
        cloud_providers_data = [
            {
                'name': 'Google Colab Pro',
                'provider_type': 'COLAB',
                'api_endpoint': 'https://colab.research.google.com',
                'api_key': '',
                'webhook_url': '',
                'region': 'global',
                'instance_type': 'Tesla T4',
                'max_runtime_hours': 12,
                'is_active': True
            },
            {
                'name': 'Kaggle Kernels',
                'provider_type': 'KAGGLE',
                'api_endpoint': 'https://www.kaggle.com/api/v1',
                'api_key': '',
                'webhook_url': '',
                'region': 'global',
                'instance_type': 'Tesla P100',
                'max_runtime_hours': 12,
                'is_active': True
            },
            {
                'name': 'Paperspace Gradient',
                'provider_type': 'PAPERSPACE',
                'api_endpoint': 'https://api.paperspace.io',
                'api_key': '',
                'webhook_url': '',
                'region': 'US-East',
                'instance_type': 'P5000',
                'max_runtime_hours': 24,
                'is_active': True
            },
            {
                'name': 'AWS SageMaker',
                'provider_type': 'AWS_SAGEMAKER',
                'api_endpoint': 'https://sagemaker.us-east-1.amazonaws.com',
                'api_key': '',
                'webhook_url': '',
                'region': 'us-east-1',
                'instance_type': 'ml.g4dn.xlarge',
                'max_runtime_hours': 72,
                'is_active': True
            },
            {
                'name': 'Azure Machine Learning',
                'provider_type': 'AZURE_ML',
                'api_endpoint': 'https://ml.azure.com',
                'api_key': '',
                'webhook_url': '',
                'region': 'eastus',
                'instance_type': 'Standard_NC6s_v3',
                'max_runtime_hours': 48,
                'is_active': True
            },
            {
                'name': 'Google Cloud Vertex AI',
                'provider_type': 'GCP_VERTEX',
                'api_endpoint': 'https://us-central1-aiplatform.googleapis.com',
                'api_key': '',
                'webhook_url': '',
                'region': 'us-central1',
                'instance_type': 'n1-highmem-8',
                'max_runtime_hours': 48,
                'is_active': True
            }
        ]

        for provider_data in cloud_providers_data:
            provider, created = CloudProvider.objects.get_or_create(
                name=provider_data['name'],
                defaults=provider_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created cloud provider: {provider.name}')
                )
            else:
                self.stdout.write(f'Cloud provider already exists: {provider.name}')

        # Create sample hardware profiles if they don't exist
        hardware_profiles_data = [
            {
                'name': 'Tesla T4 (Colab)',
                'device_type': 'RTX4090',  # Closest available choice
                'memory_gb': 16.0,
                'compute_capability': '7.5',
                'fp16_tflops': 65.0,
                'fp32_tflops': 8.1,
                'bf16_tflops': 65.0,
                'cost_per_hour': 9.99,
                'utilization_factor': 0.75
            },
            {
                'name': 'Tesla P100 (Kaggle)',
                'device_type': 'V100',
                'memory_gb': 16.0,
                'compute_capability': '6.0',
                'fp16_tflops': 28.0,
                'fp32_tflops': 9.3,
                'bf16_tflops': None,
                'cost_per_hour': 0.00,
                'utilization_factor': 0.80
            },
            {
                'name': 'P5000 (Paperspace)',
                'device_type': 'RTX3090',  # Closest available choice
                'memory_gb': 16.0,
                'compute_capability': '6.1',
                'fp16_tflops': 35.0,
                'fp32_tflops': 8.8,
                'bf16_tflops': None,
                'cost_per_hour': 0.45,
                'utilization_factor': 0.70
            },
            {
                'name': 'g4dn.xlarge (AWS)',
                'device_type': 'RTX4090',  # Closest available choice
                'memory_gb': 16.0,
                'compute_capability': '7.5',
                'fp16_tflops': 65.0,
                'fp32_tflops': 8.1,
                'bf16_tflops': 65.0,
                'cost_per_hour': 3.06,
                'utilization_factor': 0.75
            },
            {
                'name': 'Standard_NC6s_v3 (Azure)',
                'device_type': 'V100',
                'memory_gb': 16.0,
                'compute_capability': '7.0',
                'fp16_tflops': 125.0,
                'fp32_tflops': 15.7,
                'bf16_tflops': None,
                'cost_per_hour': 2.88,
                'utilization_factor': 0.85
            },
            {
                'name': 'n1-highmem-8 + V100 (GCP)',
                'device_type': 'V100',
                'memory_gb': 16.0,
                'compute_capability': '7.0',
                'fp16_tflops': 125.0,
                'fp32_tflops': 15.7,
                'bf16_tflops': None,
                'cost_per_hour': 2.48,
                'utilization_factor': 0.85
            }
        ]

        for hardware_data in hardware_profiles_data:
            hardware, created = HardwareProfile.objects.get_or_create(
                name=hardware_data['name'],
                defaults=hardware_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created hardware profile: {hardware.name}')
                )

        # Create sample base model profiles
        model_profiles_data = [
            {
                'name': 'BERT Base',
                'model_type': 'TRANSFORMER',
                'parameter_count': 110000000,
                'context_length': 512,
                'vocab_size': 30522,
                'flops_per_token': 890000000000,  # Approximate FLOPs per token
                'memory_footprint_gb': 4.0,
                'supports_lora': True,
                'supports_prefix_tuning': False,
                'supported_precisions': ['FP32', 'FP16', 'BF16'],
                'hf_model_name': 'bert-base-uncased',
                'hf_requires_auth': False,
                'description': 'BERT base model for text classification and NLU tasks'
            },
            {
                'name': 'GPT-2 Medium',
                'model_type': 'TRANSFORMER',
                'parameter_count': 345000000,
                'context_length': 1024,
                'vocab_size': 50257,
                'flops_per_token': 2760000000000,
                'memory_footprint_gb': 6.0,
                'supports_lora': True,
                'supports_prefix_tuning': True,
                'supported_precisions': ['FP32', 'FP16', 'BF16'],
                'hf_model_name': 'gpt2-medium',
                'hf_requires_auth': False,
                'description': 'GPT-2 medium model for text generation tasks'
            },
            {
                'name': 'RoBERTa Large',
                'model_type': 'TRANSFORMER',
                'parameter_count': 355000000,
                'context_length': 512,
                'vocab_size': 50265,
                'flops_per_token': 2840000000000,
                'memory_footprint_gb': 8.0,
                'supports_lora': True,
                'supports_prefix_tuning': False,
                'supported_precisions': ['FP32', 'FP16', 'BF16'],
                'hf_model_name': 'roberta-large',
                'hf_requires_auth': False,
                'description': 'RoBERTa large model for advanced NLU tasks'
            },
            {
                'name': 'T5 Small',
                'model_type': 'TRANSFORMER',
                'parameter_count': 60000000,
                'context_length': 512,
                'vocab_size': 32128,
                'flops_per_token': 480000000000,
                'memory_footprint_gb': 3.0,
                'supports_lora': True,
                'supports_prefix_tuning': True,
                'supported_precisions': ['FP32', 'FP16', 'BF16'],
                'hf_model_name': 't5-small',
                'hf_requires_auth': False,
                'description': 'T5 small model for text-to-text transfer tasks'
            },
            {
                'name': 'DistilBERT',
                'model_type': 'TRANSFORMER',
                'parameter_count': 66000000,
                'context_length': 512,
                'vocab_size': 30522,
                'flops_per_token': 528000000000,
                'memory_footprint_gb': 2.0,
                'supports_lora': True,
                'supports_prefix_tuning': False,
                'supported_precisions': ['FP32', 'FP16', 'BF16'],
                'hf_model_name': 'distilbert-base-uncased',
                'hf_requires_auth': False,
                'description': 'Distilled BERT for faster inference and training'
            }
        ]

        for model_data in model_profiles_data:
            model, created = BaseModelProfile.objects.get_or_create(
                name=model_data['name'],
                defaults=model_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created model profile: {model.name}')
                )

        # Create sample dataset profiles
        dataset_profiles_data = [
            {
                'name': 'IMDB Movie Reviews',
                'dataset_type': 'TEXT',
                'num_examples': 50000,
                'avg_tokens_per_example': 230,
                'total_tokens': 11500000,
                'size_gb': 0.084,  # 84 MB converted to GB
                'hf_dataset_name': 'imdb',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Text cleaning and tokenization required',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'Binary sentiment classification on movie reviews'
            },
            {
                'name': 'AG News',
                'dataset_type': 'TEXT',
                'num_examples': 120000,
                'avg_tokens_per_example': 45,
                'total_tokens': 5400000,
                'size_gb': 0.029,  # 29 MB converted to GB
                'hf_dataset_name': 'ag_news',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Balanced dataset, minimal preprocessing needed',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'News categorization into 4 topics'
            },
            {
                'name': 'CoNLL-2003 NER',
                'dataset_type': 'TEXT',
                'num_examples': 22137,
                'avg_tokens_per_example': 14,
                'total_tokens': 309918,
                'size_gb': 0.004,  # 4 MB converted to GB
                'hf_dataset_name': 'conll2003',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'IOB tagging format, token-level labels',
                'requires_tokenization': True,
                'requires_truncation': False,
                'description': 'Named Entity Recognition dataset'
            },
            {
                'name': 'SQuAD 1.1',
                'dataset_type': 'TEXT',
                'num_examples': 107785,
                'avg_tokens_per_example': 140,
                'total_tokens': 15089900,
                'size_gb': 0.035,  # 35 MB converted to GB
                'hf_dataset_name': 'squad',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Context-question-answer triplets',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'Reading comprehension dataset'
            },
            {
                'name': 'WikiText-103',
                'dataset_type': 'TEXT',
                'num_examples': 1801350,
                'avg_tokens_per_example': 525,
                'total_tokens': 945708750,
                'size_gb': 0.181,  # 181 MB converted to GB
                'hf_dataset_name': 'wikitext',
                'hf_config_name': 'wikitext-103-raw-v1',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Pre-tokenized text, suitable for causal LM',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'Language modeling on Wikipedia articles'
            }
        ]

        for dataset_data in dataset_profiles_data:
            dataset, created = DatasetProfile.objects.get_or_create(
                name=dataset_data['name'],
                defaults=dataset_data
            )
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created dataset profile: {dataset.name}')
                )

        self.stdout.write(
            self.style.SUCCESS('Successfully populated cloud training data!')
        )
        
        # Display summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write('SUMMARY:')
        self.stdout.write(f'Cloud Providers: {CloudProvider.objects.count()}')
        self.stdout.write(f'Hardware Profiles: {HardwareProfile.objects.count()}')
        self.stdout.write(f'Model Profiles: {BaseModelProfile.objects.count()}')
        self.stdout.write(f'Dataset Profiles: {DatasetProfile.objects.count()}')
        self.stdout.write('='*50)
        
        self.stdout.write('\nNext steps:')
        self.stdout.write('1. Visit /training/cloud/ to see the cloud training dashboard')
        self.stdout.write('2. Create training configurations in /training/builder/')
        self.stdout.write('3. Start cloud training jobs and monitor them live!')
