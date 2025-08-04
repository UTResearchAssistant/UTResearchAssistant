from django.core.management.base import BaseCommand
from training_config.models import (
    HardwareProfile, BaseModelProfile, DatasetProfile, AdapterConfiguration
)


class Command(BaseCommand):
    help = 'Populate sample data for training configuration system'

    def handle(self, *args, **options):
        self.stdout.write('Creating sample training configuration data...')
        
        # Create Hardware Profiles
        hardware_profiles = [
            {
                'name': 'NVIDIA A100 80GB',
                'device_type': 'A100',
                'memory_gb': 80.0,
                'compute_capability': '8.0',
                'fp16_tflops': 312.0,
                'fp32_tflops': 156.0,
                'bf16_tflops': 312.0,
                'cost_per_hour': 3.06,
                'utilization_factor': 0.75,
            },
            {
                'name': 'NVIDIA V100 32GB',
                'device_type': 'V100',
                'memory_gb': 32.0,
                'compute_capability': '7.0',
                'fp16_tflops': 125.0,
                'fp32_tflops': 15.7,
                'cost_per_hour': 2.48,
                'utilization_factor': 0.70,
            },
            {
                'name': 'NVIDIA RTX 4090',
                'device_type': 'RTX4090',
                'memory_gb': 24.0,
                'compute_capability': '8.9',
                'fp16_tflops': 165.0,
                'fp32_tflops': 83.0,
                'cost_per_hour': 0.95,
                'utilization_factor': 0.65,
            },
            {
                'name': 'Google TPU v4 Pod',
                'device_type': 'TPU_V4',
                'memory_gb': 32.0,
                'fp16_tflops': 275.0,
                'fp32_tflops': 137.0,
                'bf16_tflops': 275.0,
                'cost_per_hour': 4.20,
                'utilization_factor': 0.80,
            },
        ]
        
        for hw_data in hardware_profiles:
            hw, created = HardwareProfile.objects.get_or_create(
                name=hw_data['name'],
                defaults=hw_data
            )
            if created:
                self.stdout.write(f'✓ Created hardware profile: {hw.name}')
        
        # Create Base Model Profiles
        model_profiles = [
            {
                'name': 'LLaMA-7B',
                'model_type': 'LLAMA',
                'parameter_count': 7000000000,
                'context_length': 2048,
                'vocab_size': 32000,
                'flops_per_token': 14000000000000,  # 14T FLOPs per token
                'memory_footprint_gb': 28.0,
                'supports_lora': True,
                'supports_prefix_tuning': True,
                'supported_precisions': ['FP16', 'BF16', 'FP32'],
                'hf_model_name': 'meta-llama/Llama-2-7b-hf',
                'hf_requires_auth': True,
                'description': 'LLaMA 7B parameter model, efficient for fine-tuning',
            },
            {
                'name': 'LLaMA-13B',
                'model_type': 'LLAMA',
                'parameter_count': 13000000000,
                'context_length': 2048,
                'vocab_size': 32000,
                'flops_per_token': 26000000000000,  # 26T FLOPs per token
                'memory_footprint_gb': 52.0,
                'supports_lora': True,
                'supports_prefix_tuning': True,
                'supported_precisions': ['FP16', 'BF16', 'FP32'],
                'hf_model_name': 'meta-llama/Llama-2-13b-hf',
                'hf_requires_auth': True,
                'description': 'LLaMA 13B parameter model, higher capacity',
            },
            {
                'name': 'Falcon-7B',
                'model_type': 'FALCON',
                'parameter_count': 7000000000,
                'context_length': 2048,
                'vocab_size': 65024,
                'flops_per_token': 14000000000000,
                'memory_footprint_gb': 28.0,
                'supports_lora': True,
                'supports_prefix_tuning': False,
                'supported_precisions': ['FP16', 'BF16', 'FP32'],
                'hf_model_name': 'tiiuae/falcon-7b',
                'hf_requires_auth': False,
                'description': 'Falcon 7B, trained on diverse web data',
            },
            {
                'name': 'Mistral-7B',
                'model_type': 'MISTRAL',
                'parameter_count': 7300000000,
                'context_length': 8192,
                'vocab_size': 32000,
                'flops_per_token': 14600000000000,
                'memory_footprint_gb': 29.0,
                'supports_lora': True,
                'supports_prefix_tuning': True,
                'supported_precisions': ['FP16', 'BF16', 'FP32'],
                'hf_model_name': 'mistralai/Mistral-7B-v0.1',
                'hf_requires_auth': False,
                'description': 'Mistral 7B with 8k context length',
            },
        ]
        
        for model_data in model_profiles:
            model, created = BaseModelProfile.objects.get_or_create(
                name=model_data['name'],
                defaults=model_data
            )
            if created:
                self.stdout.write(f'✓ Created model profile: {model.name}')
        
        # Create Dataset Profiles
        dataset_profiles = [
            {
                'name': 'Alpaca Dataset',
                'dataset_type': 'TEXT',
                'num_examples': 52000,
                'avg_tokens_per_example': 80,
                'total_tokens': 4160000,
                'size_gb': 0.025,
                'hf_dataset_name': 'tatsu-lab/alpaca',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Instruction-following dataset',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'Stanford Alpaca instruction-following dataset',
            },
            {
                'name': 'OpenAssistant Conversations',
                'dataset_type': 'TEXT',
                'num_examples': 161000,
                'avg_tokens_per_example': 120,
                'total_tokens': 19320000,
                'size_gb': 0.12,
                'hf_dataset_name': 'OpenAssistant/oasst1',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Multi-turn conversations',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'OpenAssistant conversational dataset',
            },
            {
                'name': 'CodeAlpaca Dataset',
                'dataset_type': 'TEXT',
                'num_examples': 20000,
                'avg_tokens_per_example': 150,
                'total_tokens': 3000000,
                'size_gb': 0.02,
                'hf_dataset_name': 'sahil2801/CodeAlpaca-20k',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'Code instruction dataset',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'Code generation instruction dataset',
            },
            {
                'name': 'LIMA Dataset',
                'dataset_type': 'TEXT',
                'num_examples': 1030,
                'avg_tokens_per_example': 200,
                'total_tokens': 206000,
                'size_gb': 0.001,
                'hf_dataset_name': 'GAIR/lima',
                'hf_config_name': '',
                'hf_split': 'train',
                'hf_requires_auth': False,
                'preprocessing_notes': 'High-quality alignment dataset',
                'requires_tokenization': True,
                'requires_truncation': True,
                'description': 'LIMA high-quality alignment dataset',
            },
        ]
        
        for dataset_data in dataset_profiles:
            dataset, created = DatasetProfile.objects.get_or_create(
                name=dataset_data['name'],
                defaults=dataset_data
            )
            if created:
                self.stdout.write(f'✓ Created dataset profile: {dataset.name}')
        
        # Create Adapter Configurations
        adapter_configs = [
            {
                'name': 'LoRA Rank 8',
                'adapter_type': 'LORA',
                'lora_rank': 8,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj'],
                'flops_overhead_per_token': 100000000,  # 100M FLOPs overhead
                'memory_overhead_gb': 0.5,
                'description': 'Standard LoRA configuration with rank 8',
            },
            {
                'name': 'LoRA Rank 16',
                'adapter_type': 'LORA',
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'flops_overhead_per_token': 200000000,  # 200M FLOPs overhead
                'memory_overhead_gb': 1.0,
                'description': 'Higher capacity LoRA with rank 16',
            },
            {
                'name': 'LoRA Rank 32',
                'adapter_type': 'LORA',
                'lora_rank': 32,
                'lora_alpha': 64,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'flops_overhead_per_token': 400000000,  # 400M FLOPs overhead
                'memory_overhead_gb': 2.0,
                'description': 'High capacity LoRA with rank 32',
            },
            {
                'name': 'Full Fine-tuning',
                'adapter_type': 'NONE',
                'target_modules': [],
                'flops_overhead_per_token': 0,
                'memory_overhead_gb': 0.0,
                'description': 'Full parameter fine-tuning without adapters',
            },
            {
                'name': 'QLoRA Rank 64',
                'adapter_type': 'LORA',
                'lora_rank': 64,
                'lora_alpha': 128,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'flops_overhead_per_token': 800000000,  # 800M FLOPs overhead
                'memory_overhead_gb': 4.0,
                'description': 'High-rank LoRA for maximum adaptation capacity',
            },
        ]
        
        for adapter_data in adapter_configs:
            adapter, created = AdapterConfiguration.objects.get_or_create(
                name=adapter_data['name'],
                defaults=adapter_data
            )
            if created:
                self.stdout.write(f'✓ Created adapter configuration: {adapter.name}')
        
        self.stdout.write(
            self.style.SUCCESS(
                '\n✅ Successfully populated sample training configuration data!'
                '\n\nYou can now:'
                '\n• Visit /training/ to access the training configuration dashboard'
                '\n• Create new training configurations with the builder'
                '\n• Estimate training costs and time'
                '\n• Compare different configurations'
            )
        )
