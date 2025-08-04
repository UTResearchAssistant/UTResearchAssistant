import uuid
import json
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.conf import settings


class HardwareProfile(models.Model):
    """Hardware configuration profiles for training estimation"""
    DEVICE_TYPES = [
        ('A100', 'NVIDIA A100 80GB'),
        ('V100', 'NVIDIA V100 32GB'),
        ('RTX4090', 'NVIDIA RTX 4090 24GB'),
        ('RTX3090', 'NVIDIA RTX 3090 24GB'),
        ('H100', 'NVIDIA H100 80GB'),
        ('TPU_V4', 'Google TPU v4'),
        ('TPU_V5', 'Google TPU v5'),
        ('CUSTOM', 'Custom Hardware'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    device_type = models.CharField(max_length=20, choices=DEVICE_TYPES)
    memory_gb = models.FloatField(help_text="Available memory in GB")
    compute_capability = models.CharField(max_length=10, blank=True, help_text="CUDA compute capability")
    
    # Performance metrics
    fp16_tflops = models.FloatField(help_text="FP16 TFLOPs throughput")
    fp32_tflops = models.FloatField(help_text="FP32 TFLOPs throughput")
    bf16_tflops = models.FloatField(blank=True, null=True, help_text="BF16 TFLOPs throughput")
    
    # Cost and efficiency
    cost_per_hour = models.DecimalField(max_digits=10, decimal_places=4, help_text="Cost per GPU hour in USD")
    utilization_factor = models.FloatField(
        default=0.75, 
        validators=[MinValueValidator(0.1), MaxValueValidator(1.0)],
        help_text="Practical efficiency factor (0.1-1.0)"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.device_type})"


class BaseModelProfile(models.Model):
    """Base model configurations and specifications"""
    MODEL_TYPES = [
        ('TRANSFORMER', 'Transformer (GPT/BERT-like)'),
        ('LLAMA', 'LLaMA Family'),
        ('FALCON', 'Falcon'),
        ('MISTRAL', 'Mistral'),
        ('CUSTOM', 'Custom Architecture'),
    ]

    PRECISION_SUPPORT = [
        ('FP32', 'FP32'),
        ('FP16', 'FP16'),
        ('BF16', 'BF16'),
        ('INT8', 'INT8'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    
    # Model specifications
    parameter_count = models.BigIntegerField(help_text="Total parameters")
    context_length = models.IntegerField(help_text="Maximum context length")
    vocab_size = models.IntegerField(help_text="Vocabulary size")
    
    # Performance characteristics
    flops_per_token = models.BigIntegerField(help_text="FLOPs required per token during training")
    memory_footprint_gb = models.FloatField(help_text="Memory footprint in GB")
    
    # Capabilities
    supports_lora = models.BooleanField(default=True)
    supports_prefix_tuning = models.BooleanField(default=False)
    supported_precisions = models.JSONField(default=list, help_text="List of supported precisions")
    
    # HuggingFace integration
    hf_model_name = models.CharField(max_length=200, blank=True, help_text="HuggingFace model identifier")
    hf_requires_auth = models.BooleanField(default=False)
    
    # Metadata
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.parameter_count:,} params)"


class DatasetProfile(models.Model):
    """Dataset configurations and metadata"""
    DATASET_TYPES = [
        ('TEXT', 'Text/Language'),
        ('IMAGE', 'Image'),
        ('MULTIMODAL', 'Multimodal'),
        ('AUDIO', 'Audio'),
        ('VIDEO', 'Video'),
        ('CUSTOM', 'Custom'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPES)
    
    # Dataset characteristics
    num_examples = models.IntegerField(help_text="Number of examples")
    avg_tokens_per_example = models.IntegerField(help_text="Average tokens per example")
    total_tokens = models.BigIntegerField(help_text="Total tokens in dataset")
    size_gb = models.FloatField(help_text="Dataset size on disk in GB")
    
    # HuggingFace integration
    hf_dataset_name = models.CharField(max_length=200, blank=True, help_text="HuggingFace dataset identifier")
    hf_config_name = models.CharField(max_length=100, blank=True)
    hf_split = models.CharField(max_length=50, default='train')
    hf_requires_auth = models.BooleanField(default=False)
    
    # Preprocessing requirements
    preprocessing_notes = models.TextField(blank=True)
    requires_tokenization = models.BooleanField(default=True)
    requires_truncation = models.BooleanField(default=True)
    
    # Metadata
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.num_examples:,} examples)"


class AdapterConfiguration(models.Model):
    """LoRA and other adapter configurations"""
    ADAPTER_TYPES = [
        ('LORA', 'LoRA (Low-Rank Adaptation)'),
        ('PREFIX', 'Prefix Tuning'),
        ('PROMPT', 'Prompt Tuning'),
        ('ADAPTER', 'Adapter Layers'),
        ('NONE', 'Full Fine-tuning'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    adapter_type = models.CharField(max_length=20, choices=ADAPTER_TYPES)
    
    # LoRA specific parameters
    lora_rank = models.IntegerField(
        blank=True, null=True,
        validators=[MinValueValidator(1), MaxValueValidator(512)],
        help_text="LoRA rank (r)"
    )
    lora_alpha = models.IntegerField(
        blank=True, null=True,
        validators=[MinValueValidator(1), MaxValueValidator(1024)],
        help_text="LoRA alpha scaling"
    )
    lora_dropout = models.FloatField(
        blank=True, null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(0.5)],
        help_text="LoRA dropout rate"
    )
    target_modules = models.JSONField(
        default=list,
        help_text="Target modules for adaptation (e.g., ['q_proj', 'v_proj'])"
    )
    
    # Performance impact
    flops_overhead_per_token = models.BigIntegerField(
        default=0,
        help_text="Additional FLOPs per token for this adapter"
    )
    memory_overhead_gb = models.FloatField(
        default=0.0,
        help_text="Additional memory overhead in GB"
    )
    
    # Metadata
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        if self.adapter_type == 'LORA' and self.lora_rank:
            return f"{self.name} (LoRA r={self.lora_rank})"
        return f"{self.name} ({self.get_adapter_type_display()})"


class TrainingConfiguration(models.Model):
    """Complete training configuration"""
    OPTIMIZERS = [
        ('ADAMW', 'AdamW'),
        ('ADAM', 'Adam'),
        ('SGD', 'SGD'),
        ('RMSPROP', 'RMSprop'),
    ]

    PRECISION_CHOICES = [
        ('FP32', 'FP32'),
        ('FP16', 'FP16'),
        ('BF16', 'BF16'),
    ]

    STATUS_CHOICES = [
        ('DRAFT', 'Draft'),
        ('ESTIMATED', 'Estimated'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Configuration components
    dataset = models.ForeignKey(DatasetProfile, on_delete=models.CASCADE)
    base_model = models.ForeignKey(BaseModelProfile, on_delete=models.CASCADE)
    adapter = models.ForeignKey(AdapterConfiguration, on_delete=models.CASCADE)
    hardware = models.ForeignKey(HardwareProfile, on_delete=models.CASCADE)
    
    # Hyperparameters
    learning_rate = models.FloatField(default=5e-5)
    batch_size = models.IntegerField(default=32)
    epochs = models.IntegerField(default=3)
    gradient_accumulation_steps = models.IntegerField(default=1)
    weight_decay = models.FloatField(default=0.01)
    warmup_steps = models.IntegerField(default=100)
    max_grad_norm = models.FloatField(default=1.0)
    
    # Training settings
    optimizer = models.CharField(max_length=20, choices=OPTIMIZERS, default='ADAMW')
    precision = models.CharField(max_length=10, choices=PRECISION_CHOICES, default='FP16')
    save_steps = models.IntegerField(default=500)
    eval_steps = models.IntegerField(default=500)
    early_stopping_patience = models.IntegerField(default=3)
    
    # Estimations
    estimated_time_hours = models.FloatField(blank=True, null=True)
    estimated_cost_usd = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    estimated_gpu_hours = models.FloatField(blank=True, null=True)
    estimated_total_flops = models.BigIntegerField(blank=True, null=True)
    
    # Actual results (filled after training)
    actual_time_hours = models.FloatField(blank=True, null=True)
    actual_cost_usd = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    final_train_loss = models.FloatField(blank=True, null=True)
    final_val_loss = models.FloatField(blank=True, null=True)
    
    # Metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='DRAFT')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.dataset.name} + {self.base_model.name}"

    def calculate_estimated_metrics(self):
        """Calculate estimated training time and cost"""
        # Get precision-specific throughput
        if self.precision == 'FP16':
            throughput = self.hardware.fp16_tflops * 1e12  # Convert to FLOPs/s
        elif self.precision == 'BF16' and self.hardware.bf16_tflops:
            throughput = self.hardware.bf16_tflops * 1e12
        else:
            throughput = self.hardware.fp32_tflops * 1e12
        
        # Apply utilization factor
        effective_throughput = throughput * self.hardware.utilization_factor
        
        # Calculate total FLOPs
        base_flops_per_token = self.base_model.flops_per_token
        adapter_overhead = self.adapter.flops_overhead_per_token
        total_flops_per_token = base_flops_per_token + adapter_overhead
        
        # Total tokens processed during training
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = self.dataset.num_examples // effective_batch_size
        total_steps = steps_per_epoch * self.epochs
        total_tokens = total_steps * effective_batch_size * self.dataset.avg_tokens_per_example
        
        # Total FLOPs
        self.estimated_total_flops = total_tokens * total_flops_per_token
        
        # Estimated time
        self.estimated_time_hours = (self.estimated_total_flops / effective_throughput) / 3600
        
        # GPU hours (assuming single GPU for now)
        self.estimated_gpu_hours = self.estimated_time_hours
        
        # Cost
        self.estimated_cost_usd = float(self.hardware.cost_per_hour) * self.estimated_gpu_hours
        
        return {
            'time_hours': self.estimated_time_hours,
            'cost_usd': self.estimated_cost_usd,
            'gpu_hours': self.estimated_gpu_hours,
            'total_flops': self.estimated_total_flops
        }


class TrainingExperiment(models.Model):
    """Training experiment execution and results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    configuration = models.ForeignKey(TrainingConfiguration, on_delete=models.CASCADE)
    
    # Execution details
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    error_message = models.TextField(blank=True)
    
    # Training logs and metrics
    training_logs = models.JSONField(default=dict, help_text="Training loss per step/epoch")
    validation_logs = models.JSONField(default=dict, help_text="Validation metrics per evaluation")
    
    # Final metrics
    best_val_loss = models.FloatField(blank=True, null=True)
    convergence_epoch = models.IntegerField(blank=True, null=True)
    total_steps = models.IntegerField(blank=True, null=True)
    
    # Generated artifacts
    model_path = models.CharField(max_length=500, blank=True, help_text="Path to saved model")
    report_path = models.CharField(max_length=500, blank=True, help_text="Path to generated report")
    
    # Performance analysis
    performance_notes = models.TextField(blank=True)
    recommendations = models.JSONField(default=list, help_text="Automated recommendations")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Experiment {self.configuration.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ConfigurationComparison(models.Model):
    """Compare multiple training configurations"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    configurations = models.ManyToManyField(TrainingConfiguration)
    
    # Comparison criteria
    comparison_notes = models.TextField(blank=True)
    pareto_analysis = models.JSONField(default=dict, help_text="Pareto front analysis results")
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Comparison: {self.name}"


class CloudProvider(models.Model):
    """Cloud training providers configuration"""
    PROVIDER_TYPES = [
        ('COLAB', 'Google Colab'),
        ('KAGGLE', 'Kaggle Notebooks'),
        ('PAPERSPACE', 'Paperspace Gradient'),
        ('AWS_SAGEMAKER', 'AWS SageMaker'),
        ('AZURE_ML', 'Azure ML'),
        ('GCP_VERTEX', 'GCP Vertex AI'),
        ('CUSTOM_API', 'Custom API Endpoint'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    provider_type = models.CharField(max_length=20, choices=PROVIDER_TYPES)
    
    # API Configuration
    api_endpoint = models.URLField(help_text="API endpoint for cloud provider")
    api_key = models.CharField(max_length=500, blank=True, help_text="Encrypted API key")
    webhook_url = models.URLField(blank=True, help_text="Webhook URL for status updates")
    
    # Provider specific settings
    region = models.CharField(max_length=50, blank=True)
    instance_type = models.CharField(max_length=50, blank=True)
    max_runtime_hours = models.IntegerField(default=12, help_text="Maximum runtime in hours")
    
    # Configuration
    is_active = models.BooleanField(default=True)
    supports_live_monitoring = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.get_provider_type_display()})"


class CloudTrainingJob(models.Model):
    """Cloud training job tracking"""
    JOB_STATUS = [
        ('PENDING', 'Pending'),
        ('STARTING', 'Starting'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled'),
        ('TIMEOUT', 'Timeout'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    configuration = models.ForeignKey(TrainingConfiguration, on_delete=models.CASCADE)
    cloud_provider = models.ForeignKey(CloudProvider, on_delete=models.CASCADE)
    
    # Job identification
    external_job_id = models.CharField(max_length=200, help_text="Job ID from cloud provider")
    job_url = models.URLField(blank=True, help_text="Direct link to cloud training job")
    
    # Job details
    status = models.CharField(max_length=20, choices=JOB_STATUS, default='PENDING')
    progress_percentage = models.FloatField(default=0.0)
    current_epoch = models.IntegerField(default=0)
    total_epochs = models.IntegerField(default=0)
    
    # Timestamps
    submitted_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    last_heartbeat = models.DateTimeField(blank=True, null=True)
    
    # Resources and costs
    allocated_gpu_type = models.CharField(max_length=50, blank=True)
    actual_cost_usd = models.DecimalField(max_digits=10, decimal_places=4, blank=True, null=True)
    
    # Error handling
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    
    # Results
    final_model_url = models.URLField(blank=True, help_text="URL to download trained model")
    artifacts_url = models.URLField(blank=True, help_text="URL to training artifacts")

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        return f"Cloud Job {self.configuration.name} on {self.cloud_provider.name}"


class LiveTrainingMetrics(models.Model):
    """Real-time training metrics from cloud jobs"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cloud_job = models.ForeignKey(CloudTrainingJob, on_delete=models.CASCADE, related_name='metrics')
    
    # Timing
    timestamp = models.DateTimeField(auto_now_add=True)
    epoch = models.IntegerField()
    step = models.IntegerField()
    
    # Training metrics
    train_loss = models.FloatField()
    validation_loss = models.FloatField(blank=True, null=True)
    learning_rate = models.FloatField(blank=True, null=True)
    
    # Performance metrics
    train_accuracy = models.FloatField(blank=True, null=True)
    validation_accuracy = models.FloatField(blank=True, null=True)
    
    # System metrics
    gpu_utilization = models.FloatField(blank=True, null=True)
    memory_usage_gb = models.FloatField(blank=True, null=True)
    temperature = models.FloatField(blank=True, null=True)
    
    # Additional metrics (JSON for flexibility)
    custom_metrics = models.JSONField(default=dict, help_text="Additional custom metrics")

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['cloud_job', '-timestamp']),
            models.Index(fields=['cloud_job', 'epoch']),
        ]

    def __str__(self):
        return f"Metrics {self.cloud_job.configuration.name} - Epoch {self.epoch}"


class TrainingNotebook(models.Model):
    """Generated training notebooks for cloud platforms"""
    NOTEBOOK_TYPES = [
        ('COLAB', 'Google Colab Notebook'),
        ('KAGGLE', 'Kaggle Notebook'),
        ('JUPYTER', 'Jupyter Notebook'),
        ('PAPERSPACE', 'Paperspace Notebook'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    configuration = models.ForeignKey(TrainingConfiguration, on_delete=models.CASCADE)
    notebook_type = models.CharField(max_length=20, choices=NOTEBOOK_TYPES)
    
    # Notebook content
    notebook_content = models.JSONField(help_text="Complete notebook JSON structure")
    requirements_txt = models.TextField(help_text="Python requirements")
    
    # URLs and sharing
    notebook_url = models.URLField(blank=True, help_text="Public URL to notebook")
    sharing_enabled = models.BooleanField(default=False)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    is_template = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_notebook_type_display()} for {self.configuration.name}"


class TrainingRun(models.Model):
    """Individual training run for a configuration"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    configuration = models.ForeignKey(TrainingConfiguration, on_delete=models.CASCADE)
    
    # Execution details
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    error_message = models.TextField(blank=True)
    
    # Training logs and metrics
    training_logs = models.JSONField(default=dict, help_text="Training loss per step/epoch")
    validation_logs = models.JSONField(default=dict, help_text="Validation metrics per evaluation")
    
    # Final metrics
    best_val_loss = models.FloatField(blank=True, null=True)
    convergence_epoch = models.IntegerField(blank=True, null=True)
    total_steps = models.IntegerField(blank=True, null=True)
    
    # Generated artifacts
    model_path = models.CharField(max_length=500, blank=True, help_text="Path to saved model")
    report_path = models.CharField(max_length=500, blank=True, help_text="Path to generated report")
    
    # Performance analysis
    performance_notes = models.TextField(blank=True)
    recommendations = models.JSONField(default=list, help_text="Automated recommendations")
    
    # Metadata
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Run for {self.configuration.name} at {self.created_at}"


class HyperparameterSearch(models.Model):
    """Defines a hyperparameter search study using Optuna."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    base_configuration = models.ForeignKey(TrainingConfiguration, on_delete=models.CASCADE, null=True, blank=True)
    
    # Search space definition (as JSON)
    search_space = models.JSONField(help_text="Optuna search space definition")
    
    # Study settings
    n_trials = models.PositiveIntegerField(default=50, help_text="Number of trials to run")
    direction = models.CharField(max_length=10, choices=[('minimize', 'Minimize'), ('maximize', 'Maximize')], default='minimize')
    objective_metric = models.CharField(max_length=50, default='validation_loss')

    # Status
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name

class SearchTrial(models.Model):
    """Represents a single trial within a hyperparameter search."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    search = models.ForeignKey(HyperparameterSearch, related_name='trials', on_delete=models.CASCADE)
    trial_number = models.IntegerField()
    
    # Parameters and result
    params = models.JSONField(help_text="Parameters suggested for this trial")
    value = models.FloatField(null=True, blank=True, help_text="Objective value")
    
    # Status
    STATUS_CHOICES = [
        ('RUNNING', 'Running'),
        ('COMPLETE', 'Complete'),
        ('PRUNED', 'Pruned'),
        ('FAIL', 'Fail'),
    ]
    state = models.CharField(max_length=10, choices=STATUS_CHOICES)
    
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['search', 'trial_number']

    def __str__(self):
        return f"Trial {self.trial_number} for {self.search.name}"
