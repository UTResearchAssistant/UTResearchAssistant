from django.db import models
import uuid
import json


class ResearchField(models.Model):
    """Research fields for categorizing datasets and models"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    parent_field = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Dataset(models.Model):
    """AI/ML Datasets"""
    
    DATASET_TYPES = [
        ('image', 'Image Dataset'),
        ('text', 'Text Dataset'),
        ('audio', 'Audio Dataset'),
        ('video', 'Video Dataset'),
        ('tabular', 'Tabular Dataset'),
        ('time_series', 'Time Series'),
        ('graph', 'Graph Dataset'),
        ('multimodal', 'Multimodal Dataset'),
    ]
    
    LICENSE_TYPES = [
        ('public', 'Public Domain'),
        ('mit', 'MIT License'),
        ('apache', 'Apache 2.0'),
        ('gpl', 'GPL License'),
        ('cc', 'Creative Commons'),
        ('commercial', 'Commercial'),
        ('restricted', 'Restricted Use'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField()
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPES)
    research_fields = models.ManyToManyField(ResearchField, related_name='datasets')
    
    # Dataset details
    source_url = models.URLField(blank=True)
    download_url = models.URLField(blank=True)
    paper_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)
    
    # Metadata
    size_mb = models.FloatField(null=True, blank=True, help_text="Dataset size in MB")
    num_samples = models.IntegerField(null=True, blank=True)
    num_classes = models.IntegerField(null=True, blank=True)
    license_type = models.CharField(max_length=20, choices=LICENSE_TYPES, default='public')
    
    # Additional metadata as JSON
    metadata = models.JSONField(default=dict, blank=True)
    
    # Tags and keywords
    tags = models.JSONField(default=list, blank=True)
    keywords = models.JSONField(default=list, blank=True)
    
    # Popularity metrics
    download_count = models.IntegerField(default=0)
    citation_count = models.IntegerField(default=0)
    star_count = models.IntegerField(default=0)
    
    # Publishing info
    authors = models.JSONField(default=list, blank=True)
    publication_year = models.IntegerField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-citation_count', '-download_count', 'name']

    def __str__(self):
        return self.name

    @property
    def tags_list(self):
        """Return tags as a list"""
        if isinstance(self.tags, str):
            try:
                return json.loads(self.tags)
            except:
                return []
        return self.tags if self.tags else []

    @property
    def keywords_list(self):
        """Return keywords as a list"""
        if isinstance(self.keywords, str):
            try:
                return json.loads(self.keywords)
            except:
                return []
        return self.keywords if self.keywords else []

    @property
    def authors_list(self):
        """Return authors as a list"""
        if isinstance(self.authors, str):
            try:
                return json.loads(self.authors)
            except:
                return []
        return self.authors if self.authors else []


class MLModel(models.Model):
    """AI/ML Models"""
    
    MODEL_TYPES = [
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('detection', 'Object Detection'),
        ('segmentation', 'Segmentation'),
        ('nlp', 'Natural Language Processing'),
        ('generation', 'Generative Model'),
        ('recommendation', 'Recommendation'),
        ('clustering', 'Clustering'),
        ('reinforcement', 'Reinforcement Learning'),
        ('multimodal', 'Multimodal'),
    ]
    
    FRAMEWORKS = [
        ('pytorch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('scikit', 'Scikit-learn'),
        ('huggingface', 'Hugging Face'),
        ('onnx', 'ONNX'),
        ('jax', 'JAX'),
        ('mxnet', 'MXNet'),
        ('other', 'Other'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField()
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    framework = models.CharField(max_length=20, choices=FRAMEWORKS)
    research_fields = models.ManyToManyField(ResearchField, related_name='models')
    
    # Model details
    architecture = models.CharField(max_length=100, blank=True)
    model_url = models.URLField(blank=True)
    paper_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)
    huggingface_url = models.URLField(blank=True)
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    
    # Model specifications
    num_parameters = models.BigIntegerField(null=True, blank=True)
    model_size_mb = models.FloatField(null=True, blank=True)
    inference_time_ms = models.FloatField(null=True, blank=True)
    
    # Training details
    training_datasets = models.ManyToManyField(Dataset, related_name='trained_models', blank=True)
    pretrained = models.BooleanField(default=False)
    
    # Additional metadata
    metadata = models.JSONField(default=dict, blank=True)
    requirements = models.JSONField(default=list, blank=True)
    
    # Tags and keywords
    tags = models.JSONField(default=list, blank=True)
    keywords = models.JSONField(default=list, blank=True)
    
    # Popularity metrics
    download_count = models.IntegerField(default=0)
    citation_count = models.IntegerField(default=0)
    star_count = models.IntegerField(default=0)
    
    # Publishing info
    authors = models.JSONField(default=list, blank=True)
    publication_year = models.IntegerField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-citation_count', '-download_count', 'name']

    def __str__(self):
        return self.name

    @property
    def tags_list(self):
        """Return tags as a list"""
        if isinstance(self.tags, str):
            try:
                return json.loads(self.tags)
            except:
                return []
        return self.tags if self.tags else []

    @property
    def keywords_list(self):
        """Return keywords as a list"""
        if isinstance(self.keywords, str):
            try:
                return json.loads(self.keywords)
            except:
                return []
        return self.keywords if self.keywords else []

    @property
    def authors_list(self):
        """Return authors as a list"""
        if isinstance(self.authors, str):
            try:
                return json.loads(self.authors)
            except:
                return []
        return self.authors if self.authors else []

    @property
    def requirements_list(self):
        """Return requirements as a list"""
        if isinstance(self.requirements, str):
            try:
                return json.loads(self.requirements)
            except:
                return []
        return self.requirements if self.requirements else []


class SearchHistory(models.Model):
    """Track search history for datasets and models"""
    
    SEARCH_TYPES = [
        ('dataset', 'Dataset Search'),
        ('model', 'Model Search'),
        ('combined', 'Combined Search'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    search_type = models.CharField(max_length=20, choices=SEARCH_TYPES)
    query = models.CharField(max_length=500)
    research_field = models.ForeignKey(ResearchField, on_delete=models.SET_NULL, null=True, blank=True)
    filters = models.JSONField(default=dict, blank=True)
    results_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.search_type}: {self.query}"
