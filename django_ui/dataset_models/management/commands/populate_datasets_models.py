from django.core.management.base import BaseCommand
from dataset_models.models import ResearchField, Dataset, MLModel


class Command(BaseCommand):
    help = 'Populate database with sample datasets and models'

    def handle(self, *args, **options):
        self.stdout.write('Creating research fields...')
        
        # Create research fields
        fields_data = [
            ('Computer Vision', 'Research in visual perception and image understanding'),
            ('Natural Language Processing', 'Processing and understanding human language'),
            ('Machine Learning', 'Algorithms that learn from data'),
            ('Deep Learning', 'Neural networks with multiple layers'),
            ('Reinforcement Learning', 'Learning through interaction with environment'),
            ('Speech Recognition', 'Converting speech to text'),
            ('Robotics', 'Autonomous systems and robot control'),
            ('Bioinformatics', 'Computational analysis of biological data'),
            ('Medical AI', 'AI applications in healthcare and medicine'),
            ('Time Series Analysis', 'Analysis of sequential data over time'),
        ]
        
        fields = {}
        for name, description in fields_data:
            field, created = ResearchField.objects.get_or_create(
                name=name,
                defaults={'description': description}
            )
            fields[name] = field
            if created:
                self.stdout.write(f'  Created field: {name}')

        self.stdout.write('Creating sample datasets...')
        
        # Create sample datasets
        datasets_data = [
            {
                'name': 'ImageNet',
                'description': 'Large visual database designed for use in visual object recognition research',
                'dataset_type': 'image',
                'license_type': 'public',
                'research_fields': ['Computer Vision', 'Deep Learning'],
                'source_url': 'https://www.image-net.org/',
                'num_samples': 14197122,
                'num_classes': 21841,
                'size_mb': 160000.0,
                'citation_count': 98000,
                'download_count': 250000,
                'tags': ['classification', 'object detection', 'benchmark'],
                'keywords': ['visual recognition', 'deep learning', 'CNN'],
                'authors': ['Jia Deng', 'Wei Dong', 'Richard Socher', 'Li-Jia Li'],
                'publication_year': 2009,
            },
            {
                'name': 'MNIST',
                'description': 'Database of handwritten digits commonly used for training image processing systems',
                'dataset_type': 'image',
                'license_type': 'public',
                'research_fields': ['Computer Vision', 'Machine Learning'],
                'source_url': 'http://yann.lecun.com/exdb/mnist/',
                'num_samples': 70000,
                'num_classes': 10,
                'size_mb': 11.0,
                'citation_count': 45000,
                'download_count': 500000,
                'tags': ['handwriting', 'digits', 'classification'],
                'keywords': ['OCR', 'pattern recognition', 'benchmark'],
                'authors': ['Yann LeCun', 'Corinna Cortes', 'Christopher Burges'],
                'publication_year': 1998,
            },
            {
                'name': 'CIFAR-10',
                'description': '60,000 32x32 color images in 10 classes with 6,000 images per class',
                'dataset_type': 'image',
                'license_type': 'public',
                'research_fields': ['Computer Vision', 'Machine Learning'],
                'num_samples': 60000,
                'num_classes': 10,
                'size_mb': 163.0,
                'citation_count': 25000,
                'download_count': 300000,
                'tags': ['classification', 'small images', 'benchmark'],
                'keywords': ['object recognition', 'deep learning', 'CNN'],
                'publication_year': 2009,
            },
            {
                'name': 'Common Crawl',
                'description': 'Repository of web crawl data composed of over 8 billion web pages',
                'dataset_type': 'text',
                'license_type': 'public',
                'research_fields': ['Natural Language Processing', 'Machine Learning'],
                'source_url': 'https://commoncrawl.org/',
                'num_samples': 8000000000,
                'size_mb': 50000000.0,
                'citation_count': 15000,
                'download_count': 50000,
                'tags': ['web crawl', 'large corpus', 'multilingual'],
                'keywords': ['NLP', 'language modeling', 'web data'],
                'publication_year': 2011,
            },
            {
                'name': 'IMDB Movie Reviews',
                'description': 'Dataset for binary sentiment classification with highly polar movie reviews',
                'dataset_type': 'text',
                'license_type': 'public',
                'research_fields': ['Natural Language Processing'],
                'num_samples': 50000,
                'num_classes': 2,
                'size_mb': 80.0,
                'citation_count': 8000,
                'download_count': 150000,
                'tags': ['sentiment analysis', 'movie reviews', 'binary classification'],
                'keywords': ['sentiment', 'text classification', 'NLP'],
                'publication_year': 2011,
            },
            {
                'name': 'LibriSpeech',
                'description': 'Corpus of approximately 1000 hours of read English speech',
                'dataset_type': 'audio',
                'license_type': 'cc',
                'research_fields': ['Speech Recognition', 'Natural Language Processing'],
                'num_samples': 281241,
                'size_mb': 60000.0,
                'citation_count': 3500,
                'download_count': 25000,
                'tags': ['speech recognition', 'ASR', 'audiobooks'],
                'keywords': ['speech', 'automatic speech recognition', 'audio'],
                'publication_year': 2015,
            },
        ]
        
        for dataset_data in datasets_data:
            research_field_names = dataset_data.pop('research_fields')
            dataset, created = Dataset.objects.get_or_create(
                name=dataset_data['name'],
                defaults=dataset_data
            )
            if created:
                # Add research fields
                for field_name in research_field_names:
                    if field_name in fields:
                        dataset.research_fields.add(fields[field_name])
                self.stdout.write(f'  Created dataset: {dataset.name}')

        self.stdout.write('Creating sample models...')
        
        # Create sample models
        models_data = [
            {
                'name': 'ResNet-50',
                'description': 'Deep residual network with 50 layers for image classification',
                'model_type': 'classification',
                'framework': 'pytorch',
                'architecture': 'ResNet',
                'research_fields': ['Computer Vision', 'Deep Learning'],
                'num_parameters': 25557032,
                'model_size_mb': 98.0,
                'accuracy': 76.15,
                'pretrained': True,
                'citation_count': 89000,
                'download_count': 1200000,
                'tags': ['residual networks', 'image classification', 'CNN'],
                'keywords': ['deep learning', 'computer vision', 'ResNet'],
                'authors': ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren', 'Jian Sun'],
                'publication_year': 2016,
                'huggingface_url': 'https://huggingface.co/microsoft/resnet-50',
            },
            {
                'name': 'BERT-Base',
                'description': 'Bidirectional encoder representations from transformers',
                'model_type': 'nlp',
                'framework': 'huggingface',
                'architecture': 'Transformer',
                'research_fields': ['Natural Language Processing', 'Deep Learning'],
                'num_parameters': 110000000,
                'model_size_mb': 440.0,
                'f1_score': 0.884,
                'pretrained': True,
                'citation_count': 65000,
                'download_count': 2500000,
                'tags': ['transformer', 'language model', 'bidirectional'],
                'keywords': ['NLP', 'BERT', 'pre-training'],
                'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
                'publication_year': 2018,
                'huggingface_url': 'https://huggingface.co/bert-base-uncased',
            },
            {
                'name': 'YOLOv5',
                'description': 'Real-time object detection model with high accuracy and speed',
                'model_type': 'detection',
                'framework': 'pytorch',
                'architecture': 'YOLO',
                'research_fields': ['Computer Vision'],
                'num_parameters': 46200000,
                'model_size_mb': 90.0,
                'accuracy': 56.8,
                'inference_time_ms': 15.0,
                'pretrained': True,
                'citation_count': 12000,
                'download_count': 800000,
                'tags': ['object detection', 'real-time', 'YOLO'],
                'keywords': ['detection', 'computer vision', 'real-time'],
                'github_url': 'https://github.com/ultralytics/yolov5',
                'publication_year': 2020,
            },
            {
                'name': 'GPT-2',
                'description': 'Generative pre-trained transformer for text generation',
                'model_type': 'generation',
                'framework': 'huggingface',
                'architecture': 'Transformer',
                'research_fields': ['Natural Language Processing'],
                'num_parameters': 1500000000,
                'model_size_mb': 6000.0,
                'pretrained': True,
                'citation_count': 25000,
                'download_count': 750000,
                'tags': ['text generation', 'transformer', 'autoregressive'],
                'keywords': ['GPT', 'language generation', 'transformer'],
                'authors': ['Alec Radford', 'Jeffrey Wu', 'Rewon Child'],
                'publication_year': 2019,
                'huggingface_url': 'https://huggingface.co/gpt2',
            },
            {
                'name': 'EfficientNet-B7',
                'description': 'Highly efficient convolutional neural network for image classification',
                'model_type': 'classification',
                'framework': 'tensorflow',
                'architecture': 'EfficientNet',
                'research_fields': ['Computer Vision', 'Deep Learning'],
                'num_parameters': 66347960,
                'model_size_mb': 256.0,
                'accuracy': 84.4,
                'pretrained': True,
                'citation_count': 15000,
                'download_count': 450000,
                'tags': ['efficient', 'image classification', 'scaling'],
                'keywords': ['efficient networks', 'computer vision', 'scaling'],
                'authors': ['Mingxing Tan', 'Quoc V. Le'],
                'publication_year': 2019,
            },
        ]
        
        for model_data in models_data:
            research_field_names = model_data.pop('research_fields')
            model, created = MLModel.objects.get_or_create(
                name=model_data['name'],
                defaults=model_data
            )
            if created:
                # Add research fields
                for field_name in research_field_names:
                    if field_name in fields:
                        model.research_fields.add(fields[field_name])
                self.stdout.write(f'  Created model: {model.name}')

        self.stdout.write(
            self.style.SUCCESS('Successfully populated database with sample data!')
        )
