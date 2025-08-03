"""
Llama 3.2 3B Text Processing Service
Replaces SentenceTransformer with local Llama 3.2 3B model for embeddings and text analysis
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import json
import re

logger = logging.getLogger(__name__)


class Llama32TextProcessor:
    """
    Text processing service using Llama 3.2 3B model instead of SentenceTransformer
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize text processing model with fallback options"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Try to load models in order of preference
            models_to_try = [
                "meta-llama/Llama-3.2-3B",  # Original choice (gated)
                "microsoft/DialoGPT-medium",  # Open alternative for text generation
                "distilbert-base-uncased",  # Smaller open model
                "gpt2"  # Fallback open model
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Add padding token if missing
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto" if torch.cuda.is_available() else None,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'cpu'
                    logger.info(f"Successfully loaded model: {model_name}")
                    self.model_name = model_name
                    return  # Success! Exit the function
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue  # Try next model
            
            # If we get here, no models loaded successfully
            raise Exception("All model loading attempts failed")
            
        except ImportError:
            logger.warning("Transformers library not available. Install with: pip install transformers torch")
            self._fallback_initialization()
        except Exception as e:
            logger.warning(f"Failed to load any model: {e}. Using fallback methods.")
            self._fallback_initialization()
    
    def _fallback_initialization(self):
        """Fallback initialization without model"""
        self.model = None
        self.tokenizer = None
        logger.info("Using fallback text processing methods")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts using Llama 3.2 3B
        Falls back to simple methods if model not available
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_embeddings(texts)
        
        try:
            import torch
            
            embeddings = []
            
            for text in texts:
                # Prepare input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Get hidden states (embeddings)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    # Use last hidden state as embedding
                    hidden_states = outputs.hidden_states[-1]
                    # Mean pooling
                    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                    embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Llama: {e}")
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple fallback embedding method using TF-IDF-like approach"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimensionality to match typical embedding size
            svd = TruncatedSVD(n_components=min(384, tfidf_matrix.shape[1]))
            embeddings = svd.fit_transform(tfidf_matrix)
            
            return embeddings
            
        except ImportError:
            logger.warning("Scikit-learn not available for fallback embeddings")
            # Very simple fallback - character-based embeddings
            embeddings = []
            for text in texts:
                # Simple character frequency embedding
                char_counts = np.zeros(256)
                for char in text[:1000]:  # Limit text length
                    char_counts[ord(char) % 256] += 1
                embeddings.append(char_counts / max(len(text), 1))
            
            return np.array(embeddings)
    
    def extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """
        Extract keywords using Llama 3.2 3B or fallback methods
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_keyword_extraction(text, max_keywords)
        
        try:
            # Use Llama for keyword extraction
            prompt = f"""Extract the most important keywords from the following text. Return only a comma-separated list of keywords.

Text: {text[:1000]}

Keywords:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            keywords_text = response.split("Keywords:")[-1].strip()
            
            # Parse keywords
            keywords = [kw.strip() for kw in keywords_text.split(",")]
            keywords = [kw for kw in keywords if kw and len(kw) > 2]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with Llama: {e}")
            return self._fallback_keyword_extraction(text, max_keywords)
    
    def _fallback_keyword_extraction(self, text: str, max_keywords: int = 15) -> List[str]:
        """Fallback keyword extraction using frequency analysis"""
        # Simple keyword extraction
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'paper', 'research', 'study', 'work', 'approach', 'method', 'results'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text using Llama 3.2 3B or fallback methods
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_summarization(text, max_length)
        
        try:
            prompt = f"""Summarize the following text in 2-3 sentences:

{text[:2000]}

Summary:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = response.split("Summary:")[-1].strip()
            
            return summary[:max_length]
            
        except Exception as e:
            logger.error(f"Error summarizing with Llama: {e}")
            return self._fallback_summarization(text, max_length)
    
    def _fallback_summarization(self, text: str, max_length: int = 200) -> str:
        """Simple extractive summarization fallback"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 2:
            return text[:max_length]
        
        # Score sentences by position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            # First and last sentences get higher scores
            if i == 0 or i == len(sentences) - 1:
                score += 2
            # Prefer medium-length sentences
            if 50 <= len(sentence) <= 200:
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        summary_sentences = [sent for _, sent in scored_sentences[:2]]
        
        summary = " ".join(summary_sentences)
        return summary[:max_length]
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify text into categories using Llama 3.2 3B
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_classification(text, categories)
        
        try:
            categories_str = ", ".join(categories)
            prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: {text[:1000]}

The most relevant category is:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = response.split("The most relevant category is:")[-1].strip().lower()
            
            # Calculate scores
            scores = {}
            for category in categories:
                if category.lower() in prediction:
                    scores[category] = 0.9
                else:
                    scores[category] = 0.1 / len(categories)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error classifying with Llama: {e}")
            return self._fallback_classification(text, categories)
    
    def _fallback_classification(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Simple keyword-based classification fallback"""
        text_lower = text.lower()
        scores = {}
        
        # Define keywords for each category (simplified)
        category_keywords = {
            "Computer Science": ["algorithm", "computer", "software", "programming"],
            "Machine Learning": ["machine learning", "neural network", "deep learning", "ai"],
            "Natural Language Processing": ["nlp", "language", "text", "linguistic"],
            "Computer Vision": ["vision", "image", "visual", "recognition"],
            "Biomedical": ["medical", "clinical", "patient", "disease"],
            "Physics": ["quantum", "particle", "energy", "physics"],
            "Mathematics": ["mathematical", "theorem", "proof", "optimization"],
            "Engineering": ["engineering", "design", "system", "control"]
        }
        
        for category in categories:
            score = 0
            keywords = category_keywords.get(category, [category.lower()])
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            scores[category] = score / max(len(keywords), 1)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v/total_score for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(categories) for k in categories}
        
        return scores


# Create global instance
llama_text_processor = Llama32TextProcessor()
