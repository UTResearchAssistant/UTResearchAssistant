"""Embedder for document chunks.

This module orchestrates the generation of vector embeddings for
document chunks using multiple embedding models including Llama 3.2 3B,
OpenAI, and local models with fallback support.
"""

import logging
from typing import List, Union, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Service for generating embeddings from document chunks"""
    
    def __init__(self):
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the best available embedding model"""
        try:
            # Try to use the Django-integrated Llama text processor
            from django_ui.services.llama_text_processor import llama_text_processor
            self.embedding_model = llama_text_processor
            self.model_type = "llama"
            logger.info("Using Llama 3.2 3B for embeddings")
        except ImportError:
            try:
                # Fallback to OpenAI
                import openai
                self.embedding_model = openai
                self.model_type = "openai"
                logger.info("Using OpenAI for embeddings")
            except ImportError:
                try:
                    # Fallback to SentenceTransformers
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.model_type = "sentence_transformer"
                    logger.info("Using SentenceTransformer for embeddings")
                except ImportError:
                    # Final fallback to simple embeddings
                    self.embedding_model = None
                    self.model_type = "simple"
                    logger.warning("Using simple fallback embeddings")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts
        
        Parameters
        ----------
        texts : list[str]
            List of text strings to embed
        
        Returns
        -------
        list[list[float]]
            List of embedding vectors
        """
        if self.model_type == "llama":
            return self._llama_embeddings(texts)
        elif self.model_type == "openai":
            return self._openai_embeddings(texts)
        elif self.model_type == "sentence_transformer":
            return self._sentence_transformer_embeddings(texts)
        else:
            return self._simple_embeddings(texts)
    
    def _llama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Llama 3.2 3B"""
        try:
            embeddings_array = self.embedding_model.generate_embeddings(texts)
            return embeddings_array.tolist()
        except Exception as e:
            logger.error(f"Llama embedding error: {e}")
            return self._simple_embeddings(texts)
    
    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            embeddings = []
            for text in texts:
                response = self.embedding_model.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response['data'][0]['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return self._simple_embeddings(texts)
    
    def _sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {e}")
            return self._simple_embeddings(texts)
    
    def _simple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Simple fallback embeddings using TF-IDF-like approach"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimensionality to standard embedding size
            svd = TruncatedSVD(n_components=min(384, tfidf_matrix.shape[1]))
            embeddings = svd.fit_transform(tfidf_matrix)
            
            return embeddings.tolist()
            
        except ImportError:
            # Ultimate fallback - character-based embeddings
            embeddings = []
            for text in texts:
                # Simple character frequency embedding
                char_counts = np.zeros(256)
                for char in text[:1000]:  # Limit text length
                    char_counts[ord(char) % 256] += 1
                # Normalize
                char_counts = char_counts / max(len(text), 1)
                embeddings.append(char_counts.tolist())
            
            return embeddings


def embed_document_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks.

    Parameters
    ----------
    chunks : list[str]
        A list of text segments from a document.

    Returns
    -------
    list[list[float]]
        A list of embedding vectors.
    """
    embedder = DocumentEmbedder()
    return embedder.generate_embeddings(chunks)


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for texts (numpy array output).
    
    Parameters
    ----------
    texts : list[str]
        List of texts to embed
    
    Returns
    -------
    np.ndarray
        Array of embeddings
    """
    embedder = DocumentEmbedder()
    embeddings_list = embedder.generate_embeddings(texts)
    return np.array(embeddings_list)


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text.
    
    Parameters
    ----------
    text : str
        Text to embed
    
    Returns
    -------
    list[float]
        Embedding vector
    """
    embedder = DocumentEmbedder()
    embeddings = embedder.generate_embeddings([text])
    return embeddings[0] if embeddings else []


# Create global instance for compatibility
document_embedder = DocumentEmbedder()
