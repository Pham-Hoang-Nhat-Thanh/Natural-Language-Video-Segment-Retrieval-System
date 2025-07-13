"""
Text Encoder Service for converting natural language queries to embeddings.
Supports multiple embedding models with caching and batch processing.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional, Union
import time
import hashlib
import pickle
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
from config import Settings

# Try to import CLIP if available
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available, falling back to sentence transformers")

logger = logging.getLogger(__name__)

class TextEncoder:
    """High-performance text encoder with caching and multiple model support."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.cache = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Redis cache if available
        try:
            self.cache = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False,
                socket_timeout=1.0
            )
            self.cache.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.cache = None
        
        # Initialize default model
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the default sentence transformer model."""
        try:
            # Try to use local CLIP models first if available
            local_model_path = Path("/app/models/clip")
            if CLIP_AVAILABLE and local_model_path.exists():
                logger.info("Using local CLIP model for text encoding")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Try to load CLIP model from local path
                model_files = list(local_model_path.glob("*.pt"))
                if model_files:
                    # Use the first available model
                    model_file = model_files[0]
                    model_name = model_file.stem.replace("-", "/")  # Convert ViT-B-32.pt to ViT-B/32
                    
                    logger.info(f"Loading CLIP model: {model_name} from {model_file}")
                    model, preprocess = clip.load(model_name, device=device, download_root=str(local_model_path.parent))
                    
                    # Wrap CLIP model to match SentenceTransformer interface
                    class CLIPTextEncoder:
                        def __init__(self, clip_model, device):
                            self.clip_model = clip_model
                            self.device = device
                            self.embedding_dim = 512  # CLIP ViT-B/32 text embedding dimension
                            self.max_seq_length = 77   # CLIP max sequence length
                        
                        def encode(self, texts, **kwargs):
                            if isinstance(texts, str):
                                texts = [texts]
                            
                            with torch.no_grad():
                                text_tokens = clip.tokenize(texts).to(self.device)
                                text_features = self.clip_model.encode_text(text_tokens)
                                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                return text_features.cpu().numpy()
                        
                        def eval(self):
                            self.clip_model.eval()
                        
                        def get_sentence_embedding_dimension(self):
                            return self.embedding_dim
                    
                    wrapped_model = CLIPTextEncoder(model, device)
                    wrapped_model.eval()
                    
                    self.models["default"] = {
                        "model": wrapped_model,
                        "embedding_dim": wrapped_model.embedding_dim,
                        "max_length": wrapped_model.max_seq_length
                    }
                    
                    logger.info(f"Loaded CLIP model with embedding dimension: {self.models['default']['embedding_dim']}")
                    return
            
            # Fallback to sentence transformers
            model_name = self.settings.TEXT_ENCODER_MODEL
            logger.info(f"Loading text encoder model: {model_name}")
            
            # Use sentence-transformers for easier handling
            model = SentenceTransformer(model_name, device=self.device)
            model.eval()
            
            self.models["default"] = {
                "model": model,
                "embedding_dim": model.get_sentence_embedding_dimension(),
                "max_length": model.max_seq_length
            }
            
            logger.info(f"Loaded model with embedding dimension: {self.models['default']['embedding_dim']}")
            
        except Exception as e:
            logger.error(f"Failed to load text encoder model: {e}")
            # Fallback to a lightweight model
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.models["default"] = {
                    "model": model,
                    "embedding_dim": model.get_sentence_embedding_dimension(),
                    "max_length": model.max_seq_length
                }
                logger.info("Loaded fallback model: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise
    
    def _get_cache_key(self, text: str, model_name: str = "default") -> str:
        """Generate cache key for text embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"text_embed:{model_name}:{text_hash}"
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray, ttl: int = 3600):
        """Cache embedding with TTL."""
        if not self.cache:
            return
        
        try:
            self.cache.setex(
                cache_key,
                ttl,
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    async def encode_text(
        self,
        text: str,
        model_name: str = "default",
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text to encode
            model_name: Model to use for encoding
            normalize: Whether to normalize the embedding
            
        Returns:
            Normalized embedding vector
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text, model_name)
        cached_embedding = self._get_cached_embedding(cache_key)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for text encoding: {time.time() - start_time:.3f}s")
            return cached_embedding
        
        # Get model
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.models[model_name]
        model = model_info["model"]
        
        # Encode text in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self._encode_text_sync,
            model,
            text,
            normalize
        )
        
        # Cache the result
        self._cache_embedding(cache_key, embedding)
        
        encode_time = time.time() - start_time
        logger.debug(f"Text encoding completed in {encode_time:.3f}s")
        
        return embedding
    
    def _encode_text_sync(
        self,
        model: SentenceTransformer,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Synchronous text encoding."""
        with torch.no_grad():
            # Encode text
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embedding.astype(np.float32)
    
    async def encode_batch(
        self,
        texts: List[str],
        model_name: str = "default",
        normalize: bool = True,
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Encode multiple texts in batches.
        
        Args:
            texts: List of texts to encode
            model_name: Model to use for encoding
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        start_time = time.time()
        embeddings = []
        
        # Check cache for all texts
        cache_keys = [self._get_cache_key(text, model_name) for text in texts]
        cached_embeddings = {}
        
        if self.cache:
            try:
                cached_data = self.cache.mget(cache_keys)
                for i, data in enumerate(cached_data):
                    if data:
                        cached_embeddings[i] = pickle.loads(data)
            except Exception as e:
                logger.warning(f"Batch cache retrieval error: {e}")
        
        # Identify texts that need encoding
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            if i not in cached_embeddings:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode missing texts
        if texts_to_encode:
            model_info = self.models[model_name]
            model = model_info["model"]
            
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                self.executor,
                self._encode_batch_sync,
                model,
                texts_to_encode,
                normalize,
                batch_size
            )
            
            # Cache new embeddings
            if self.cache:
                try:
                    cache_data = {}
                    for i, embedding in enumerate(new_embeddings):
                        idx = indices_to_encode[i]
                        cache_key = cache_keys[idx]
                        cache_data[cache_key] = pickle.dumps(embedding)
                    
                    if cache_data:
                        self.cache.mset(cache_data)
                        # Set TTL for all keys
                        for key in cache_data.keys():
                            self.cache.expire(key, 3600)
                            
                except Exception as e:
                    logger.warning(f"Batch cache storage error: {e}")
            
            # Add new embeddings to cached ones
            for i, embedding in enumerate(new_embeddings):
                idx = indices_to_encode[i]
                cached_embeddings[idx] = embedding
        
        # Assemble final embeddings in order
        embeddings = [cached_embeddings[i] for i in range(len(texts))]
        
        encode_time = time.time() - start_time
        logger.info(f"Batch encoding of {len(texts)} texts completed in {encode_time:.3f}s")
        
        return embeddings
    
    def _encode_batch_sync(
        self,
        model: SentenceTransformer,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """Synchronous batch text encoding."""
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return [emb.astype(np.float32) for emb in embeddings]
    
    def get_embedding_dimension(self, model_name: str = "default") -> int:
        """Get embedding dimension for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name]["embedding_dim"]
    
    def get_max_length(self, model_name: str = "default") -> int:
        """Get maximum sequence length for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name]["max_length"]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the text encoder."""
        try:
            # Test encoding
            test_text = "This is a test query"
            embedding = await self.encode_text(test_text)
            
            return {
                "status": "healthy",
                "models_loaded": list(self.models.keys()),
                "embedding_dim": self.get_embedding_dimension(),
                "device": str(self.device),
                "cache_available": self.cache is not None,
                "test_embedding_shape": embedding.shape
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
