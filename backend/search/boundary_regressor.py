"""
Boundary Regressor for refining video segment boundaries.
Uses a regression model to predict more accurate start/end times for video segments.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import redis
import hashlib
from pathlib import Path
from config import Settings

logger = logging.getLogger(__name__)

class BoundaryRegressionModel(nn.Module):
    """Neural network model for boundary regression."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Boundary regressors
        self.start_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.end_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.encoder(x)
        
        start_offset = self.start_regressor(features)
        end_offset = self.end_regressor(features)
        confidence = self.confidence_estimator(features)
        
        return {
            'start_offset': start_offset,
            'end_offset': end_offset,
            'confidence': confidence
        }

class BoundaryRegressor:
    """Boundary regression service for refining video segment boundaries."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.text_encoder = None
        self.cache = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_path = Path(settings.MODELS_PATH) / "boundary_regressor.pth"
        
        # Initialize Redis cache if available
        try:
            self.cache = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False,
                socket_timeout=1.0
            )
            self.cache.ping()
            logger.info("Connected to Redis cache for boundary regressor")
        except Exception as e:
            logger.warning(f"Redis cache not available for boundary regressor: {e}")
            self.cache = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the boundary regression model."""
        try:
            # Load text encoder for feature extraction
            encoder_model = self.settings.TEXT_ENCODER_MODEL
            logger.info(f"Loading text encoder for boundary regression: {encoder_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.text_encoder = AutoModel.from_pretrained(encoder_model)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
            
            # Initialize regression model
            embedding_dim = self.text_encoder.config.hidden_size
            self.model = BoundaryRegressionModel(input_dim=embedding_dim)
            self.model.to(self.device)
            
            # Load pre-trained weights if available
            if self.model_path.exists():
                logger.info(f"Loading pre-trained boundary regression model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded pre-trained boundary regression model")
            else:
                logger.warning("No pre-trained boundary regression model found, using random initialization")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize boundary regression model: {e}")
            self.model = None
            self.text_encoder = None
            self.tokenizer = None
    
    def _get_cache_key(self, query: str, original_start: float, original_end: float) -> str:
        """Generate cache key for boundary regression."""
        key_text = f"{query}:{original_start}:{original_end}"
        key_hash = hashlib.md5(key_text.encode()).hexdigest()
        return f"boundary_regress:{key_hash}"
    
    def _get_cached_boundaries(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Retrieve cached boundary regression result."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Boundary regression cache retrieval error: {e}")
        
        return None
    
    def _cache_boundaries(self, cache_key: str, boundaries: Dict[str, float], ttl: int = 3600):
        """Cache boundary regression result with TTL."""
        if not self.cache:
            return
        
        try:
            self.cache.setex(
                cache_key,
                ttl,
                pickle.dumps(boundaries)
            )
        except Exception as e:
            logger.warning(f"Boundary regression cache storage error: {e}")
    
    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode query text to embeddings."""
        if not self.text_encoder or not self.tokenizer:
            raise RuntimeError("Text encoder not available")
        
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use CLS token embedding or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    async def refine_boundaries(
        self,
        query: str,
        original_start: float,
        original_end: float,
        video_duration: float,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Refine video segment boundaries using regression model.
        
        Args:
            query: Search query
            original_start: Original start time in seconds
            original_end: Original end time in seconds
            video_duration: Total video duration in seconds
            confidence_threshold: Minimum confidence for applying refinement
            
        Returns:
            Dictionary with refined boundaries and confidence scores
        """
        start_time = time.time()
        
        if not self.model:
            # Return original boundaries if model not available
            return {
                'start_time': original_start,
                'end_time': original_end,
                'confidence': 0.0,
                'refined': False,
                'original_start': original_start,
                'original_end': original_end
            }
        
        # Check cache
        cache_key = self._get_cache_key(query, original_start, original_end)
        cached_result = self._get_cached_boundaries(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for boundary regression")
            return cached_result
        
        try:
            # Encode query
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self._encode_query,
                query
            )
            
            # Run boundary regression
            prediction = await loop.run_in_executor(
                self.executor,
                self._predict_boundaries_sync,
                query_embedding,
                original_start,
                original_end,
                video_duration
            )
            
            # Extract predictions
            start_offset = float(prediction['start_offset'].cpu().item())
            end_offset = float(prediction['end_offset'].cpu().item())
            confidence = float(prediction['confidence'].cpu().item())
            
            # Convert offsets to actual timestamps
            segment_duration = original_end - original_start
            
            # Offsets are in [0, 1] range, scale to segment duration
            start_adjustment = (start_offset - 0.5) * segment_duration * 0.5  # Max 25% adjustment
            end_adjustment = (end_offset - 0.5) * segment_duration * 0.5
            
            refined_start = max(0, original_start + start_adjustment)
            refined_end = min(video_duration, original_end + end_adjustment)
            
            # Ensure start < end
            if refined_start >= refined_end:
                refined_start = original_start
                refined_end = original_end
                confidence = 0.0
            
            # Apply refinement only if confidence is high enough
            apply_refinement = confidence >= confidence_threshold
            
            result = {
                'start_time': refined_start if apply_refinement else original_start,
                'end_time': refined_end if apply_refinement else original_end,
                'confidence': confidence,
                'refined': apply_refinement,
                'original_start': original_start,
                'original_end': original_end,
                'start_adjustment': start_adjustment if apply_refinement else 0.0,
                'end_adjustment': end_adjustment if apply_refinement else 0.0
            }
            
            # Cache the result
            self._cache_boundaries(cache_key, result)
            
            refine_time = time.time() - start_time
            logger.debug(f"Boundary refinement completed in {refine_time:.3f}s, confidence: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in boundary refinement: {e}")
            # Return original boundaries on error
            result = {
                'start_time': original_start,
                'end_time': original_end,
                'confidence': 0.0,
                'refined': False,
                'original_start': original_start,
                'original_end': original_end,
                'error': str(e)
            }
            return result
    
    def _predict_boundaries_sync(
        self,
        query_embedding: torch.Tensor,
        original_start: float,
        original_end: float,
        video_duration: float
    ) -> Dict[str, torch.Tensor]:
        """Synchronous boundary prediction."""
        with torch.no_grad():
            # Add additional features (normalized times, duration, etc.)
            segment_duration = original_end - original_start
            normalized_start = original_start / video_duration
            normalized_end = original_end / video_duration
            normalized_duration = segment_duration / video_duration
            
            # Create feature vector
            additional_features = torch.tensor([
                normalized_start,
                normalized_end,
                normalized_duration,
                np.log(segment_duration + 1),  # Log duration
                np.log(video_duration + 1)     # Log total duration
            ], device=self.device, dtype=torch.float32).unsqueeze(0)
            
            # Concatenate query embedding with additional features
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
            
            features = torch.cat([query_embedding, additional_features], dim=1)
            
            # Predict boundaries
            prediction = self.model(features)
            
            return prediction
    
    async def batch_refine_boundaries(
        self,
        queries: List[str],
        segments: List[Dict[str, float]],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Refine boundaries for multiple segments in batch.
        
        Args:
            queries: List of search queries
            segments: List of segment dictionaries with start_time, end_time, video_duration
            confidence_threshold: Minimum confidence for applying refinement
            
        Returns:
            List of refined boundary results
        """
        if len(queries) != len(segments):
            raise ValueError("Number of queries must match number of segments")
        
        # Process each query-segment pair
        tasks = []
        for query, segment in zip(queries, segments):
            task = self.refine_boundaries(
                query,
                segment['start_time'],
                segment['end_time'],
                segment.get('video_duration', 3600),  # Default 1 hour if not provided
                confidence_threshold
            )
            tasks.append(task)
        
        # Execute all refinement tasks concurrently
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def train_model(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train the boundary regression model (for future use).
        
        Args:
            training_data: List of training examples with query, original_boundaries, true_boundaries
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training statistics
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"Training boundary regression model with {len(training_data)} examples")
        
        # This is a placeholder for training implementation
        # In a real scenario, you would implement the full training loop
        
        return {
            "status": "training_not_implemented",
            "message": "Training functionality is placeholder for future implementation",
            "data_size": len(training_data)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the boundary regressor."""
        try:
            if not self.model:
                return {
                    "status": "unhealthy",
                    "error": "Model not loaded"
                }
            
            # Test boundary refinement
            test_result = await self.refine_boundaries(
                query="test query",
                original_start=10.0,
                original_end=20.0,
                video_duration=100.0
            )
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "text_encoder_loaded": self.text_encoder is not None,
                "device": str(self.device),
                "cache_available": self.cache is not None,
                "test_refinement_successful": 'start_time' in test_result
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
