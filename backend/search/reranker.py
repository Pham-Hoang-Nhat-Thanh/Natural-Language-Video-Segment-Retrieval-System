"""
Cross-Encoder Reranker for improving search result quality.
Uses a cross-encoder model to rerank initial search results based on query-segment relevance.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle
import hashlib
from config import Settings

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-encoder based reranking for improving search relevance."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Redis cache if available
        try:
            self.cache = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False,
                socket_timeout=1.0
            )
            self.cache.ping()
            logger.info("Connected to Redis cache for reranker")
        except Exception as e:
            logger.warning(f"Redis cache not available for reranker: {e}")
            self.cache = None
        
        # Load reranker model
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder reranker model."""
        try:
            model_name = self.settings.RERANKER_MODEL
            logger.info(f"Loading reranker model: {model_name}")
            
            # Use sentence-transformers CrossEncoder for easier handling
            self.model = CrossEncoder(model_name, device=self.device)
            self.model.model.eval()
            
            logger.info(f"Loaded reranker model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            # Try fallback model
            try:
                fallback_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.info(f"Trying fallback reranker model: {fallback_model}")
                self.model = CrossEncoder(fallback_model, device=self.device)
                self.model.model.eval()
                logger.info("Loaded fallback reranker model")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback reranker model: {fallback_error}")
                self.model = None
    
    def _get_cache_key(self, query: str, text: str) -> str:
        """Generate cache key for query-text pair."""
        pair_text = f"{query}[SEP]{text}"
        text_hash = hashlib.md5(pair_text.encode()).hexdigest()
        return f"rerank:{text_hash}"
    
    def _get_cached_score(self, cache_key: str) -> Optional[float]:
        """Retrieve cached reranking score."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Reranker cache retrieval error: {e}")
        
        return None
    
    def _cache_score(self, cache_key: str, score: float, ttl: int = 3600):
        """Cache reranking score with TTL."""
        if not self.cache:
            return
        
        try:
            self.cache.setex(
                cache_key,
                ttl,
                pickle.dumps(score)
            )
        except Exception as e:
            logger.warning(f"Reranker cache storage error: {e}")
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate results with 'text' field
            top_k: Number of top results to return (None = return all)
            
        Returns:
            Reranked list of candidates with updated scores
        """
        start_time = time.time()
        
        if not self.model:
            logger.warning("Reranker model not available, returning original order")
            return candidates[:top_k] if top_k else candidates
        
        if not candidates:
            return []
        
        # Extract texts and check cache
        query_text_pairs = []
        cache_keys = []
        cached_scores = {}
        
        for i, candidate in enumerate(candidates):
            # Get text content for reranking
            text = self._extract_text_for_reranking(candidate)
            if not text:
                continue
                
            cache_key = self._get_cache_key(query, text)
            cache_keys.append(cache_key)
            
            # Check cache
            cached_score = self._get_cached_score(cache_key)
            if cached_score is not None:
                cached_scores[i] = cached_score
            else:
                query_text_pairs.append((query, text))
        
        # Compute scores for uncached pairs
        new_scores = []
        if query_text_pairs:
            loop = asyncio.get_event_loop()
            new_scores = await loop.run_in_executor(
                self.executor,
                self._compute_scores_sync,
                query_text_pairs
            )
            
            # Cache new scores
            if self.cache and len(new_scores) == len(query_text_pairs):
                try:
                    cache_data = {}
                    pair_idx = 0
                    for i, candidate in enumerate(candidates):
                        if i not in cached_scores:
                            cache_key = cache_keys[i] if i < len(cache_keys) else None
                            if cache_key and pair_idx < len(new_scores):
                                cache_data[cache_key] = pickle.dumps(new_scores[pair_idx])
                                pair_idx += 1
                    
                    if cache_data:
                        self.cache.mset(cache_data)
                        for key in cache_data.keys():
                            self.cache.expire(key, 3600)
                            
                except Exception as e:
                    logger.warning(f"Reranker batch cache storage error: {e}")
        
        # Assign scores to candidates
        reranked_candidates = []
        new_score_idx = 0
        
        for i, candidate in enumerate(candidates):
            candidate_copy = candidate.copy()
            
            if i in cached_scores:
                rerank_score = cached_scores[i]
            elif new_score_idx < len(new_scores):
                rerank_score = new_scores[new_score_idx]
                new_score_idx += 1
            else:
                # Fallback to original score
                rerank_score = candidate.get('score', 0.0)
            
            candidate_copy['rerank_score'] = float(rerank_score)
            candidate_copy['original_score'] = candidate.get('score', 0.0)
            reranked_candidates.append(candidate_copy)
        
        # Sort by rerank score (descending)
        reranked_candidates.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        
        # Return top_k results
        if top_k:
            reranked_candidates = reranked_candidates[:top_k]
        
        rerank_time = time.time() - start_time
        logger.info(f"Reranked {len(candidates)} candidates in {rerank_time:.3f}s")
        
        return reranked_candidates
    
    def _extract_text_for_reranking(self, candidate: Dict[str, Any]) -> str:
        """Extract text content from candidate for reranking."""
        # Try different fields that might contain text
        text_fields = ['text', 'description', 'content', 'title', 'transcript']
        
        for field in text_fields:
            if field in candidate and candidate[field]:
                return str(candidate[field])
        
        # Fallback: concatenate available text fields
        text_parts = []
        for key, value in candidate.items():
            if isinstance(value, str) and len(value) > 10:  # Skip short strings
                text_parts.append(value)
        
        return " ".join(text_parts) if text_parts else ""
    
    def _compute_scores_sync(self, query_text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Synchronously compute reranking scores."""
        if not query_text_pairs:
            return []
        
        try:
            with torch.no_grad():
                scores = self.model.predict(query_text_pairs)
                
                # Convert to list and handle different output formats
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                elif torch.is_tensor(scores):
                    scores = scores.cpu().numpy().tolist()
                
                # Ensure we have a list of floats
                if not isinstance(scores, list):
                    scores = [float(scores)]
                else:
                    scores = [float(s) for s in scores]
                
                return scores
                
        except Exception as e:
            logger.error(f"Error computing reranking scores: {e}")
            # Return neutral scores
            return [0.5] * len(query_text_pairs)
    
    async def score_query_text_pair(
        self,
        query: str,
        text: str
    ) -> float:
        """
        Score a single query-text pair.
        
        Args:
            query: Search query
            text: Text to score against the query
            
        Returns:
            Relevance score
        """
        if not self.model:
            return 0.5  # Neutral score
        
        # Check cache
        cache_key = self._get_cache_key(query, text)
        cached_score = self._get_cached_score(cache_key)
        if cached_score is not None:
            return cached_score
        
        # Compute score
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            self.executor,
            self._compute_scores_sync,
            [(query, text)]
        )
        
        score = scores[0] if scores else 0.5
        
        # Cache the result
        self._cache_score(cache_key, score)
        
        return score
    
    async def batch_rerank(
        self,
        queries: List[str],
        candidate_lists: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple queries in batch.
        
        Args:
            queries: List of search queries
            candidate_lists: List of candidate lists for each query
            top_k: Number of top results per query
            
        Returns:
            List of reranked candidate lists
        """
        if len(queries) != len(candidate_lists):
            raise ValueError("Number of queries must match number of candidate lists")
        
        # Process each query-candidate pair
        tasks = []
        for query, candidates in zip(queries, candidate_lists):
            task = self.rerank(query, candidates, top_k)
            tasks.append(task)
        
        # Execute all reranking tasks concurrently
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the reranker."""
        try:
            if not self.model:
                return {
                    "status": "unhealthy",
                    "error": "Model not loaded"
                }
            
            # Test reranking
            test_query = "test query"
            test_candidates = [
                {"text": "This is a relevant test document", "score": 0.8},
                {"text": "This is an irrelevant document", "score": 0.6}
            ]
            
            reranked = await self.rerank(test_query, test_candidates)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": str(self.device),
                "cache_available": self.cache is not None,
                "test_rerank_successful": len(reranked) == 2
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
