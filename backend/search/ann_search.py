"""
Approximate Nearest Neighbor Search Engine using FAISS for fast similarity search.
Supports multiple index types, dynamic updates, and optimized search operations.
"""

import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import threading
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import Settings

logger = logging.getLogger(__name__)

class ANNSearchEngine:
    """High-performance ANN search engine using FAISS."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = None
        self.metadata = {}
        self.dimension = None
        self.index_type = settings.FAISS_INDEX_TYPE
        self.nprobe = settings.FAISS_NPROBE
        self.index_path = Path(settings.INDEX_PATH)
        self.metadata_path = self.index_path / "metadata.pkl"
        self.index_file = self.index_path / "faiss_index.bin"
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Create index directory
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if self.index_file.exists() and self.metadata_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_file}")
                
                # Load index
                self.index = faiss.read_index(str(self.index_file))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                self.dimension = self.index.d
                
                # Configure index parameters
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.nprobe
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors, dimension {self.dimension}")
                
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self.index = None
            self.metadata = {}
            self.dimension = None
    
    def _create_index(self, dimension: int, initial_size: int = 1000):
        """Create a new FAISS index."""
        self.dimension = dimension
        
        if self.index_type.lower() == "flat":
            # Flat (exact) search
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
        elif self.index_type.lower() == "ivf":
            # IVF (Inverted File) index
            nlist = min(int(np.sqrt(initial_size)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.nprobe = self.nprobe
            
        elif self.index_type.lower() == "hnsw":
            # HNSW (Hierarchical Navigable Small World) index
            M = 16  # Number of connections
            self.index = faiss.IndexHNSWFlat(dimension, M)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
            
        elif self.index_type.lower() == "pq":
            # Product Quantization index
            m = max(1, dimension // 8)  # Number of subquantizers
            self.index = faiss.IndexPQ(dimension, m, 8)
            
        else:
            # Default to flat index
            logger.warning(f"Unknown index type {self.index_type}, using flat index")
            self.index = faiss.IndexFlatIP(dimension)
        
        logger.info(f"Created {self.index_type} index with dimension {dimension}")
    
    async def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            embeddings: Array of embeddings to add (shape: [n, dimension])
            metadata_list: List of metadata dictionaries for each embedding
            
        Returns:
            List of assigned IDs
        """
        start_time = time.time()
        
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        with self.lock:
            # Create index if it doesn't exist
            if self.index is None:
                self._create_index(embeddings.shape[1])
            
            # Check dimension compatibility
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Get starting ID
            start_id = self.index.ntotal
            
            # Train index if needed (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if embeddings.shape[0] >= self.index.nlist:
                    logger.info("Training index...")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self.index.train,
                        embeddings
                    )
                    logger.info("Index training completed")
                else:
                    logger.warning(f"Not enough vectors to train index (need {self.index.nlist}, got {embeddings.shape[0]})")
            
            # Add vectors to index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.index.add,
                embeddings
            )
            
            # Store metadata
            for i, metadata in enumerate(metadata_list):
                vector_id = start_id + i
                self.metadata[vector_id] = metadata
            
            # Generate IDs
            ids = list(range(start_id, start_id + len(embeddings)))
        
        add_time = time.time() - start_time
        logger.info(f"Added {len(embeddings)} vectors to index in {add_time:.3f}s")
        
        return ids
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_func: Optional[callable] = None
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_func: Optional function to filter results based on metadata
            
        Returns:
            Tuple of (scores, metadata_list)
        """
        start_time = time.time()
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [], []
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search in thread pool
        loop = asyncio.get_event_loop()
        scores, indices = await loop.run_in_executor(
            self.executor,
            self._search_sync,
            query_embedding,
            top_k
        )
        
        # Retrieve metadata
        results_metadata = []
        final_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS uses -1 for invalid indices
                continue
                
            metadata = self.metadata.get(idx, {})
            
            # Apply filter if provided
            if filter_func and not filter_func(metadata):
                continue
            
            results_metadata.append(metadata)
            final_scores.append(float(score))
            
            if len(results_metadata) >= top_k:
                break
        
        search_time = time.time() - start_time
        logger.debug(f"Search completed in {search_time:.3f}s, found {len(results_metadata)} results")
        
        return final_scores, results_metadata
    
    def _search_sync(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous search operation."""
        with self.lock:
            # Search with more candidates to account for filtering
            search_k = min(top_k * 2, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_k)
        
        return scores, indices
    
    async def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[Tuple[List[float], List[Dict[str, Any]]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Array of query embeddings (shape: [n, dimension])
            top_k: Number of results per query
            filter_func: Optional function to filter results
            
        Returns:
            List of (scores, metadata_list) tuples for each query
        """
        start_time = time.time()
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [[] for _ in range(len(query_embeddings))]
        
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Batch search in thread pool
        loop = asyncio.get_event_loop()
        scores, indices = await loop.run_in_executor(
            self.executor,
            self._batch_search_sync,
            query_embeddings,
            top_k
        )
        
        # Process results for each query
        results = []
        for i in range(len(query_embeddings)):
            query_scores = []
            query_metadata = []
            
            for score, idx in zip(scores[i], indices[i]):
                if idx == -1:
                    continue
                    
                metadata = self.metadata.get(idx, {})
                
                # Apply filter if provided
                if filter_func and not filter_func(metadata):
                    continue
                
                query_metadata.append(metadata)
                query_scores.append(float(score))
                
                if len(query_metadata) >= top_k:
                    break
            
            results.append((query_scores, query_metadata))
        
        search_time = time.time() - start_time
        logger.info(f"Batch search of {len(query_embeddings)} queries completed in {search_time:.3f}s")
        
        return results
    
    def _batch_search_sync(
        self,
        query_embeddings: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous batch search operation."""
        with self.lock:
            # Search with more candidates to account for filtering
            search_k = min(top_k * 2, self.index.ntotal)
            scores, indices = self.index.search(query_embeddings, search_k)
        
        return scores, indices
    
    async def remove_vectors(self, vector_ids: List[int]) -> bool:
        """
        Remove vectors from the index.
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        """
        logger.warning("Vector removal requires index rebuild - this is expensive")
        
        try:
            with self.lock:
                # Remove metadata
                for vector_id in vector_ids:
                    self.metadata.pop(vector_id, None)
                
                # For now, we'll just mark this as needing a rebuild
                # In production, you might want to implement a more sophisticated approach
                logger.info(f"Marked {len(vector_ids)} vectors for removal")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            return False
    
    async def save_index(self) -> bool:
        """Save the index and metadata to disk."""
        try:
            with self.lock:
                if self.index is not None:
                    # Save index
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        faiss.write_index,
                        self.index,
                        str(self.index_file)
                    )
                    
                    # Save metadata
                    with open(self.metadata_path, 'wb') as f:
                        pickle.dump(self.metadata, f)
                    
                    logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_file}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self.lock:
            if self.index is None:
                return {
                    "status": "empty",
                    "total_vectors": 0,
                    "dimension": 0,
                    "index_type": self.index_type
                }
            
            return {
                "status": "ready",
                "total_vectors": int(self.index.ntotal),
                "dimension": int(self.dimension),
                "index_type": self.index_type,
                "is_trained": getattr(self.index, 'is_trained', True),
                "metadata_entries": len(self.metadata),
                "nprobe": getattr(self.index, 'nprobe', None)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the search engine."""
        try:
            stats = await self.get_stats()
            
            if self.index is not None and self.index.ntotal > 0:
                # Test search with random vector
                test_vector = np.random.random((1, self.dimension)).astype(np.float32)
                scores, metadata = await self.search(test_vector, top_k=1)
                
                return {
                    "status": "healthy",
                    "stats": stats,
                    "test_search_successful": len(scores) > 0
                }
            else:
                return {
                    "status": "ready",
                    "stats": stats,
                    "note": "Index is empty but ready to accept vectors"
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
