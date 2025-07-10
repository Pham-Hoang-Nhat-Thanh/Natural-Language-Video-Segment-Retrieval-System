"""
Unit tests for the search service.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "TEXT_ENCODER_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": 6379,
    "DB_HOST": "localhost",
    "DB_PORT": 5432,
    "DB_NAME": "test_video_retrieval",
    "DB_USER": "test_user",
    "DB_PASSWORD": "test_password",
    "INDEX_PATH": "./test_faiss_index",
    "MODELS_PATH": "./test_models",
    "FAISS_INDEX_TYPE": "flat",
    "FAISS_NPROBE": 32
}

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from config import Settings
    
    # Create a mock settings object
    settings = Mock(spec=Settings)
    for key, value in TEST_CONFIG.items():
        setattr(settings, key, value)
    
    return settings

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestTextEncoder:
    """Test cases for TextEncoder class."""
    
    @pytest.fixture
    async def text_encoder(self, mock_settings, temp_dir):
        """Create a TextEncoder instance for testing."""
        from text_encoder import TextEncoder
        
        # Mock Redis to avoid connection issues in tests
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            encoder = TextEncoder(mock_settings)
            yield encoder
    
    @pytest.mark.asyncio
    async def test_encode_text(self, text_encoder):
        """Test text encoding functionality."""
        test_text = "This is a test query about cats playing"
        
        embedding = await text_encoder.encode_text(test_text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    @pytest.mark.asyncio
    async def test_encode_batch(self, text_encoder):
        """Test batch text encoding."""
        test_texts = [
            "This is the first test query",
            "This is the second test query",
            "This is the third test query"
        ]
        
        embeddings = await text_encoder.encode_batch(test_texts)
        
        assert len(embeddings) == len(test_texts)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.dtype == np.float32 for emb in embeddings)
        assert all(emb.shape[0] == embeddings[0].shape[0] for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_health_check(self, text_encoder):
        """Test text encoder health check."""
        health = await text_encoder.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        assert "embedding_dim" in health or "error" in health

class TestANNSearchEngine:
    """Test cases for ANNSearchEngine class."""
    
    @pytest.fixture
    async def search_engine(self, mock_settings, temp_dir):
        """Create an ANNSearchEngine instance for testing."""
        from ann_search import ANNSearchEngine
        
        # Set up temporary index path
        mock_settings.INDEX_PATH = temp_dir
        
        engine = ANNSearchEngine(mock_settings)
        yield engine
    
    @pytest.mark.asyncio
    async def test_add_and_search_vectors(self, search_engine):
        """Test adding vectors and searching."""
        # Create test embeddings
        dimension = 384
        num_vectors = 10
        embeddings = np.random.random((num_vectors, dimension)).astype(np.float32)
        
        # Create metadata
        metadata_list = [
            {
                "segment_id": f"seg_{i}",
                "video_id": f"video_{i//3}",
                "start_time": i * 10.0,
                "end_time": (i + 1) * 10.0,
                "text": f"This is segment {i}"
            }
            for i in range(num_vectors)
        ]
        
        # Add vectors
        ids = await search_engine.add_vectors(embeddings, metadata_list)
        assert len(ids) == num_vectors
        
        # Search
        query_embedding = embeddings[0]  # Search for the first vector
        scores, results = await search_engine.search(query_embedding, top_k=5)
        
        assert len(scores) <= 5
        assert len(results) == len(scores)
        assert scores[0] > 0.9  # Should find the exact match with high score
    
    @pytest.mark.asyncio
    async def test_batch_search(self, search_engine):
        """Test batch search functionality."""
        # Add some vectors first
        dimension = 384
        num_vectors = 20
        embeddings = np.random.random((num_vectors, dimension)).astype(np.float32)
        metadata_list = [{"segment_id": f"seg_{i}"} for i in range(num_vectors)]
        
        await search_engine.add_vectors(embeddings, metadata_list)
        
        # Batch search
        query_embeddings = embeddings[:3]  # Search for first 3 vectors
        results = await search_engine.batch_search(query_embeddings, top_k=3)
        
        assert len(results) == 3
        assert all(len(scores) <= 3 and len(metadata) <= 3 for scores, metadata in results)
    
    @pytest.mark.asyncio
    async def test_health_check(self, search_engine):
        """Test search engine health check."""
        health = await search_engine.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "ready", "unhealthy"]

class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker class."""
    
    @pytest.fixture
    async def reranker(self, mock_settings):
        """Create a CrossEncoderReranker instance for testing."""
        from reranker import CrossEncoderReranker
        
        # Mock Redis to avoid connection issues
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            reranker = CrossEncoderReranker(mock_settings)
            yield reranker
    
    @pytest.mark.asyncio
    async def test_rerank_candidates(self, reranker):
        """Test reranking functionality."""
        if not reranker.model:
            pytest.skip("Reranker model not available")
        
        query = "cats playing with toys"
        candidates = [
            {
                "segment_id": "seg_1",
                "text": "cats playing with colorful toys in the garden",
                "score": 0.7
            },
            {
                "segment_id": "seg_2", 
                "text": "dogs running in the park",
                "score": 0.8
            },
            {
                "segment_id": "seg_3",
                "text": "children playing with cats and toys",
                "score": 0.6
            }
        ]
        
        reranked = await reranker.rerank(query, candidates, top_k=3)
        
        assert len(reranked) <= 3
        assert all("rerank_score" in candidate for candidate in reranked)
        # The first candidate should be most relevant to cats + toys
        assert reranked[0]["segment_id"] in ["seg_1", "seg_3"]
    
    @pytest.mark.asyncio
    async def test_score_query_text_pair(self, reranker):
        """Test scoring a single query-text pair."""
        if not reranker.model:
            pytest.skip("Reranker model not available")
        
        query = "cats playing"
        text = "cute cats playing with toys"
        
        score = await reranker.score_query_text_pair(query, text)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

class TestBoundaryRegressor:
    """Test cases for BoundaryRegressor class."""
    
    @pytest.fixture
    async def boundary_regressor(self, mock_settings, temp_dir):
        """Create a BoundaryRegressor instance for testing."""
        from boundary_regressor import BoundaryRegressor
        
        mock_settings.MODELS_PATH = temp_dir
        
        # Mock Redis to avoid connection issues
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            regressor = BoundaryRegressor(mock_settings)
            yield regressor
    
    @pytest.mark.asyncio
    async def test_refine_boundaries(self, boundary_regressor):
        """Test boundary refinement."""
        if not boundary_regressor.model:
            pytest.skip("Boundary regressor model not available")
        
        query = "cats playing with toys"
        original_start = 10.0
        original_end = 20.0
        video_duration = 100.0
        
        result = await boundary_regressor.refine_boundaries(
            query, original_start, original_end, video_duration
        )
        
        assert "start_time" in result
        assert "end_time" in result
        assert "confidence" in result
        assert "refined" in result
        assert result["start_time"] >= 0
        assert result["end_time"] <= video_duration
        assert result["start_time"] < result["end_time"]
    
    @pytest.mark.asyncio
    async def test_batch_refine_boundaries(self, boundary_regressor):
        """Test batch boundary refinement."""
        if not boundary_regressor.model:
            pytest.skip("Boundary regressor model not available")
        
        queries = ["cats playing", "dogs running", "birds flying"]
        segments = [
            {"start_time": 10.0, "end_time": 20.0, "video_duration": 100.0},
            {"start_time": 30.0, "end_time": 40.0, "video_duration": 150.0},
            {"start_time": 50.0, "end_time": 60.0, "video_duration": 200.0}
        ]
        
        results = await boundary_regressor.batch_refine_boundaries(queries, segments)
        
        assert len(results) == len(queries)
        assert all("start_time" in result for result in results)
        assert all("confidence" in result for result in results)

class TestSearchDatabase:
    """Test cases for SearchDatabase class."""
    
    @pytest.fixture
    async def mock_db(self, mock_settings):
        """Create a mock SearchDatabase instance."""
        from database import SearchDatabase
        
        # Mock the connection pool
        with patch('asyncpg.create_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.return_value = mock_conn
            
            db = SearchDatabase(mock_settings)
            db.pool = mock_conn
            
            yield db
    
    @pytest.mark.asyncio
    async def test_log_search_query(self, mock_db):
        """Test logging search queries."""
        mock_db.pool.acquire.return_value.__aenter__.return_value.fetchval.return_value = 123
        
        query_id = await mock_db.log_search_query(
            query_text="test query",
            user_id="user_123",
            session_id="session_456"
        )
        
        assert query_id == 123
    
    @pytest.mark.asyncio
    async def test_get_video_metadata(self, mock_db):
        """Test getting video metadata."""
        mock_row = {
            "video_id": "test_video",
            "title": "Test Video",
            "description": "A test video",
            "duration": 120.0
        }
        
        mock_db.pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = mock_row
        
        metadata = await mock_db.get_video_metadata("test_video")
        
        assert metadata == mock_row

@pytest.mark.asyncio
async def test_search_integration():
    """Integration test for the search pipeline."""
    # This would test the full search pipeline
    # For now, it's a placeholder for future implementation
    pass

# Utility functions for testing
def create_test_embeddings(num_vectors: int, dimension: int = 384) -> np.ndarray:
    """Create test embeddings for testing purposes."""
    return np.random.random((num_vectors, dimension)).astype(np.float32)

def create_test_metadata(num_items: int) -> list:
    """Create test metadata for testing purposes."""
    return [
        {
            "segment_id": f"test_seg_{i}",
            "video_id": f"test_video_{i // 5}",
            "start_time": i * 10.0,
            "end_time": (i + 1) * 10.0,
            "text": f"This is test segment number {i}"
        }
        for i in range(num_items)
    ]
