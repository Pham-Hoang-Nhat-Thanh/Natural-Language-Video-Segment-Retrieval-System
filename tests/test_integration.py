import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'ingest'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'search'))

from backend.ingest.main import app as ingest_app
from backend.search.main import app as search_app

class TestIngestService:
    """Test cases for the ingestion service"""
    
    @pytest.fixture
    def client(self):
        """Create test client for ingestion service"""
        return TestClient(ingest_app)
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests"""
        temp_dir = tempfile.mkdtemp()
        # Create necessary subdirectories
        videos_dir = Path(temp_dir) / "videos" / "datasets" / "custom"
        videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy MP4 file for testing
        dummy_video = videos_dir / "test_video.mp4"
        dummy_video.write_bytes(b"fake video content")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "video-processing"
    
    def test_list_videos(self, client, temp_data_dir):
        """Test video listing endpoint"""
        # Mock the settings data_path
        import backend.ingest.main
        original_data_path = backend.ingest.main.settings.data_path
        backend.ingest.main.settings.data_path = temp_data_dir
        
        try:
            response = client.get("/api/videos")
            assert response.status_code == 200
            data = response.json()
            assert "videos" in data
            assert len(data["videos"]) == 1
            assert data["videos"][0]["video_id"] == "test_video"
            assert data["videos"][0]["filename"] == "test_video.mp4"
        finally:
            backend.ingest.main.settings.data_path = original_data_path
    
    def test_get_video_status_not_found(self, client):
        """Test status endpoint for non-existent video"""
        response = client.get("/api/videos/nonexistent/status")
        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "nonexistent"
        assert data["processed"] == False
    
    def test_get_service_stats(self, client):
        """Test service statistics endpoint"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_mp4_files" in data
        assert "processed_videos" in data
        assert "model_info" in data

class TestSearchService:
    """Test cases for the search service"""
    
    @pytest.fixture
    def client(self):
        """Create test client for search service"""
        return TestClient(search_app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
    
    def test_text_embedding(self, client):
        """Test text embedding endpoint"""
        response = client.post("/api/embed/text", json={"text": "a person walking"})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0
    
    def test_search_empty_query(self, client):
        """Test search with empty query"""
        response = client.post("/api/search", json={"query": ""})
        # Should handle empty query gracefully
        assert response.status_code in [200, 400]
    
    def test_search_valid_query(self, client):
        """Test search with valid query"""
        response = client.post("/api/search", json={
            "query": "a person walking",
            "top_k": 5,
            "threshold": 0.3
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query_time_ms" in data
        assert "total_results" in data
        assert isinstance(data["results"], list)
    
    def test_rerank_endpoint(self, client):
        """Test reranking endpoint"""
        candidates = [
            {"video_id": "test1", "start_time": 0, "end_time": 10, "score": 0.8},
            {"video_id": "test2", "start_time": 5, "end_time": 15, "score": 0.7}
        ]
        response = client.post("/api/rerank", json={
            "query": "test query",
            "candidates": candidates
        })
        assert response.status_code == 200
        data = response.json()
        assert "reranked_results" in data
    
    def test_get_service_stats(self, client):
        """Test service statistics endpoint"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_searches" in data
        assert "model_info" in data

class TestAPIGateway:
    """Test cases for the API Gateway"""
    
    @pytest.fixture
    def mock_services(self, monkeypatch):
        """Mock the backend services for testing"""
        import httpx
        
        class MockResponse:
            def __init__(self, json_data, status_code=200):
                self.json_data = json_data
                self.status_code = status_code
                self.data = json_data
            
            def json(self):
                return self.json_data
        
        async def mock_get(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if '/health' in url:
                return MockResponse({"status": "healthy"})
            elif '/api/videos' in url:
                return MockResponse({"videos": []})
            elif '/api/stats' in url:
                return MockResponse({"total_searches": 0})
            return MockResponse({})
        
        async def mock_post(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if '/api/search' in url:
                return MockResponse({
                    "results": [],
                    "query_time_ms": 100,
                    "total_results": 0
                })
            elif '/api/process' in url:
                return MockResponse({"status": "success", "video_id": "test"})
            return MockResponse({})
        
        async def mock_delete(*args, **kwargs):
            return MockResponse({"status": "deleted"})
        
        monkeypatch.setattr("axios.get", mock_get)
        monkeypatch.setattr("axios.post", mock_post)
        monkeypatch.setattr("axios.delete", mock_delete)
    
    def test_health_endpoint_integration(self):
        """Test health endpoint with service integration"""
        # This would require running actual services
        # For now, test the endpoint structure
        pass
    
    def test_search_endpoint_integration(self):
        """Test search endpoint integration"""
        # This would test the full search pipeline
        pass

class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_video_processing_workflow(self):
        """Test complete video processing workflow"""
        # 1. Upload video
        # 2. Process video
        # 3. Check status
        # 4. Search segments
        # 5. Clean up
        pass
    
    @pytest.mark.asyncio
    async def test_search_workflow(self):
        """Test complete search workflow"""
        # 1. Process sample video
        # 2. Search for segments
        # 3. Verify results
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
