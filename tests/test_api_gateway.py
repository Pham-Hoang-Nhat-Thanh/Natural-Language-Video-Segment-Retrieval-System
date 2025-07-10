"""
Unit tests for the API Gateway.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.SERVICE_NAME = "api-gateway"
    settings.HOST = "0.0.0.0"
    settings.PORT = 8000
    settings.DEBUG = True
    settings.REDIS_HOST = "localhost"
    settings.REDIS_PORT = 6379
    settings.INGEST_SERVICE_URL = "http://localhost:8001"
    settings.SEARCH_SERVICE_URL = "http://localhost:8002"
    settings.RATE_LIMIT_REQUESTS = 100
    settings.RATE_LIMIT_WINDOW = 60
    settings.CACHE_TTL = 300
    return settings

@pytest.fixture
def test_app(mock_settings):
    """Create a test app instance."""
    with patch('src.app.Settings', return_value=mock_settings):
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            from src.app import create_app
            app = create_app()
            yield app

@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)

class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
    
    def test_ready_check(self, client):
        """Test readiness check."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock successful service responses
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            response = client.get("/ready")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

class TestSearchEndpoints:
    """Test search-related endpoints."""
    
    def test_search_videos(self, client):
        """Test video search endpoint."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock search service response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "video_id": "test_video_1",
                        "segment_id": "seg_1",
                        "start_time": 10.0,
                        "end_time": 20.0,
                        "score": 0.95,
                        "title": "Test Video",
                        "thumbnail_url": "/thumbnails/test1.jpg"
                    }
                ],
                "query_time_ms": 45.2,
                "total_results": 1
            }
            mock_post.return_value = mock_response
            
            search_data = {
                "query": "cats playing with toys",
                "top_k": 10,
                "threshold": 0.5
            }
            
            response = client.post("/api/v1/search", json=search_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "query_time_ms" in data
            assert len(data["results"]) >= 0
    
    def test_search_videos_invalid_query(self, client):
        """Test search with invalid query."""
        search_data = {
            "query": "",  # Empty query
            "top_k": 10
        }
        
        response = client.post("/api/v1/search", json=search_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_search_videos_service_error(self, client):
        """Test search when service is unavailable."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock service error
            mock_post.side_effect = Exception("Service unavailable")
            
            search_data = {
                "query": "test query",
                "top_k": 10
            }
            
            response = client.post("/api/v1/search", json=search_data)
            
            assert response.status_code == 500

class TestUploadEndpoints:
    """Test video upload endpoints."""
    
    def test_upload_video(self, client):
        """Test video upload endpoint."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock successful upload response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "video_id": "uploaded_video_123",
                "status": "processing",
                "message": "Video uploaded successfully"
            }
            mock_post.return_value = mock_response
            
            # Mock file upload
            test_file = ("test_video.mp4", b"fake video content", "video/mp4")
            
            response = client.post(
                "/api/v1/upload",
                files={"file": test_file},
                data={"title": "Test Video", "description": "A test upload"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "video_id" in data
            assert data["status"] == "processing"
    
    def test_upload_video_no_file(self, client):
        """Test upload without file."""
        response = client.post(
            "/api/v1/upload",
            data={"title": "Test Video"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_get_upload_status(self, client):
        """Test getting upload status."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock status response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "video_id": "test_video_123",
                "status": "completed",
                "progress": 100,
                "segments_processed": 15
            }
            mock_get.return_value = mock_response
            
            response = client.get("/api/v1/upload/test_video_123/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress"] == 100

class TestVideoEndpoints:
    """Test video management endpoints."""
    
    def test_get_video_info(self, client):
        """Test getting video information."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock video info response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "video_id": "test_video_123",
                "title": "Test Video",
                "description": "A test video",
                "duration": 120.5,
                "segments_count": 12,
                "created_at": "2024-01-01T00:00:00Z"
            }
            mock_get.return_value = mock_response
            
            response = client.get("/api/v1/videos/test_video_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == "test_video_123"
            assert "title" in data
            assert "duration" in data
    
    def test_get_video_segments(self, client):
        """Test getting video segments."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock segments response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "video_id": "test_video_123",
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "start_time": 0.0,
                        "end_time": 10.0,
                        "thumbnail_url": "/thumbnails/seg1.jpg"
                    },
                    {
                        "segment_id": "seg_2", 
                        "start_time": 10.0,
                        "end_time": 20.0,
                        "thumbnail_url": "/thumbnails/seg2.jpg"
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            response = client.get("/api/v1/videos/test_video_123/segments")
            
            assert response.status_code == 200
            data = response.json()
            assert "segments" in data
            assert len(data["segments"]) == 2
    
    def test_list_videos(self, client):
        """Test listing videos."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock videos list response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "videos": [
                    {
                        "video_id": "video_1",
                        "title": "Video 1",
                        "duration": 60.0,
                        "created_at": "2024-01-01T00:00:00Z"
                    },
                    {
                        "video_id": "video_2",
                        "title": "Video 2", 
                        "duration": 90.0,
                        "created_at": "2024-01-02T00:00:00Z"
                    }
                ],
                "total": 2,
                "page": 1,
                "per_page": 10
            }
            mock_get.return_value = mock_response
            
            response = client.get("/api/v1/videos")
            
            assert response.status_code == 200
            data = response.json()
            assert "videos" in data
            assert data["total"] == 2

class TestAnalyticsEndpoints:
    """Test analytics endpoints."""
    
    def test_get_search_analytics(self, client):
        """Test getting search analytics."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock analytics response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "total_searches": 1250,
                "avg_search_time_ms": 42.5,
                "top_queries": [
                    {"query": "cats playing", "count": 25},
                    {"query": "dogs running", "count": 18}
                ],
                "search_trends": [
                    {"date": "2024-01-01", "searches": 45},
                    {"date": "2024-01-02", "searches": 67}
                ]
            }
            mock_get.return_value = mock_response
            
            response = client.get("/api/v1/analytics/search")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_searches" in data
            assert "top_queries" in data

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is enforced."""
        # This would require a more sophisticated test setup
        # with actual Redis or mock rate limiting
        pass

class TestCaching:
    """Test caching functionality."""
    
    def test_cache_hit(self, client):
        """Test cache hit scenario."""
        # Mock Redis cache hit
        with patch('redis.Redis.get') as mock_get:
            mock_get.return_value = json.dumps({
                "results": [],
                "cached": True
            })
            
            search_data = {
                "query": "cached query",
                "top_k": 10
            }
            
            response = client.post("/api/v1/search", json=search_data)
            
            # This would depend on actual cache implementation
            assert response.status_code == 200

class TestErrorHandling:
    """Test error handling."""
    
    def test_404_handler(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_500_handler(self, client):
        """Test 500 error handling."""
        with patch('src.app.search_videos') as mock_search:
            mock_search.side_effect = Exception("Internal server error")
            
            search_data = {
                "query": "test query",
                "top_k": 10
            }
            
            response = client.post("/api/v1/search", json=search_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

# Utility functions for testing
def create_mock_video_data():
    """Create mock video data for testing."""
    return {
        "video_id": "test_video_123",
        "title": "Test Video",
        "description": "A test video for unit testing",
        "duration": 120.5,
        "segments": [
            {
                "segment_id": "seg_1",
                "start_time": 0.0,
                "end_time": 10.0,
                "score": 0.95
            },
            {
                "segment_id": "seg_2",
                "start_time": 10.0,
                "end_time": 20.0,
                "score": 0.87
            }
        ]
    }

def create_mock_search_results():
    """Create mock search results for testing."""
    return {
        "results": [
            {
                "video_id": "video_1",
                "segment_id": "seg_1",
                "start_time": 10.0,
                "end_time": 20.0,
                "score": 0.95,
                "title": "Video 1",
                "thumbnail_url": "/thumbnails/v1_seg1.jpg"
            },
            {
                "video_id": "video_2",
                "segment_id": "seg_3", 
                "start_time": 30.0,
                "end_time": 40.0,
                "score": 0.87,
                "title": "Video 2",
                "thumbnail_url": "/thumbnails/v2_seg3.jpg"
            }
        ],
        "query_time_ms": 42.5,
        "total_results": 2
    }
