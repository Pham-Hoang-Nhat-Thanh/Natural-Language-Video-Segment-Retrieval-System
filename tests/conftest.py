"""
Test configuration and fixtures for the video retrieval system.
"""

import pytest
import asyncio
import os
from typing import Generator
import httpx
import psycopg2
import redis

# Test configuration
TEST_CONFIG = {
    "api_gateway_url": "http://localhost:8000",
    "ingest_service_url": "http://localhost:8001", 
    "search_service_url": "http://localhost:8002",
    "database_url": "postgresql://postgres:postgres@localhost:5432/test_video_retrieval",
    "redis_url": "redis://localhost:6379/1"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def http_client():
    """HTTP client for making API requests."""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.fixture(scope="session")
def test_database():
    """Test database connection."""
    conn = psycopg2.connect(TEST_CONFIG["database_url"])
    yield conn
    conn.close()

@pytest.fixture(scope="session")
def test_redis():
    """Test Redis connection."""
    r = redis.from_url(TEST_CONFIG["redis_url"])
    yield r
    r.flushdb()
    r.close()

@pytest.fixture
def sample_video_data():
    """Sample video data for testing."""
    return {
        "video_id": "test_video_1",
        "title": "Test Video",
        "duration": 120.5,
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 30.0,
                "text": "Introduction to machine learning"
            },
            {
                "start_time": 30.0,
                "end_time": 60.0, 
                "text": "Neural networks and deep learning"
            }
        ]
    }

@pytest.fixture
def sample_search_queries():
    """Sample search queries for testing."""
    return [
        "machine learning basics",
        "neural networks",
        "deep learning introduction",
        "AI fundamentals"
    ]
