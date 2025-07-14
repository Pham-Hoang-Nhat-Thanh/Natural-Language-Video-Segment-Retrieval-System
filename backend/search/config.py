"""
Configuration settings for the search service.
"""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Configuration settings for the search service."""
    
    # Service settings
    SERVICE_NAME: str = "search-service"
    SERVICE_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False
    
    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "video_retrieval"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Environment variable support (for Docker Compose)
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    MODEL_PATH: Optional[str] = None
    DATA_PATH: Optional[str] = None
    FAISS_INDEX_PATH: Optional[str] = None
    ENABLE_GPU: bool = False
    BATCH_SIZE: int = 32
    CACHE_TTL: int = 3600
    
    # Model settings
    TEXT_ENCODER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODELS_PATH: str = "./models"
    
    # FAISS settings
    INDEX_PATH: str = "./faiss_index"
    FAISS_INDEX_TYPE: str = "flat"  # flat, ivf, hnsw, pq
    FAISS_NPROBE: int = 32
    
    # Search settings
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    SIMILARITY_THRESHOLD: float = 0.3
    RERANKING_ENABLED: bool = True
    BOUNDARY_REGRESSION_ENABLED: bool = True
    BOUNDARY_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    EMBEDDING_CACHE_TTL: int = 3600  # 1 hour
    SEARCH_CACHE_TTL: int = 300      # 5 minutes
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API settings
    API_TITLE: str = "Video Search Service"
    API_DESCRIPTION: str = "Natural language video segment search and retrieval"
    API_VERSION: str = "1.0.0"
    
    # Monitoring settings
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 8003
    HEALTH_CHECK_INTERVAL: int = 60
    
    class Config:
        case_sensitive = True
        extra = "ignore"  # Allow extra fields to be ignored
