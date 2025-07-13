from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List, Tuple
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    log_level: str = "INFO"
    
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5432/video_retrieval"
    redis_url: str = "redis://localhost:6379"
    
    # Model settings
    model_path: str = "models"
    clip_model_name: str = "ViT-B/32"
    use_onnx: bool = True
    device: str = "auto"
    batch_size: int = 32
    embedding_dimension: int = 512
    
    # Data paths
    data_path: str = "data"
    video_storage_path: str = "data/videos"
    thumbnail_storage_path: str = "data/thumbnails"
    
    # Shot detection settings
    hist_threshold: float = 0.3
    ecr_threshold: float = 0.4
    min_shot_length: int = 30
    
    # Keyframe extraction settings
    keyframe_size: Tuple[int, int] = (224, 224)
    keyframe_quality: int = 95
    frames_per_shot: int = 1
    
    # Processing settings
    max_video_size_mb: int = 1000
    max_video_duration_minutes: int = 120
    supported_formats: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    
    # Performance settings
    max_workers: int = 4
    processing_timeout_seconds: int = 3600  # 1 hour
    
    # Storage settings
    use_s3: bool = False
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: str = "us-east-1"
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9001
    
    class Config:
        case_sensitive = False

# Global settings instance
settings = Settings()
