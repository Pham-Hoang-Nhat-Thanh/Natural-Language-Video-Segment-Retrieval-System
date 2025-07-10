"""
Unit tests for the ingestion service.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import cv2

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.SHOT_DETECTOR_MODEL = "TransNetV2"
    settings.SHOT_THRESHOLD = 0.5
    settings.KEYFRAME_INTERVAL = 1.0
    settings.KEYFRAME_QUALITY = 0.95
    settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    settings.FAISS_INDEX_PATH = "./test_index"
    settings.VIDEO_STORAGE_PATH = "./test_videos"
    settings.THUMBNAIL_STORAGE_PATH = "./test_thumbnails"
    settings.DB_HOST = "localhost"
    settings.DB_PORT = 5432
    settings.DB_NAME = "test_db"
    settings.DB_USER = "test_user"
    settings.DB_PASSWORD = "test_password"
    return settings

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestShotDetector:
    """Test cases for ShotDetector class."""
    
    @pytest.fixture
    def shot_detector(self, mock_settings):
        """Create a ShotDetector instance for testing."""
        from shot_detector import ShotDetector
        
        with patch('shot_detector.TransNetV2'):
            detector = ShotDetector(mock_settings)
            yield detector
    
    def test_detect_shots_mock(self, shot_detector):
        """Test shot detection with mock video."""
        # Mock video frames
        mock_frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
        
        # Mock TransNetV2 predictions
        mock_predictions = np.array([0.1, 0.2, 0.8, 0.1, 0.9, 0.1] + [0.1] * 94)
        
        with patch.object(shot_detector.model, 'predict') as mock_predict:
            mock_predict.return_value = mock_predictions
            
            shots = shot_detector.detect_shots(mock_frames, fps=30.0)
            
            # Should detect shots at frames where prediction > threshold (0.5)
            expected_shots = [2, 4]  # Frames with scores 0.8 and 0.9
            assert len(shots) == len(expected_shots)
    
    def test_detect_shots_empty_video(self, shot_detector):
        """Test shot detection with empty video."""
        empty_frames = np.empty((0, 480, 640, 3), dtype=np.uint8)
        
        shots = shot_detector.detect_shots(empty_frames, fps=30.0)
        
        assert shots == []

class TestKeyframeExtractor:
    """Test cases for KeyframeExtractor class."""
    
    @pytest.fixture
    def keyframe_extractor(self, mock_settings, temp_dir):
        """Create a KeyframeExtractor instance for testing."""
        from keyframe_extractor import KeyframeExtractor
        
        mock_settings.THUMBNAIL_STORAGE_PATH = temp_dir
        extractor = KeyframeExtractor(mock_settings)
        yield extractor
    
    @pytest.mark.asyncio
    async def test_extract_keyframes(self, keyframe_extractor, temp_dir):
        """Test keyframe extraction."""
        # Create a mock video file
        video_path = Path(temp_dir) / "test_video.mp4"
        
        # Create mock video frames
        mock_frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock video capture
            mock_cap_instance = Mock()
            mock_cap.return_value = mock_cap_instance
            
            # Mock video properties
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 300.0
            }.get(prop, 0)
            
            # Mock frame reading
            frame_iter = iter([(True, frame) for frame in mock_frames] + [(False, None)])
            mock_cap_instance.read.side_effect = lambda: next(frame_iter)
            
            shot_boundaries = [
                {"start_frame": 0, "end_frame": 100, "start_time": 0.0, "end_time": 3.33},
                {"start_frame": 100, "end_frame": 200, "start_time": 3.33, "end_time": 6.67},
                {"start_frame": 200, "end_frame": 300, "start_time": 6.67, "end_time": 10.0}
            ]
            
            with patch('cv2.imwrite') as mock_imwrite:
                mock_imwrite.return_value = True
                
                keyframes = await keyframe_extractor.extract_keyframes(
                    str(video_path), shot_boundaries
                )
                
                assert len(keyframes) == len(shot_boundaries)
                assert all('keyframe_path' in kf for kf in keyframes)
                assert all('timestamp' in kf for kf in keyframes)
    
    def test_select_best_keyframe(self, keyframe_extractor):
        """Test best keyframe selection."""
        # Create mock frames with different qualities
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        
        # Mock quality scores (variance-based)
        with patch.object(keyframe_extractor, '_calculate_frame_quality') as mock_quality:
            mock_quality.side_effect = [0.2, 0.8, 0.5, 0.9, 0.3]
            
            best_frame, best_idx = keyframe_extractor._select_best_keyframe(frames)
            
            assert best_idx == 3  # Frame with highest quality (0.9)
            assert np.array_equal(best_frame, frames[3])

class TestEmbeddingService:
    """Test cases for EmbeddingService class."""
    
    @pytest.fixture
    async def embedding_service(self, mock_settings):
        """Create an EmbeddingService instance for testing."""
        from embedding_service import EmbeddingService
        
        # Mock Redis connection
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Redis not available")
            
            service = EmbeddingService(mock_settings)
            yield service
    
    @pytest.mark.asyncio
    async def test_encode_text(self, embedding_service):
        """Test text encoding."""
        test_text = "This is a test text for encoding"
        
        embedding = await embedding_service.encode_text(test_text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    @pytest.mark.asyncio
    async def test_encode_image(self, embedding_service):
        """Test image encoding."""
        # Create a mock image
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        embedding = await embedding_service.encode_image(mock_image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    @pytest.mark.asyncio
    async def test_create_multimodal_embedding(self, embedding_service):
        """Test multimodal embedding creation."""
        test_text = "A cat playing with a toy"
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        embedding = await embedding_service.create_multimodal_embedding(
            text=test_text,
            image=mock_image
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1

class TestIngestionDatabase:
    """Test cases for IngestionDatabase class."""
    
    @pytest.fixture
    async def mock_db(self, mock_settings):
        """Create a mock IngestionDatabase instance."""
        from database import IngestionDatabase
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.return_value = mock_conn
            
            db = IngestionDatabase(mock_settings)
            db.pool = mock_conn
            
            yield db
    
    @pytest.mark.asyncio
    async def test_store_video_metadata(self, mock_db):
        """Test storing video metadata."""
        mock_db.pool.acquire.return_value.__aenter__.return_value.fetchval.return_value = 1
        
        video_data = {
            "video_id": "test_video_123",
            "title": "Test Video",
            "description": "A test video for unit testing",
            "duration": 120.5,
            "file_path": "/path/to/video.mp4"
        }
        
        video_id = await mock_db.store_video_metadata(video_data)
        
        assert video_id == 1
    
    @pytest.mark.asyncio
    async def test_store_video_segments(self, mock_db):
        """Test storing video segments."""
        segments = [
            {
                "video_id": "test_video",
                "segment_id": "seg_1",
                "start_time": 0.0,
                "end_time": 10.0,
                "keyframe_path": "/path/to/keyframe1.jpg"
            },
            {
                "video_id": "test_video", 
                "segment_id": "seg_2",
                "start_time": 10.0,
                "end_time": 20.0,
                "keyframe_path": "/path/to/keyframe2.jpg"
            }
        ]
        
        await mock_db.store_video_segments(segments)
        
        # Verify executemany was called
        mock_db.pool.acquire.return_value.__aenter__.return_value.executemany.assert_called_once()

@pytest.mark.asyncio
async def test_ingestion_pipeline_integration():
    """Integration test for the full ingestion pipeline."""
    # This would test the complete ingestion workflow
    # For now, it's a placeholder for future implementation
    pass

# Utility functions for testing
def create_mock_video_file(path: Path, duration: float = 10.0, fps: float = 30.0):
    """Create a mock video file for testing."""
    # This would create an actual video file for integration tests
    pass

def create_test_frames(num_frames: int, height: int = 480, width: int = 640) -> np.ndarray:
    """Create test video frames."""
    return np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
