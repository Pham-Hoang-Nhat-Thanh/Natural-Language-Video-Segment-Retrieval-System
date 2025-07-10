import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
from shot_detector import Shot

logger = logging.getLogger(__name__)

@dataclass
class Keyframe:
    keyframe_id: str
    shot_id: int
    frame_number: int
    timestamp: float
    image_path: str
    image_data: np.ndarray = None

class KeyframeExtractor:
    """
    Keyframe extraction service using middle-frame strategy
    """
    
    def __init__(self, 
                 output_dir: str = "data/thumbnails",
                 image_size: Tuple[int, int] = (224, 224),
                 quality: int = 95):
        """
        Initialize keyframe extractor
        
        Args:
            output_dir: Directory to save keyframe images
            image_size: Size to resize keyframes to
            quality: JPEG quality for saved images
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.quality = quality
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def extract_keyframes(self, 
                              video_path: str, 
                              shots: List[Shot]) -> List[Keyframe]:
        """
        Extract keyframes from video shots
        
        Args:
            video_path: Path to video file
            shots: List of detected shots
            
        Returns:
            List of extracted keyframes
        """
        logger.info(f"Extracting keyframes from {len(shots)} shots")
        
        # Run extraction in thread pool
        loop = asyncio.get_event_loop()
        keyframes = await loop.run_in_executor(
            self.executor,
            self._extract_keyframes_sync,
            video_path,
            shots
        )
        
        logger.info(f"Extracted {len(keyframes)} keyframes")
        return keyframes
    
    def _extract_keyframes_sync(self, 
                               video_path: str, 
                               shots: List[Shot]) -> List[Keyframe]:
        """Synchronous keyframe extraction implementation"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_name = Path(video_path).stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        
        # Create directory for this video's keyframes
        video_dir = self.output_dir / video_name
        video_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {len(shots)} shots for keyframe extraction")
        
        for shot in shots:
            try:
                keyframe = self._extract_shot_keyframe(
                    cap, shot, video_name, video_dir, fps
                )
                if keyframe:
                    keyframes.append(keyframe)
            except Exception as e:
                logger.error(f"Failed to extract keyframe for shot {shot.shot_id}: {e}")
                continue
        
        cap.release()
        return keyframes
    
    def _extract_shot_keyframe(self, 
                              cap: cv2.VideoCapture,
                              shot: Shot,
                              video_name: str,
                              video_dir: Path,
                              fps: float) -> Keyframe:
        """Extract keyframe from a single shot using middle-frame strategy"""
        
        # Calculate middle frame
        middle_frame = (shot.start_frame + shot.end_frame) // 2
        middle_timestamp = middle_frame / fps
        
        # Seek to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame {middle_frame} for shot {shot.shot_id}")
            return None
        
        # Resize frame
        resized_frame = cv2.resize(frame, self.image_size)
        
        # Generate keyframe ID and path
        keyframe_id = f"{video_name}_shot_{shot.shot_id:04d}_frame_{middle_frame:06d}"
        image_filename = f"{keyframe_id}.jpg"
        image_path = video_dir / image_filename
        
        # Save keyframe image
        cv2.imwrite(
            str(image_path), 
            resized_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        
        # Create keyframe object
        keyframe = Keyframe(
            keyframe_id=keyframe_id,
            shot_id=shot.shot_id,
            frame_number=middle_frame,
            timestamp=middle_timestamp,
            image_path=str(image_path),
            image_data=resized_frame
        )
        
        return keyframe
    
    async def extract_multiple_keyframes(self, 
                                       video_path: str, 
                                       shots: List[Shot],
                                       frames_per_shot: int = 3) -> List[Keyframe]:
        """
        Extract multiple keyframes per shot for better representation
        
        Args:
            video_path: Path to video file
            shots: List of detected shots
            frames_per_shot: Number of keyframes to extract per shot
            
        Returns:
            List of extracted keyframes
        """
        logger.info(f"Extracting {frames_per_shot} keyframes per shot from {len(shots)} shots")
        
        loop = asyncio.get_event_loop()
        keyframes = await loop.run_in_executor(
            self.executor,
            self._extract_multiple_keyframes_sync,
            video_path,
            shots,
            frames_per_shot
        )
        
        logger.info(f"Extracted {len(keyframes)} keyframes total")
        return keyframes
    
    def _extract_multiple_keyframes_sync(self, 
                                       video_path: str, 
                                       shots: List[Shot],
                                       frames_per_shot: int) -> List[Keyframe]:
        """Synchronous multiple keyframes extraction"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_name = Path(video_path).stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        
        # Create directory for this video's keyframes
        video_dir = self.output_dir / video_name
        video_dir.mkdir(exist_ok=True)
        
        for shot in shots:
            try:
                shot_keyframes = self._extract_shot_multiple_keyframes(
                    cap, shot, video_name, video_dir, fps, frames_per_shot
                )
                keyframes.extend(shot_keyframes)
            except Exception as e:
                logger.error(f"Failed to extract keyframes for shot {shot.shot_id}: {e}")
                continue
        
        cap.release()
        return keyframes
    
    def _extract_shot_multiple_keyframes(self, 
                                       cap: cv2.VideoCapture,
                                       shot: Shot,
                                       video_name: str,
                                       video_dir: Path,
                                       fps: float,
                                       frames_per_shot: int) -> List[Keyframe]:
        """Extract multiple keyframes from a single shot"""
        keyframes = []
        shot_length = shot.end_frame - shot.start_frame
        
        # Skip very short shots
        if shot_length < frames_per_shot * 10:  # Need at least 10 frames between keyframes
            return [self._extract_shot_keyframe(cap, shot, video_name, video_dir, fps)]
        
        # Calculate frame positions
        frame_positions = []
        for i in range(frames_per_shot):
            position = shot.start_frame + (i + 1) * shot_length // (frames_per_shot + 1)
            frame_positions.append(position)
        
        # Extract keyframes at calculated positions
        for i, frame_pos in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame {frame_pos} for shot {shot.shot_id}")
                continue
            
            # Resize frame
            resized_frame = cv2.resize(frame, self.image_size)
            
            # Generate keyframe ID and path
            keyframe_id = f"{video_name}_shot_{shot.shot_id:04d}_kf_{i:02d}_frame_{frame_pos:06d}"
            image_filename = f"{keyframe_id}.jpg"
            image_path = video_dir / image_filename
            
            # Save keyframe image
            cv2.imwrite(
                str(image_path), 
                resized_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            
            # Create keyframe object
            keyframe = Keyframe(
                keyframe_id=keyframe_id,
                shot_id=shot.shot_id,
                frame_number=frame_pos,
                timestamp=frame_pos / fps,
                image_path=str(image_path),
                image_data=resized_frame
            )
            
            keyframes.append(keyframe)
        
        return keyframes
    
    def load_keyframe_image(self, keyframe: Keyframe) -> np.ndarray:
        """Load keyframe image from disk"""
        if keyframe.image_data is not None:
            return keyframe.image_data
        
        image_path = Path(keyframe.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Keyframe image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    async def cleanup_keyframes(self, video_name: str):
        """Remove all keyframes for a video"""
        video_dir = self.output_dir / video_name
        if video_dir.exists():
            import shutil
            shutil.rmtree(video_dir)
            logger.info(f"Cleaned up keyframes for video: {video_name}")
