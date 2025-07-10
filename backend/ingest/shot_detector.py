import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float

class ShotDetector:
    """
    Shot detection service using histogram-based and edge change ratio methods
    """
    
    def __init__(self, 
                 hist_threshold: float = 0.3,
                 ecr_threshold: float = 0.4,
                 min_shot_length: int = 30):
        """
        Initialize shot detector
        
        Args:
            hist_threshold: Threshold for histogram-based detection
            ecr_threshold: Threshold for edge change ratio detection  
            min_shot_length: Minimum number of frames for a shot
        """
        self.hist_threshold = hist_threshold
        self.ecr_threshold = ecr_threshold
        self.min_shot_length = min_shot_length
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def detect_shots(self, video_path: str) -> List[Shot]:
        """
        Detect shot boundaries in video
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of detected shots
        """
        logger.info(f"Starting shot detection for: {video_path}")
        
        # Run shot detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        shots = await loop.run_in_executor(
            self.executor, 
            self._detect_shots_sync, 
            video_path
        )
        
        logger.info(f"Shot detection complete. Found {len(shots)} shots")
        return shots
    
    def _detect_shots_sync(self, video_path: str) -> List[Shot]:
        """Synchronous shot detection implementation"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Storage for frame features
        hist_features = []
        edge_features = []
        frame_timestamps = []
        
        logger.info(f"Processing {total_frames} frames at {fps} fps")
        
        # Extract features from all frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            frame_timestamps.append(timestamp)
            
            # Calculate histogram features
            hist_feat = self._calculate_histogram_features(frame)
            hist_features.append(hist_feat)
            
            # Calculate edge features
            edge_feat = self._calculate_edge_features(frame)
            edge_features.append(edge_feat)
            
            frame_idx += 1
            
            if frame_idx % 1000 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        # Detect shot boundaries using both methods
        hist_boundaries = self._detect_histogram_boundaries(hist_features)
        edge_boundaries = self._detect_edge_boundaries(edge_features)
        
        # Combine and filter boundaries
        combined_boundaries = self._combine_boundaries(
            hist_boundaries, edge_boundaries, total_frames
        )
        
        # Convert boundaries to shots
        shots = self._boundaries_to_shots(
            combined_boundaries, frame_timestamps, fps
        )
        
        return shots
    
    def _calculate_histogram_features(self, frame: np.ndarray) -> np.ndarray:
        """Calculate normalized color histogram for frame"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        # Concatenate all features
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def _calculate_edge_features(self, frame: np.ndarray) -> float:
        """Calculate edge density for frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density (ratio of edge pixels)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return edge_density
    
    def _detect_histogram_boundaries(self, hist_features: List[np.ndarray]) -> List[int]:
        """Detect shot boundaries using histogram comparison"""
        boundaries = []
        
        for i in range(1, len(hist_features)):
            # Calculate chi-square distance between consecutive frames
            prev_hist = hist_features[i-1]
            curr_hist = hist_features[i]
            
            # Avoid division by zero
            eps = 1e-10
            chi_square = np.sum((prev_hist - curr_hist) ** 2 / (prev_hist + curr_hist + eps))
            
            if chi_square > self.hist_threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _detect_edge_boundaries(self, edge_features: List[float]) -> List[int]:
        """Detect shot boundaries using edge change ratio"""
        boundaries = []
        
        for i in range(1, len(edge_features)):
            prev_edge = edge_features[i-1]
            curr_edge = edge_features[i]
            
            # Calculate edge change ratio
            if prev_edge > 0:
                ecr = abs(curr_edge - prev_edge) / prev_edge
            else:
                ecr = curr_edge
            
            if ecr > self.ecr_threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _combine_boundaries(self, 
                          hist_boundaries: List[int], 
                          edge_boundaries: List[int],
                          total_frames: int) -> List[int]:
        """Combine and filter shot boundaries from different methods"""
        # Combine all boundaries
        all_boundaries = set(hist_boundaries + edge_boundaries)
        
        # Add start and end of video
        all_boundaries.add(0)
        all_boundaries.add(total_frames - 1)
        
        # Sort boundaries
        sorted_boundaries = sorted(list(all_boundaries))
        
        # Filter out boundaries that create too short shots
        filtered_boundaries = [sorted_boundaries[0]]  # Always keep first boundary
        
        for boundary in sorted_boundaries[1:]:
            if boundary - filtered_boundaries[-1] >= self.min_shot_length:
                filtered_boundaries.append(boundary)
        
        # Ensure we end with the last frame
        if filtered_boundaries[-1] != total_frames - 1:
            filtered_boundaries.append(total_frames - 1)
        
        return filtered_boundaries
    
    def _boundaries_to_shots(self, 
                           boundaries: List[int], 
                           timestamps: List[float],
                           fps: float) -> List[Shot]:
        """Convert frame boundaries to shot objects"""
        shots = []
        
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            
            start_time = timestamps[start_frame]
            end_time = timestamps[min(end_frame, len(timestamps) - 1)]
            
            # Calculate confidence based on shot length
            shot_length = end_frame - start_frame
            confidence = min(1.0, shot_length / (fps * 2))  # 2 seconds = full confidence
            
            shot = Shot(
                shot_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence
            )
            
            shots.append(shot)
        
        return shots
