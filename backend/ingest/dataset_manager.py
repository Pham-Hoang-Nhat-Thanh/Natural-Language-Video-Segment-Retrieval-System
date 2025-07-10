"""
Simple Video Manager
Manages MP4 files manually added to the dataset directory.
No uploads, no pre-built datasets - just your MP4 files.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleVideoManager:
    """
    Simple manager for MP4 files in the dataset directory.
    No uploads, no complex dataset management - just processes your MP4 files.
    """
    
    def __init__(self, data_root: str = "/app/data"):
        self.data_root = Path(data_root)
        self.videos_dir = self.data_root / "videos" / "datasets" / "custom"
        
    def list_mp4_files(self) -> List[Dict[str, Any]]:
        """List all MP4 files in the dataset directory"""
        videos = []
        
        if not self.videos_dir.exists():
            logger.warning(f"Dataset directory does not exist: {self.videos_dir}")
            return videos
            
        for video_file in self.videos_dir.glob("*.mp4"):
            videos.append({
                "video_id": video_file.stem,
                "filename": video_file.name,
                "path": str(video_file),
                "file_size": video_file.stat().st_size
            })
            
        return videos
    
    def get_video_path(self, video_id: str) -> Optional[str]:
        """Get the path to a specific video file"""
        video_path = self.videos_dir / f"{video_id}.mp4"
        return str(video_path) if video_path.exists() else None
    
    def video_exists(self, video_id: str) -> bool:
        """Check if a video file exists"""
        video_path = self.videos_dir / f"{video_id}.mp4"
        return video_path.exists()
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a video file"""
        video_path = self.videos_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            return None
            
        return {
            "video_id": video_id,
            "filename": video_path.name,
            "path": str(video_path),
            "file_size": video_path.stat().st_size
        }

# Simple singleton instance
video_manager = SimpleVideoManager()

# Example usage
if __name__ == "__main__":
    def main():
        # Initialize video manager
        manager = SimpleVideoManager()
        
        # List available videos
        videos = manager.list_mp4_files()
        print(f"Found {len(videos)} MP4 files")
        
        for video in videos:
            print(f"- {video['filename']} (ID: {video['video_id']})")
            print(f"  Size: {video['file_size'] / 1024 / 1024:.1f} MB")
            print(f"  Path: {video['path']}")
    
    main()
