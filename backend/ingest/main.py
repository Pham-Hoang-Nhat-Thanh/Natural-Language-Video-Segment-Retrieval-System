from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
import logging
from datetime import datetime
import asyncio
import glob

from shot_detector import ShotDetector
from keyframe_extractor import KeyframeExtractor
from embedding_service import EmbeddingService
from database import DatabaseManager
from config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Initialize services
shot_detector = ShotDetector()
keyframe_extractor = KeyframeExtractor()
embedding_service = EmbeddingService(settings.model_path)
db_manager = DatabaseManager(settings.database_url)

app = FastAPI(
    title="Video Processing Service",
    description="Service for processing MP4 videos from dataset directory",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_startup
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Video Processing Service...")
    await db_manager.connect()
    await embedding_service.load_model()
    logger.info("Service initialization complete")

@app.on_shutdown
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Video Processing Service...")
    await db_manager.disconnect()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "video-processing",
        "version": "1.0.0"
    }

@app.get("/api/videos")
async def list_videos():
    """List all MP4 videos in the dataset directory"""
    try:
        videos_dir = Path(settings.data_path) / "videos" / "datasets" / "custom"
        videos = []
        
        if videos_dir.exists():
            for video_file in videos_dir.glob("*.mp4"):
                video_id = video_file.stem
                # Check if video is already processed
                status = await db_manager.get_video_status(video_id)
                
                videos.append({
                    "video_id": video_id,
                    "filename": video_file.name,
                    "path": str(video_file),
                    "processed": status is not None,
                    "file_size": video_file.stat().st_size if video_file.exists() else 0
                })
        
        return {"videos": videos}
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")

@app.post("/api/process/{video_id}")
async def process_video(video_id: str):
    """
    Process a specific MP4 video from the dataset directory
    """
    try:
        # Find the video file
        videos_dir = Path(settings.data_path) / "videos" / "datasets" / "custom"
        video_path = videos_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id}.mp4 not found in dataset directory")
        
        logger.info(f"Starting processing for video: {video_id}")
        
        # Step 1: Shot Detection
        logger.info("Detecting shots...")
        shots = await shot_detector.detect_shots(str(video_path))
        logger.info(f"Detected {len(shots)} shots")
        
        # Step 2: Keyframe Extraction
        logger.info("Extracting keyframes...")
        keyframes = await keyframe_extractor.extract_keyframes(str(video_path), shots)
        logger.info(f"Extracted {len(keyframes)} keyframes")
        
        # Step 3: Generate Embeddings
        logger.info("Generating embeddings...")
        embeddings = await embedding_service.generate_embeddings(keyframes)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in Database and Vector Index
        logger.info("Storing data...")
        await db_manager.store_video_data(video_id, shots, keyframes, embeddings)
        
        logger.info(f"Processing complete for video: {video_id}")
        
        return {
            "video_id": video_id,
            "status": "success",
            "shots_detected": len(shots),
            "keyframes_extracted": len(keyframes),
            "embeddings_generated": len(embeddings),
            "message": "Video processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Processing failed for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/process-all")
async def process_all_videos():
    """Process all unprocessed MP4 videos in the dataset directory"""
    try:
        videos_dir = Path(settings.data_path) / "videos" / "datasets" / "custom"
        
        if not videos_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset directory not found")
        
        processed_count = 0
        failed_videos = []
        
        for video_file in videos_dir.glob("*.mp4"):
            video_id = video_file.stem
            
            # Check if already processed
            status = await db_manager.get_video_status(video_id)
            if status:
                logger.info(f"Video {video_id} already processed, skipping")
                continue
            
            try:
                await process_video(video_id)
                processed_count += 1
                logger.info(f"Successfully processed {video_id}")
            except Exception as e:
                logger.error(f"Failed to process {video_id}: {e}")
                failed_videos.append({"video_id": video_id, "error": str(e)})
        
        return {
            "status": "completed",
            "processed_count": processed_count,
            "failed_videos": failed_videos,
            "message": f"Processed {processed_count} videos"
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """Get processing status for a video"""
    try:
        status = await db_manager.get_video_status(video_id)
        
        if not status:
            return {"video_id": video_id, "processed": False}
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get status for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/videos/{video_id}")
async def delete_video_data(video_id: str):
    """Delete video processing data (but keep the MP4 file)"""
    try:
        # Remove from database only - keep the MP4 file
        await db_manager.delete_video(video_id)
        
        # Remove thumbnails if they exist
        thumb_dir = Path(settings.data_path) / "thumbnails" / "datasets" / "custom" / video_id
        if thumb_dir.exists():
            import shutil
            shutil.rmtree(thumb_dir)
        
        return {
            "video_id": video_id,
            "status": "data_deleted",
            "note": "MP4 file preserved in dataset directory"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete data for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = await db_manager.get_ingestion_stats()
        
        # Count total MP4 files in dataset
        videos_dir = Path(settings.data_path) / "videos" / "datasets" / "custom"
        total_files = len(list(videos_dir.glob("*.mp4"))) if videos_dir.exists() else 0
        
        return {
            "total_mp4_files": total_files,
            "processed_videos": stats.get("total_videos", 0),
            "total_shots": stats.get("total_shots", 0),
            "total_keyframes": stats.get("total_keyframes", 0),
            "service_uptime": datetime.utcnow().isoformat(),
            "model_info": {
                "clip_model": embedding_service.model_name,
                "embedding_dim": embedding_service.embedding_dim
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True if os.getenv("NODE_ENV") == "development" else False,
        log_level="info"
    )
