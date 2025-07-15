import asyncpg
import redis.asyncio as redis
import json
import numpy as np
import asyncio
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import time

from shot_detector import Shot
from keyframe_extractor import Keyframe
from embedding_service import Embedding

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for video ingestion data
    Handles PostgreSQL for metadata and Redis for caching
    """
    
    def __init__(self, database_url: str, redis_url: str = "redis://localhost:6379"):
        self.database_url = database_url
        self.redis_url = redis_url
        self.pg_pool = None
        self.redis_client = None
    
    async def connect(self):
        """Initialize database connections with retry logic"""
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # PostgreSQL connection pool
                self.pg_pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                
                # Redis connection
                self.redis_client = redis.from_url(self.redis_url)
                
                # Test connections
                async with self.pg_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                
                # Create tables if they don't exist
                await self._create_tables()
                
                logger.info(f"Database connections established successfully on attempt {attempt + 1}")
                return
                
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to databases after {max_retries} attempts")
                    raise
    
    async def disconnect(self):
        """Close database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Database connections closed")
    
    async def _create_tables(self):
        """Create database tables if they don't exist"""
        create_tables_sql = """
        -- Videos table
        CREATE TABLE IF NOT EXISTS videos (
            video_id VARCHAR(255) PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size_bytes BIGINT,
            duration_seconds FLOAT,
            fps FLOAT,
            width INTEGER,
            height INTEGER,
            format VARCHAR(50),
            status VARCHAR(50) DEFAULT 'processing',
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            processing_start_time TIMESTAMP WITH TIME ZONE,
            processing_end_time TIMESTAMP WITH TIME ZONE
        );
        
        -- Shots table
        CREATE TABLE IF NOT EXISTS shots (
            shot_id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) REFERENCES videos(video_id) ON DELETE CASCADE,
            shot_index INTEGER NOT NULL,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            confidence FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(video_id, shot_index)
        );
        
        -- Keyframes table
        CREATE TABLE IF NOT EXISTS keyframes (
            keyframe_id VARCHAR(255) PRIMARY KEY,
            video_id VARCHAR(255) REFERENCES videos(video_id) ON DELETE CASCADE,
            shot_id INTEGER REFERENCES shots(shot_id) ON DELETE CASCADE,
            frame_number INTEGER NOT NULL,
            timestamp FLOAT NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            image_size_bytes INTEGER,
            width INTEGER,
            height INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Embeddings table
        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id SERIAL PRIMARY KEY,
            keyframe_id VARCHAR(255) REFERENCES keyframes(keyframe_id) ON DELETE CASCADE,
            model_name VARCHAR(100) NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            embedding_vector BYTEA NOT NULL,  -- Stored as binary
            norm FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(keyframe_id, model_name)  -- Allow one embedding per model per keyframe
        );
        
        -- Processing statistics table
        CREATE TABLE IF NOT EXISTS processing_stats (
            stat_id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) REFERENCES videos(video_id),
            operation VARCHAR(50) NOT NULL,
            duration_seconds FLOAT NOT NULL,
            status VARCHAR(50) NOT NULL,
            error_message TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
        CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
        CREATE INDEX IF NOT EXISTS idx_shots_video_id ON shots(video_id);
        CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);
        CREATE INDEX IF NOT EXISTS idx_keyframes_shot_id ON keyframes(shot_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_keyframe_id ON embeddings(keyframe_id);
        CREATE INDEX IF NOT EXISTS idx_processing_stats_video_id ON processing_stats(video_id);
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(create_tables_sql)
        
        logger.info("Database tables created/verified")
    
    async def store_video_data(self, 
                             video_id: str, 
                             shots: List[Shot], 
                             keyframes: List[Keyframe], 
                             embeddings: List[Embedding],
                             enhanced_features: Optional[Dict] = None,
                             video_metadata: Optional[Dict] = None) -> bool:
        """
        Store complete video processing data
        
        Args:
            video_id: Unique video identifier
            shots: List of detected shots
            keyframes: List of extracted keyframes
            embeddings: List of generated embeddings
            enhanced_features: Dictionary of enhanced features per keyframe
            video_metadata: Optional video metadata
            video_metadata: Additional video metadata
        
        Returns:
            Success status
        """
        try:
            async with self.pg_pool.acquire() as conn:
                async with conn.transaction():
                    # Store video metadata
                    if video_metadata:
                        await self._store_video_metadata(conn, video_id, video_metadata)
                    
                    # Store shots
                    shot_ids = await self._store_shots(conn, video_id, shots)
                    
                    # Store keyframes
                    await self._store_keyframes(conn, video_id, keyframes, shot_ids)
                    
                    # Store embeddings
                    await self._store_embeddings(conn, embeddings)
                    
                    # Store enhanced features if available
                    if enhanced_features:
                        await self._store_enhanced_features(conn, video_id, enhanced_features)
                    
                    # Update video status
                    await conn.execute(
                        "UPDATE videos SET status = 'completed', processing_end_time = NOW() WHERE video_id = $1",
                        video_id
                    )
            
            # Cache embeddings in Redis for fast retrieval
            await self._cache_embeddings(video_id, embeddings)
            
            logger.info(f"Successfully stored data for video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store video data for {video_id}: {e}")
            # Update video status to failed
            try:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE videos SET status = 'failed', error_message = $1 WHERE video_id = $2",
                        str(e), video_id
                    )
            except:
                pass
            return False
    
    async def _store_video_metadata(self, conn, video_id: str, metadata: Dict):
        """Store video metadata"""
        await conn.execute("""
            INSERT INTO videos (
                video_id, filename, file_path, file_size_bytes, duration_seconds,
                fps, width, height, format, status, processing_start_time
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'processing', NOW())
            ON CONFLICT (video_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                file_path = EXCLUDED.file_path,
                file_size_bytes = EXCLUDED.file_size_bytes,
                duration_seconds = EXCLUDED.duration_seconds,
                fps = EXCLUDED.fps,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                format = EXCLUDED.format,
                updated_at = NOW()
        """, 
            video_id,
            metadata.get('filename', ''),
            metadata.get('file_path', ''),
            metadata.get('file_size_bytes', 0),
            metadata.get('duration_seconds', 0.0),
            metadata.get('fps', 0.0),
            metadata.get('width', 0),
            metadata.get('height', 0),
            metadata.get('format', 'unknown')
        )
    
    async def _store_shots(self, conn, video_id: str, shots: List[Shot]) -> Dict[int, int]:
        """Store shots and return mapping of shot_index to shot_id"""
        shot_ids = {}
        
        for shot in shots:
            shot_id = await conn.fetchval("""
                INSERT INTO shots (
                    video_id, shot_index, start_frame, end_frame, 
                    start_time, end_time, confidence
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING shot_id
            """,
                video_id, shot.shot_id, shot.start_frame, shot.end_frame,
                shot.start_time, shot.end_time, shot.confidence
            )
            shot_ids[shot.shot_id] = shot_id
        
        return shot_ids
    
    async def _store_keyframes(self, conn, video_id: str, keyframes: List[Keyframe], shot_ids: Dict[int, int]):
        """Store keyframes"""
        for keyframe in keyframes:
            shot_id = shot_ids.get(keyframe.shot_id)
            if shot_id is None:
                logger.warning(f"No shot_id found for keyframe {keyframe.keyframe_id}")
                continue
            
            await conn.execute("""
                INSERT INTO keyframes (
                    keyframe_id, video_id, shot_id, frame_number, timestamp, image_path
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (keyframe_id) DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    shot_id = EXCLUDED.shot_id,
                    frame_number = EXCLUDED.frame_number,
                    timestamp = EXCLUDED.timestamp,
                    image_path = EXCLUDED.image_path
            """,
                keyframe.keyframe_id, video_id, shot_id,
                keyframe.frame_number, keyframe.timestamp, keyframe.image_path
            )
    
    async def _store_embeddings(self, conn, embeddings: List[Embedding]):
        """Store embeddings"""
        for embedding in embeddings:
            # Convert numpy array to bytes
            embedding_bytes = embedding.embedding.astype(np.float32).tobytes()
            norm = float(np.linalg.norm(embedding.embedding))
            
            await conn.execute("""
                INSERT INTO embeddings (
                    keyframe_id, model_name, embedding_dimension, embedding_vector, norm
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (keyframe_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    embedding_dimension = EXCLUDED.embedding_dimension,
                    embedding_vector = EXCLUDED.embedding_vector,
                    norm = EXCLUDED.norm,
                    created_at = NOW()
            """,
                embedding.keyframe_id, embedding.model_name,
                embedding.dimension, embedding_bytes, norm
            )
    
    async def _store_enhanced_features(self, conn, video_id: str, features: Dict):
        """Store enhanced features for keyframes"""
        for shot_index, feature_data in features.items():
            shot_id = await conn.fetchval(
                "SELECT shot_id FROM shots WHERE video_id = $1 AND shot_index = $2",
                video_id, shot_index
            )
            
            if not shot_id:
                logger.warning(f"No shot found for video {video_id}, shot_index {shot_index}")
                continue
            
            # Assuming feature_data is a dictionary with feature_name: value pairs
            for feature_name, value in feature_data.items():
                await conn.execute("""
                    INSERT INTO keyframe_features (keyframe_id, feature_name, feature_value)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (keyframe_id, feature_name) DO UPDATE SET
                        feature_value = EXCLUDED.feature_value
                """,
                    f"{video_id}_keyframe_{shot_index}", feature_name, value
                )
    
    async def _cache_embeddings(self, video_id: str, embeddings: List[Embedding]):
        """Cache embeddings in Redis for fast retrieval"""
        try:
            # Prepare embedding data for caching
            embedding_data = {}
            for embedding in embeddings:
                embedding_data[embedding.keyframe_id] = {
                    'vector': embedding.embedding.tolist(),
                    'dimension': embedding.dimension,
                    'model': embedding.model_name
                }
            
            # Store in Redis with expiration (24 hours)
            cache_key = f"embeddings:{video_id}"
            await self.redis_client.setex(
                cache_key, 
                86400,  # 24 hours
                json.dumps(embedding_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings for {video_id}: {e}")
    
    async def get_video_shots(self, video_id: str) -> List[Shot]:
        """Get shots for a video"""
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT shot_index, start_frame, end_frame, start_time, end_time, confidence
                FROM shots 
                WHERE video_id = $1 
                ORDER BY shot_index
            """, video_id)
            
            shots = []
            for row in rows:
                shot = Shot(
                    shot_id=row['shot_index'],
                    start_frame=row['start_frame'],
                    end_frame=row['end_frame'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    confidence=row['confidence']
                )
                shots.append(shot)
            
            return shots
    
    async def get_video_status(self, video_id: str) -> Optional[Dict]:
        """Get processing status for a video"""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT video_id, status, error_message, created_at, 
                       processing_start_time, processing_end_time
                FROM videos 
                WHERE video_id = $1
            """, video_id)
            
            if not row:
                return None
            
            return dict(row)
    
    async def update_keyframes(self, video_id: str, keyframes: List[Keyframe]):
        """Update keyframes for a video"""
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Delete existing keyframes
                await conn.execute("DELETE FROM keyframes WHERE video_id = $1", video_id)
                
                # Get shot IDs mapping
                shot_rows = await conn.fetch(
                    "SELECT shot_index, shot_id FROM shots WHERE video_id = $1", 
                    video_id
                )
                shot_ids = {row['shot_index']: row['shot_id'] for row in shot_rows}
                
                # Insert new keyframes
                await self._store_keyframes(conn, video_id, keyframes, shot_ids)
    
    async def delete_video(self, video_id: str):
        """Delete video and all associated data"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM videos WHERE video_id = $1", video_id)
        
        # Remove from Redis cache
        cache_key = f"embeddings:{video_id}"
        await self.redis_client.delete(cache_key)
    
    async def get_ingestion_stats(self) -> Dict:
        """Get ingestion statistics"""
        async with self.pg_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_videos,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_videos,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_videos,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_videos
                FROM videos
            """)
            
            shot_stats = await conn.fetchrow("""
                SELECT COUNT(*) as total_shots
                FROM shots s
                JOIN videos v ON s.video_id = v.video_id
                WHERE v.status = 'completed'
            """)
            
            keyframe_stats = await conn.fetchrow("""
                SELECT COUNT(*) as total_keyframes
                FROM keyframes k
                JOIN videos v ON k.video_id = v.video_id
                WHERE v.status = 'completed'
            """)
            
            processing_stats = await conn.fetchrow("""
                SELECT AVG(EXTRACT(EPOCH FROM (processing_end_time - processing_start_time))) as avg_processing_time
                FROM videos
                WHERE status = 'completed' AND processing_start_time IS NOT NULL AND processing_end_time IS NOT NULL
            """)
            
            return {
                'total_videos': stats['total_videos'],
                'completed_videos': stats['completed_videos'], 
                'failed_videos': stats['failed_videos'],
                'processing_videos': stats['processing_videos'],
                'total_shots': shot_stats['total_shots'],
                'total_keyframes': keyframe_stats['total_keyframes'],
                'avg_processing_time': float(processing_stats['avg_processing_time'] or 0)
            }
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_video_status(self, video_id: str):
        """Get processing status for a video"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT * FROM videos WHERE video_id = $1", video_id
                )
                if not result:
                    return None
                
                return {
                    "video_id": video_id,
                    "processed": result['processing_end_time'] is not None,
                    "shots_count": await conn.fetchval(
                        "SELECT COUNT(*) FROM shots WHERE video_id = $1", video_id
                    ),
                    "keyframes_count": await conn.fetchval(
                        "SELECT COUNT(*) FROM keyframes WHERE video_id = $1", video_id
                    ),
                    "embeddings_count": await conn.fetchval(
                        "SELECT COUNT(*) FROM embeddings WHERE video_id = $1", video_id
                    ),
                    "processed_at": result['processing_end_time'].isoformat() if result['processing_end_time'] else None
                }
        except Exception as e:
            logger.error(f"Failed to get video status: {e}")
            return None

    async def store_video_data(self, video_id: str, shots: list, keyframes: list, embeddings: list, video_metadata: dict = None):
        """Store video processing data"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Store video metadata
                await conn.execute("""
                    INSERT INTO videos (video_id, filename, file_path, file_size_bytes, duration_seconds, fps, width, height, format, status, processing_start_time, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'processing', NOW(), NOW(), NOW())
                    ON CONFLICT (video_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        file_path = EXCLUDED.file_path,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        duration_seconds = EXCLUDED.duration_seconds,
                        fps = EXCLUDED.fps,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        format = EXCLUDED.format,
                        updated_at = NOW()
                """, 
                    video_id,
                    video_metadata.get('filename', f"{video_id}.mp4") if video_metadata else f"{video_id}.mp4",
                    video_metadata.get('file_path') if video_metadata else None,
                    video_metadata.get('file_size_bytes', 0) if video_metadata else 0,
                    video_metadata.get('duration_seconds', 0.0) if video_metadata else 0.0,
                    video_metadata.get('fps', 0.0) if video_metadata else 0.0,
                    video_metadata.get('width', 0) if video_metadata else 0,
                    video_metadata.get('height', 0) if video_metadata else 0,
                    video_metadata.get('format', 'unknown') if video_metadata else 'unknown'
                )
                
                # Store shots
                for i, shot in enumerate(shots):
                    shot_result = await conn.fetchrow("""
                        INSERT INTO shots (video_id, shot_index, start_frame, end_frame, start_time, end_time, confidence, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        ON CONFLICT (video_id, shot_index) DO UPDATE SET
                            start_frame = EXCLUDED.start_frame,
                            end_frame = EXCLUDED.end_frame,
                            start_time = EXCLUDED.start_time,
                            end_time = EXCLUDED.end_time,
                            confidence = EXCLUDED.confidence
                        RETURNING shot_id
                    """, video_id, i, shot.get('start_frame', 0), shot.get('end_frame', 0), 
                        shot.get('start_time', 0), shot.get('end_time', 0), shot.get('confidence', 0.0))
                    
                    # Store shot_id for keyframe references
                    if shot_result:
                        shot_db_id = shot_result['shot_id']
                    else:
                        # Get existing shot_id if update occurred
                        shot_db_id = await conn.fetchval(
                            "SELECT shot_id FROM shots WHERE video_id = $1 AND shot_index = $2",
                            video_id, i
                        )
                
                # Store keyframes and embeddings
                shot_id_map = {}  # Map shot_index to database shot_id
                
                for i, (keyframe, embedding) in enumerate(zip(keyframes, embeddings)):
                    keyframe_id = f"{video_id}_keyframe_{i}"
                    shot_index = keyframe.get('shot_index', 0)
                    
                    # Get the actual database shot_id
                    if shot_index not in shot_id_map:
                        shot_db_id = await conn.fetchval(
                            "SELECT shot_id FROM shots WHERE video_id = $1 AND shot_index = $2",
                            video_id, shot_index
                        )
                        shot_id_map[shot_index] = shot_db_id
                    else:
                        shot_db_id = shot_id_map[shot_index]
                    
                    if not shot_db_id:
                        logger.warning(f"No shot found for video {video_id}, shot_index {shot_index}")
                        continue
                    
                    await conn.execute("""
                        INSERT INTO keyframes (keyframe_id, video_id, shot_id, frame_number, timestamp, image_path, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        ON CONFLICT (keyframe_id) DO UPDATE SET
                            shot_id = EXCLUDED.shot_id,
                            frame_number = EXCLUDED.frame_number,
                            timestamp = EXCLUDED.timestamp,
                            image_path = EXCLUDED.image_path
                    """, keyframe_id, video_id, shot_db_id, keyframe.get('frame_number', 0), keyframe.get('timestamp', 0), 
                        keyframe.get('image_path'))
                    
                    await conn.execute("""
                        INSERT INTO embeddings (keyframe_id, model_name, embedding_dimension, embedding_vector, norm, created_at)
                        VALUES ($1, $2, $3, $4, $5, NOW())
                        ON CONFLICT (keyframe_id, model_name) DO UPDATE SET
                            embedding_dimension = EXCLUDED.embedding_dimension,
                            embedding_vector = EXCLUDED.embedding_vector,
                            norm = EXCLUDED.norm
                    """, keyframe_id, embedding.model_name, embedding.dimension, 
                        embedding.embedding.astype(np.float32).tobytes(), float(np.linalg.norm(embedding.embedding)))
                
                # Update video status
                await conn.execute(
                    "UPDATE videos SET status = 'completed', processing_end_time = NOW() WHERE video_id = $1",
                    video_id
                )
    
            # Cache embeddings in Redis for fast retrieval
            await self._cache_embeddings(video_id, embeddings)
            
            logger.info(f"Successfully stored data for video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store video data for {video_id}: {e}")
            # Update video status to failed
            try:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE videos SET status = 'failed', error_message = $1 WHERE video_id = $2",
                        str(e), video_id
                    )
            except:
                pass
            return False

    async def delete_video(self, video_id: str):
        """Delete video data from database"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("DELETE FROM videos WHERE video_id = $1", video_id)
            
            # Remove from Redis cache
            cache_key = f"embeddings:{video_id}"
            await self.redis_client.delete(cache_key)
            
            logger.info(f"Deleted video data for {video_id}")
        except Exception as e:
            logger.error(f"Failed to delete video data: {e}")
            raise

    async def get_ingestion_stats(self):
        """Get ingestion service statistics"""
        try:
            async with self.pg_pool.acquire() as conn:
                total_videos = await conn.fetchval("SELECT COUNT(*) FROM videos")
                processed_videos = await conn.fetchval(
                    "SELECT COUNT(*) FROM videos WHERE processing_end_time IS NOT NULL"
                )
                total_shots = await conn.fetchval("SELECT COUNT(*) FROM shots")
                total_keyframes = await conn.fetchval("SELECT COUNT(*) FROM keyframes")
                
                return {
                    "total_videos": total_videos,
                    "processed_videos": processed_videos,
                    "total_shots": total_shots,
                    "total_keyframes": total_keyframes
                }
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {}
