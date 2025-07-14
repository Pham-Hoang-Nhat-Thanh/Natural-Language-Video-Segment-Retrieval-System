"""
Database interface for the search service.
Handles video metadata, search history, and result caching.
"""

import asyncpg
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import time
from datetime import datetime, timedelta
from config import Settings

logger = logging.getLogger(__name__)

class SearchDatabase:
    """Database interface for search service operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = None
        self.connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        # Use DATABASE_URL if available, otherwise build from individual settings
        if hasattr(self.settings, 'DATABASE_URL') and self.settings.DATABASE_URL:
            return self.settings.DATABASE_URL
        else:
            return (
                f"postgresql://{self.settings.DB_USER}:{self.settings.DB_PASSWORD}"
                f"@{self.settings.DB_HOST}:{self.settings.DB_PORT}/{self.settings.DB_NAME}"
            )
    
    async def connect(self):
        """Initialize database connection pool with retry logic."""
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=10,
                    command_timeout=30
                )
                
                # Test connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                
                logger.info(f"Connected to search database on attempt {attempt + 1}")
                
                # Initialize schema if needed
                await self._initialize_schema()
                return
                
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to database after {max_retries} attempts")
                    raise
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from search database")
    
    async def _initialize_schema(self):
        """Initialize database schema for search service."""
        schema_sql = """
        -- Videos table for storing video metadata
        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) UNIQUE NOT NULL,
            title TEXT,
            description TEXT,
            duration FLOAT,
            file_path TEXT,
            thumbnail_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Video segments table for storing detected segments
        CREATE TABLE IF NOT EXISTS video_segments (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) NOT NULL,
            segment_id VARCHAR(255) UNIQUE NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            keyframe_path TEXT,
            transcript TEXT,
            embedding_id INTEGER,
            confidence FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );
        
        -- Search queries table for logging and analytics
        CREATE TABLE IF NOT EXISTS search_queries (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            query_embedding BYTEA,
            user_id VARCHAR(255),
            session_id VARCHAR(255),
            results_count INTEGER DEFAULT 0,
            search_time_ms FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Search results table for caching and analytics
        CREATE TABLE IF NOT EXISTS search_results (
            id SERIAL PRIMARY KEY,
            query_id INTEGER NOT NULL,
            segment_id VARCHAR(255) NOT NULL,
            rank_position INTEGER NOT NULL,
            similarity_score FLOAT,
            rerank_score FLOAT,
            final_score FLOAT,
            boundary_refined BOOLEAN DEFAULT FALSE,
            refined_start_time FLOAT,
            refined_end_time FLOAT,
            clicked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES search_queries(id) ON DELETE CASCADE,
            FOREIGN KEY (segment_id) REFERENCES video_segments(segment_id) ON DELETE CASCADE
        );
        
        -- User interactions for feedback and analytics
        CREATE TABLE IF NOT EXISTS user_interactions (
            id SERIAL PRIMARY KEY,
            query_id INTEGER,
            segment_id VARCHAR(255),
            interaction_type VARCHAR(50) NOT NULL, -- 'click', 'play', 'like', 'dislike'
            interaction_data JSONB DEFAULT '{}'::jsonb,
            user_id VARCHAR(255),
            session_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES search_queries(id) ON DELETE SET NULL,
            FOREIGN KEY (segment_id) REFERENCES video_segments(segment_id) ON DELETE CASCADE
        );
        
        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id);
        CREATE INDEX IF NOT EXISTS idx_video_segments_video_id ON video_segments(video_id);
        CREATE INDEX IF NOT EXISTS idx_video_segments_segment_id ON video_segments(segment_id);
        CREATE INDEX IF NOT EXISTS idx_video_segments_times ON video_segments(start_time, end_time);
        CREATE INDEX IF NOT EXISTS idx_search_queries_text ON search_queries USING gin(to_tsvector('english', query_text));
        CREATE INDEX IF NOT EXISTS idx_search_queries_created_at ON search_queries(created_at);
        CREATE INDEX IF NOT EXISTS idx_search_results_query_id ON search_results(query_id);
        CREATE INDEX IF NOT EXISTS idx_search_results_segment_id ON search_results(segment_id);
        CREATE INDEX IF NOT EXISTS idx_search_results_scores ON search_results(final_score DESC);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_created_at ON user_interactions(created_at);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        logger.info("Database schema initialized")
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video metadata by video ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM videos WHERE video_id = $1",
                video_id
            )
            
            if row:
                return dict(row)
            return None
    
    async def get_video_segments(
        self,
        video_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get video segments for a video, optionally filtered by time range."""
        query = "SELECT * FROM video_segments WHERE video_id = $1"
        params = [video_id]
        
        if start_time is not None:
            query += " AND end_time >= $2"
            params.append(start_time)
        
        if end_time is not None:
            param_num = len(params) + 1
            query += f" AND start_time <= ${param_num}"
            params.append(end_time)
        
        query += " ORDER BY start_time"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def get_segment_metadata(self, segment_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for multiple segments."""
        if not segment_ids:
            return {}
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT vs.*, v.title, v.description, v.duration, v.file_path, v.thumbnail_path
                FROM video_segments vs
                JOIN videos v ON vs.video_id = v.video_id
                WHERE vs.segment_id = ANY($1)
                """,
                segment_ids
            )
            
            return {row['segment_id']: dict(row) for row in rows}
    
    async def log_search_query(
        self,
        query_text: str,
        query_embedding: Optional[bytes] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log a search query and return the query ID."""
        async with self.pool.acquire() as conn:
            query_id = await conn.fetchval(
                """
                INSERT INTO search_queries (query_text, query_embedding, user_id, session_id, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                query_text,
                query_embedding,
                user_id,
                session_id,
                json.dumps(metadata) if metadata else '{}'
            )
            
            return query_id
    
    async def update_search_query_results(
        self,
        query_id: int,
        results_count: int,
        search_time_ms: float
    ):
        """Update search query with results information."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE search_queries 
                SET results_count = $2, search_time_ms = $3
                WHERE id = $1
                """,
                query_id,
                results_count,
                search_time_ms
            )
    
    async def log_search_results(
        self,
        query_id: int,
        results: List[Dict[str, Any]]
    ):
        """Log search results for a query."""
        if not results:
            return
        
        values = []
        for rank, result in enumerate(results):
            values.append((
                query_id,
                result.get('segment_id'),
                rank + 1,
                result.get('similarity_score', 0.0),
                result.get('rerank_score'),
                result.get('final_score', result.get('similarity_score', 0.0)),
                result.get('boundary_refined', False),
                result.get('refined_start_time'),
                result.get('refined_end_time')
            ))
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO search_results 
                (query_id, segment_id, rank_position, similarity_score, rerank_score, 
                 final_score, boundary_refined, refined_start_time, refined_end_time)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                values
            )
    
    async def log_user_interaction(
        self,
        interaction_type: str,
        query_id: Optional[int] = None,
        segment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        interaction_data: Optional[Dict[str, Any]] = None
    ):
        """Log user interaction."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_interactions 
                (query_id, segment_id, interaction_type, interaction_data, user_id, session_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                query_id,
                segment_id,
                interaction_type,
                json.dumps(interaction_data) if interaction_data else '{}',
                user_id,
                session_id
            )
    
    async def get_search_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get search analytics data."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        async with self.pool.acquire() as conn:
            # Query statistics
            query_stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(search_time_ms) as avg_search_time,
                    AVG(results_count) as avg_results_count
                FROM search_queries 
                WHERE created_at BETWEEN $1 AND $2
                """,
                start_date, end_date
            )
            
            # Top queries
            top_queries = await conn.fetch(
                """
                SELECT query_text, COUNT(*) as frequency
                FROM search_queries 
                WHERE created_at BETWEEN $1 AND $2
                GROUP BY query_text
                ORDER BY frequency DESC
                LIMIT $3
                """,
                start_date, end_date, limit
            )
            
            # Click-through rates
            ctr_data = await conn.fetchrow(
                """
                SELECT 
                    COUNT(DISTINCT sr.query_id) as queries_with_results,
                    COUNT(DISTINCT ui.query_id) as queries_with_clicks,
                    CASE 
                        WHEN COUNT(DISTINCT sr.query_id) > 0 
                        THEN COUNT(DISTINCT ui.query_id)::float / COUNT(DISTINCT sr.query_id) 
                        ELSE 0 
                    END as ctr
                FROM search_results sr
                LEFT JOIN user_interactions ui ON sr.query_id = ui.query_id AND ui.interaction_type = 'click'
                JOIN search_queries sq ON sr.query_id = sq.id
                WHERE sq.created_at BETWEEN $1 AND $2
                """,
                start_date, end_date
            )
            
            return {
                'query_stats': dict(query_stats) if query_stats else {},
                'top_queries': [dict(row) for row in top_queries],
                'click_through_rate': dict(ctr_data) if ctr_data else {},
                'date_range': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            }
    
    async def get_popular_segments(
        self,
        limit: int = 50,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get most popular video segments based on interactions."""
        start_date = datetime.now() - timedelta(days=days)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    vs.segment_id,
                    vs.video_id,
                    vs.start_time,
                    vs.end_time,
                    v.title,
                    v.description,
                    COUNT(ui.id) as interaction_count,
                    COUNT(CASE WHEN ui.interaction_type = 'click' THEN 1 END) as click_count,
                    COUNT(CASE WHEN ui.interaction_type = 'play' THEN 1 END) as play_count
                FROM video_segments vs
                JOIN videos v ON vs.video_id = v.video_id
                LEFT JOIN user_interactions ui ON vs.segment_id = ui.segment_id
                WHERE ui.created_at >= $1 OR ui.created_at IS NULL
                GROUP BY vs.segment_id, vs.video_id, vs.start_time, vs.end_time, v.title, v.description
                ORDER BY interaction_count DESC, click_count DESC
                LIMIT $2
                """,
                start_date, limit
            )
            
            return [dict(row) for row in rows]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for database connectivity."""
        try:
            if not self.pool:
                return {"status": "unhealthy", "error": "No connection pool"}
            
            async with self.pool.acquire() as conn:
                # Test query
                result = await conn.fetchval("SELECT 1")
                
                # Get basic stats
                video_count = await conn.fetchval("SELECT COUNT(*) FROM videos")
                segment_count = await conn.fetchval("SELECT COUNT(*) FROM video_segments")
                query_count = await conn.fetchval("SELECT COUNT(*) FROM search_queries")
                
                return {
                    "status": "healthy",
                    "connection_test": result == 1,
                    "stats": {
                        "video_count": video_count,
                        "segment_count": segment_count,
                        "query_count": query_count
                    }
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None

    async def connect(self):
        """Connect to the database"""
        try:
            self.connection = await asyncpg.connect(self.database_url)
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the database"""
        if self.connection:
            await self.connection.close()
            logger.info("Database disconnected")

    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.connection:
                return False
            result = await self.connection.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_embeddings(self, limit: int = 10000):
        """Get all embeddings for ANN search"""
        try:
            rows = await self.connection.fetch("""
                SELECT e.id, e.keyframe_id, e.video_id, e.embedding, k.timestamp, s.start_time, s.end_time
                FROM embeddings e
                JOIN keyframes k ON e.keyframe_id = k.id
                JOIN shots s ON k.shot_id = s.id
                ORDER BY e.video_id, k.timestamp
                LIMIT $1
            """, limit)
            
            embeddings = []
            metadata = []
            
            for row in rows:
                embeddings.append(np.array(row['embedding']))
                metadata.append({
                    "id": row['id'],
                    "keyframe_id": row['keyframe_id'],
                    "video_id": row['video_id'],
                    "timestamp": row['timestamp'],
                    "start_time": row['start_time'],
                    "end_time": row['end_time']
                })
            
            return np.array(embeddings) if embeddings else np.array([]), metadata
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return np.array([]), []

    async def get_video_segments(self, video_id: str):
        """Get video segments with metadata"""
        try:
            rows = await self.connection.fetch("""
                SELECT e.id, e.keyframe_id, e.video_id, e.embedding, 
                       k.timestamp, k.thumbnail_path,
                       s.shot_index, s.start_time, s.end_time
                FROM embeddings e
                JOIN keyframes k ON e.keyframe_id = k.id
                JOIN shots s ON k.shot_id = s.id
                WHERE e.video_id = $1
                ORDER BY k.timestamp
            """, video_id)
            
            segments = []
            for row in rows:
                segments.append({
                    "id": row['id'],
                    "keyframe_id": row['keyframe_id'],
                    "video_id": row['video_id'],
                    "embedding": np.array(row['embedding']),
                    "timestamp": row['timestamp'],
                    "thumbnail_path": row['thumbnail_path'],
                    "shot_index": row['shot_index'],
                    "start_time": row['start_time'],
                    "end_time": row['end_time']
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to get video segments: {e}")
            return []

    async def get_search_stats(self):
        """Get search service statistics"""
        try:
            total_embeddings = await self.connection.fetchval("SELECT COUNT(*) FROM embeddings")
            unique_videos = await self.connection.fetchval(
                "SELECT COUNT(DISTINCT video_id) FROM embeddings"
            )
            
            return {
                "total_embeddings": total_embeddings,
                "unique_videos": unique_videos
            }
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}
