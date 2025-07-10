import logging
from datetime import datetime
import json
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

# Import our custom modules with error handling
try:
    from text_encoder import TextEncoder
    from ann_search import ANNSearchEngine
    from reranker import CrossEncoderReranker
    from boundary_regressor import BoundaryRegressor
    from database import SearchDatabase
    from config import Settings
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    # Define minimal fallback classes for development
    class Settings:
        def __init__(self):
            self.SERVICE_NAME = "search-service"
            self.HOST = "0.0.0.0"
            self.PORT = 8002
            self.DEBUG = True
    
    class TextEncoder:
        def __init__(self, settings): pass
    class ANNSearchEngine:
        def __init__(self, settings): pass  
    class CrossEncoderReranker:
        def __init__(self, settings): pass
    class BoundaryRegressor:
        def __init__(self, settings): pass
    class SearchDatabase:
        def __init__(self, settings): pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    threshold: float = 0.5

class SearchResult(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    score: float
    thumbnail_url: str
    title: str
    description: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float
    total_results: int

class TextEmbedRequest(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str
    candidates: List[Dict[str, Any]]

class RegressionRequest(BaseModel):
    video_segments: List[Dict[str, Any]]
    query_embedding: List[float]

# Initialize services
text_encoder = TextEncoder(settings.model_path, settings.clip_model_name)
ann_search = ANNSearchEngine(settings.faiss_index_path)
reranker = CrossEncoderReranker(settings.reranker_model_path)
boundary_regressor = BoundaryRegressor(settings.regressor_model_path)
search_db = SearchDatabase(settings.database_url, settings.redis_url)

app = FastAPI(
    title="Video Search Service",
    description="Natural language video segment search service",
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
    logger.info("Starting Video Search Service...")
    
    try:
        # Load models
        await text_encoder.load_model()
        await ann_search.load_index()
        await reranker.load_model()
        await boundary_regressor.load_model()
        
        # Connect to databases
        await search_db.connect()
        
        logger.info("Video Search Service initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

@app.on_shutdown
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Video Search Service...")
    await search_db.disconnect()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "video-search",
        "version": "1.0.0",
        "models_loaded": {
            "text_encoder": text_encoder.is_loaded,
            "ann_search": ann_search.is_loaded,
            "reranker": reranker.is_loaded,
            "boundary_regressor": boundary_regressor.is_loaded
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Main search endpoint - performs complete search pipeline
    """
    start_time = time.time()
    
    try:
        logger.info(f"Search request: '{request.query}' (top_k={request.top_k})")
        
        # Step 1: Encode query text
        query_embedding = await text_encoder.encode_text(request.query)
        
        # Step 2: ANN search for candidates
        candidates = await ann_search.search(
            query_embedding, 
            top_k=min(request.top_k * 5, 100)  # Get more candidates for reranking
        )
        
        if not candidates:
            return SearchResponse(
                results=[],
                query_time_ms=(time.time() - start_time) * 1000,
                total_results=0
            )
        
        # Step 3: Rerank candidates
        reranked_candidates = await reranker.rerank(
            request.query, 
            candidates[:50]  # Limit for reranking performance
        )
        
        # Step 4: Boundary regression for top candidates
        top_candidates = reranked_candidates[:request.top_k]
        refined_segments = await boundary_regressor.refine_boundaries(
            top_candidates, 
            query_embedding
        )
        
        # Step 5: Format results
        results = []
        for segment in refined_segments:
            if segment['score'] >= request.threshold:
                result = SearchResult(
                    video_id=segment['video_id'],
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    score=segment['score'],
                    thumbnail_url=f"/static/thumbnails/{segment['video_id']}/frame_{int(segment['start_time'])}.jpg",
                    title=segment.get('title', f"Video {segment['video_id']}"),
                    description=segment.get('description')
                )
                results.append(result)
        
        query_time = (time.time() - start_time) * 1000
        
        # Log search metrics
        await search_db.log_search_metrics(
            query=request.query,
            results_count=len(results),
            query_time_ms=query_time
        )
        
        logger.info(f"Search completed: {len(results)} results in {query_time:.1f}ms")
        
        return SearchResponse(
            results=results,
            query_time_ms=query_time,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/embed/text")
async def embed_text(request: TextEmbedRequest):
    """Generate text embedding"""
    try:
        embedding = await text_encoder.encode_text(request.text)
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model": text_encoder.model_name
        }
    except Exception as e:
        logger.error(f"Text embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/ann")
async def ann_search_endpoint(query_embedding: List[float], top_k: int = 10):
    """Direct ANN search endpoint"""
    try:
        import numpy as np
        embedding = np.array(query_embedding, dtype=np.float32)
        candidates = await ann_search.search(embedding, top_k=top_k)
        return {"candidates": candidates}
    except Exception as e:
        logger.error(f"ANN search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rerank")
async def rerank_endpoint(request: RerankRequest):
    """Rerank search candidates"""
    try:
        reranked = await reranker.rerank(request.query, request.candidates)
        return {"reranked_results": reranked}
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/regress")
async def regress_boundaries(request: RegressionRequest):
    """Refine segment boundaries"""
    try:
        import numpy as np
        query_embedding = np.array(request.query_embedding, dtype=np.float32)
        refined = await boundary_regressor.refine_boundaries(
            request.video_segments, 
            query_embedding
        )
        return {"refined_segments": refined}
    except Exception as e:
        logger.error(f"Boundary regression failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = await search_db.get_search_stats()
        
        return {
            "total_searches": stats.get("total_searches", 0),
            "avg_query_time_ms": stats.get("avg_query_time_ms", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "total_videos_indexed": await ann_search.get_index_size(),
            "model_info": {
                "text_encoder": text_encoder.get_model_info(),
                "ann_search": ann_search.get_index_info(),
                "reranker": reranker.get_model_info(),
                "boundary_regressor": boundary_regressor.get_model_info()
            },
            "service_uptime": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/similar/{video_id}")
async def find_similar_videos(video_id: str, top_k: int = 10):
    """Find videos similar to a given video"""
    try:
        # Get video embedding from database
        video_embedding = await search_db.get_video_embedding(video_id)
        
        if video_embedding is None:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Search for similar videos
        similar = await ann_search.search(video_embedding, top_k=top_k + 1)
        
        # Remove the query video itself
        similar = [s for s in similar if s['video_id'] != video_id][:top_k]
        
        return {"similar_videos": similar}
        
    except Exception as e:
        logger.error(f"Similar video search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/index/{video_id}")
async def remove_from_index(video_id: str):
    """Remove video from search index"""
    try:
        await ann_search.remove_video(video_id)
        await search_db.remove_video_embeddings(video_id)
        
        return {"message": f"Video {video_id} removed from index"}
        
    except Exception as e:
        logger.error(f"Failed to remove video from index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/rebuild")
async def rebuild_index():
    """Rebuild the search index from database"""
    try:
        logger.info("Starting index rebuild...")
        
        # This would be a long-running operation
        # In production, this should be handled by a background task
        await ann_search.rebuild_index_from_db(search_db)
        
        logger.info("Index rebuild completed")
        return {"message": "Index rebuilt successfully"}
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True if settings.debug else False,
        log_level="info"
    )
