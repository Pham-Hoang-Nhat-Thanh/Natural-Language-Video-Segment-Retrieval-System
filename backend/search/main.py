import logging
from datetime import datetime
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import Settings
from text_encoder import TextEncoder
from ann_search import ANNSearchEngine
from reranker import CrossEncoderReranker
from boundary_regressor import BoundaryRegressor
from database import SearchDatabase
from query_enhancer import LightweightQueryEnhancer

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
    use_query_enhancement: bool = True

class EnhancedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    threshold: float = 0.5
    use_llm_enhancement: bool = True
    search_weights: Optional[Dict[str, float]] = None

class SearchResult(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    score: float
    thumbnail_url: str
    title: str
    description: Optional[str] = None
    enhanced_query_used: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float
    total_results: int
    original_query: str
    enhanced_query: Optional[str] = None
    query_analysis: Optional[Dict] = None

class TextEmbedRequest(BaseModel):
    text: str

class RerankRequest(BaseModel):
    query: str
    candidates: List[Dict[str, Any]]

class RegressionRequest(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    query: str

class QueryEnhancementRequest(BaseModel):
    query: str
    context: Optional[Dict] = None
    use_llm: bool = True

# Initialize services with full settings
text_encoder = TextEncoder(settings)
ann_search = ANNSearchEngine(settings)
reranker = CrossEncoderReranker(settings)
boundary_regressor = BoundaryRegressor(settings)
search_db = SearchDatabase(settings)
query_enhancer = LightweightQueryEnhancer(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Video Search Service...")
    
    try:
        # Models are loaded in constructors, just check if they loaded successfully
        if not text_encoder.models:
            raise Exception("Text encoder failed to load")
        
        # For now, skip loading other services until text encoder is working
        # await ann_search.load_index()
        # await reranker.load_model()
        # await boundary_regressor.load_model()
        
        # Connect to databases
        await search_db.connect()
        
        logger.info("Video Search Service initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video Search Service...")
    await search_db.disconnect()

app = FastAPI(
    title="Video Search Service",
    description="Natural language video segment search service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if text_encoder.is_loaded else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "video-search",
        "version": "1.0.0",
        "models_loaded": {
            "text_encoder": text_encoder.is_loaded,
            "ann_search": False,  # Temporarily disabled
            "reranker": False,    # Temporarily disabled
            "boundary_regressor": False  # Temporarily disabled
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Main search endpoint - performs complete search pipeline with optional query enhancement
    """
    start_time = time.time()
    
    try:
        logger.info(f"Search request: '{request.query}' (top_k={request.top_k}, enhancement={request.use_query_enhancement})")
        
        # Step 1: Optionally enhance the query
        query_to_use = request.query
        enhanced_query = None
        query_analysis_info = None
        
        if request.use_query_enhancement:
            try:
                enhanced_analysis = await query_enhancer.enhance_query(
                    request.query, 
                    use_llm=False  # Use template-based enhancement for faster response
                )
                query_to_use = enhanced_analysis.enhanced_query
                enhanced_query = enhanced_analysis.enhanced_query
                query_analysis_info = {
                    "type": enhanced_analysis.query_type,
                    "entities": enhanced_analysis.entities,
                    "confidence": enhanced_analysis.confidence
                }
                logger.info(f"Using enhanced query: '{query_to_use}'")
            except Exception as e:
                logger.warning(f"Query enhancement failed, using original: {e}")
        
        # Step 2: Encode query text (enhanced or original)
        query_embedding = await text_encoder.encode_text(query_to_use)
        
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
            total_results=len(results),
            original_query=request.query,
            enhanced_query=enhanced_query,
            query_analysis=query_analysis_info
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/search/enhanced")
async def enhanced_search_videos(request: EnhancedSearchRequest):
    """
    Enhanced search endpoint with query enhancement and multi-modal features
    """
    start_time = time.time()
    
    try:
        logger.info(f"Enhanced search request: '{request.query}' (top_k={request.top_k})")
        
        # Step 1: Enhance the query using our query enhancer
        enhanced_analysis = await query_enhancer.enhance_query(
            request.query, 
            use_llm=request.use_llm_enhancement
        )
        
        logger.info(f"Enhanced query: '{enhanced_analysis.enhanced_query}'")
        logger.info(f"Query analysis: {enhanced_analysis.query_type}, entities: {enhanced_analysis.entities}")
        
        # Step 2: Encode enhanced query text
        query_embedding = await text_encoder.encode_text(enhanced_analysis.enhanced_query)
        
        # Step 3: ANN search for candidates with enhanced embedding
        candidates = await ann_search.search(
            query_embedding, 
            top_k=min(request.top_k * 5, 100)  # Get more candidates for reranking
        )
        
        if not candidates:
            return SearchResponse(
                results=[],
                query_time_ms=(time.time() - start_time) * 1000,
                total_results=0,
                original_query=request.query,
                enhanced_query=enhanced_analysis.enhanced_query,
                query_analysis={
                    "type": enhanced_analysis.query_type,
                    "entities": enhanced_analysis.entities,
                    "actions": enhanced_analysis.actions,
                    "confidence": enhanced_analysis.confidence
                }
            )
        
        # Step 4: Enhanced reranking with both original and enhanced queries
        reranked_candidates = await reranker.rerank(
            enhanced_analysis.enhanced_query, 
            candidates[:50]  # Limit for reranking performance
        )
        
        # Step 5: Boundary regression for top candidates
        top_candidates = reranked_candidates[:request.top_k]
        refined_segments = await boundary_regressor.refine_boundaries(
            top_candidates, 
            query_embedding
        )
        
        # Step 6: Format results with enhanced information
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
                    description=segment.get('description'),
                    enhanced_query_used=enhanced_analysis.enhanced_query
                )
                results.append(result)
        
        query_time = (time.time() - start_time) * 1000
        
        # Log enhanced search metrics
        await search_db.log_search_metrics(
            query=f"[ENHANCED] {request.query} -> {enhanced_analysis.enhanced_query}",
            results_count=len(results),
            query_time_ms=query_time
        )
        
        logger.info(f"Enhanced search completed: {len(results)} results in {query_time:.1f}ms")
        
        return SearchResponse(
            results=results,
            query_time_ms=query_time,
            total_results=len(results),
            original_query=request.query,
            enhanced_query=enhanced_analysis.enhanced_query,
            query_analysis={
                "type": enhanced_analysis.query_type,
                "entities": enhanced_analysis.entities,
                "actions": enhanced_analysis.actions,
                "scene_context": enhanced_analysis.scene_context,
                "temporal_context": enhanced_analysis.temporal_context,
                "confidence": enhanced_analysis.confidence
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

@app.post("/api/query/enhance")
async def enhance_query_endpoint(request: QueryEnhancementRequest):
    """
    Standalone query enhancement endpoint
    """
    try:
        enhanced_analysis = await query_enhancer.enhance_query(
            request.query,
            context=request.context,
            use_llm=request.use_llm
        )
        
        return {
            "original_query": request.query,
            "enhanced_query": enhanced_analysis.enhanced_query,
            "query_type": enhanced_analysis.query_type,
            "entities": enhanced_analysis.entities,
            "actions": enhanced_analysis.actions,
            "scene_context": enhanced_analysis.scene_context,
            "temporal_context": enhanced_analysis.temporal_context,
            "confidence": enhanced_analysis.confidence
        }
        
    except Exception as e:
        logger.error(f"Query enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query enhancement failed: {str(e)}")

@app.get("/api/query/stats")
async def get_query_enhancement_stats():
    """
    Get query enhancement statistics and model status
    """
    try:
        stats = query_enhancer.get_stats()
        return {
            "enhancement_stats": stats,
            "status": "operational" if stats["model_loaded"] else "limited"
        }
    except Exception as e:
        logger.error(f"Failed to get enhancement stats: {str(e)}")
        return {"error": str(e), "status": "error"}

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

class VideoEmbedRequest(BaseModel):
    video_id: str
    keyframes: List[str] = []

@app.post("/api/embed/video")
async def embed_video(request: VideoEmbedRequest):
    """Generate video embeddings from keyframes"""
    try:
        if not request.keyframes:
            # Get keyframes from database
            keyframes = await search_db.get_video_keyframes(request.video_id)
        else:
            keyframes = request.keyframes
            
        embeddings = []
        for keyframe_path in keyframes:
            # Load image and generate embedding
            embedding = await text_encoder.encode_image(keyframe_path)
            embeddings.append(embedding.tolist())
        
        return {
            "video_id": request.video_id,
            "embeddings": embeddings,
            "keyframes_count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "model": text_encoder.model_name
        }
    except Exception as e:
        logger.error(f"Video embedding failed: {str(e)}")
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
