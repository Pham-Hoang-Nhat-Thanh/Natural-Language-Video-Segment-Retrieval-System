/**
 * API Integration for Video Segment Retrieval
 * 
 * Provides functions to interact with the backend search API
 * Handles data transformation between frontend and backend formats
 */

import { VideoSegment, SearchResult, SearchRequest } from '@/types'

// API Configuration
const API_BASE_URL = typeof window !== 'undefined' 
  ? window.location.protocol + '//' + window.location.hostname + ':8090'
  : 'http://localhost:8090'
const SEARCH_ENDPOINT = '/api/search'
const VIDEOS_ENDPOINT = '/api/videos'

/**
 * Search for video segments using natural language query
 */
export async function searchVideoSegments(query: string, options: { topK?: number, threshold?: number } = {}): Promise<SearchResult> {
  const { topK = 20, threshold = 0.5 } = options
  
  const searchRequest: SearchRequest = {
    query: query.trim(),
    top_k: topK,
    threshold
  }

  try {
    const response = await fetch(`${API_BASE_URL}${SEARCH_ENDPOINT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(searchRequest),
    })

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    
    // Transform backend response to frontend format
    const transformedResults: VideoSegment[] = data.results?.map((result: any, index: number) => ({
      id: result.id || `${result.video_id}_${index}`,
      video_id: result.video_id,
      videoUrl: result.videoUrl || `/api/videos/${result.video_id}/stream`,
      thumbnailUrl: result.thumbnailUrl || result.thumbnail_url || `/api/thumbnails/${result.video_id}/${Math.floor(result.start_time)}.jpg`,
      startTime: result.start_time || result.startTime,
      endTime: result.end_time || result.endTime,
      title: result.title || `Video ${result.video_id}`,
      description: result.description || 'No description available',
      transcript: result.transcript,
      score: result.score || 0
    })) || []

    return {
      results: transformedResults,
      query_time_ms: data.query_time_ms || 0,
      total_results: data.total_results || transformedResults.length
    }

  } catch (error) {
    console.error('Search API error:', error)
    throw new Error(`Failed to search video segments: ${error.message}`)
  }
}

/**
 * Get available videos for processing
 */
export async function getVideos(): Promise<any[]> {
  try {
    const response = await fetch(`${API_BASE_URL}${VIDEOS_ENDPOINT}`)
    
    if (!response.ok) {
      throw new Error(`Failed to fetch videos: ${response.status}`)
    }

    const data = await response.json()
    return data.videos || data || []

  } catch (error) {
    console.error('Videos API error:', error)
    throw new Error(`Failed to get videos: ${error.message}`)
  }
}

/**
 * Process a video for indexing
 */
export async function processVideo(videoId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}${VIDEOS_ENDPOINT}/${videoId}/process`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`Failed to process video: ${response.status}`)
    }

  } catch (error) {
    console.error('Process video API error:', error)
    throw new Error(`Failed to process video: ${error.message}`)
  }
}

/**
 * Upload a new video file
 */
export async function uploadVideo(file: File, title?: string): Promise<any> {
  try {
    const formData = new FormData()
    formData.append('video', file)
    if (title) {
      formData.append('title', title)
    }

    const response = await fetch(`${API_BASE_URL}${VIDEOS_ENDPOINT}/upload`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`Failed to upload video: ${response.status}`)
    }

    return await response.json()

  } catch (error) {
    console.error('Upload video API error:', error)
    throw new Error(`Failed to upload video: ${error.message}`)
  }
}

/**
 * Delete a video and its associated data
 */
export async function deleteVideo(videoId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}${VIDEOS_ENDPOINT}/${videoId}`, {
      method: 'DELETE'
    })

    if (!response.ok) {
      throw new Error(`Failed to delete video: ${response.status}`)
    }

  } catch (error) {
    console.error('Delete video API error:', error)
    throw new Error(`Failed to delete video: ${error.message}`)
  }
}

/**
 * Health check for the API services
 */
export async function checkApiHealth(): Promise<{ ingest: boolean, search: boolean }> {
  try {
    const ingestPromise = fetch(`${API_BASE_URL}/health/ingest`).catch(() => null)
    const searchPromise = fetch(`${API_BASE_URL}/health/search`).catch(() => null)
    
    const [ingestResponse, searchResponse] = await Promise.all([ingestPromise, searchPromise])

    return {
      ingest: ingestResponse?.ok || false,
      search: searchResponse?.ok || false
    }

  } catch (error) {
    console.error('Health check error:', error)
    return { ingest: false, search: false }
  }
}

/**
 * Get system status and metrics
 */
export async function getSystemStatus(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/status`)
    
    if (!response.ok) {
      throw new Error(`Failed to get system status: ${response.status}`)
    }

    return await response.json()

  } catch (error) {
    console.error('System status API error:', error)
    throw new Error(`Failed to get system status: ${error.message}`)
  }
}

/**
 * Enhanced search for video segments with query improvement
 */
export async function enhancedSearchVideoSegments(
  query: string, 
  options: { 
    topK?: number, 
    threshold?: number, 
    useLLMEnhancement?: boolean,
    searchWeights?: { visual?: number, audio?: number, text?: number, metadata?: number }
  } = {}
): Promise<SearchResult & { originalQuery: string, enhancedQuery?: string, queryAnalysis?: any }> {
  const { topK = 20, threshold = 0.5, useLLMEnhancement = true, searchWeights } = options
  
  const searchRequest = {
    query: query.trim(),
    top_k: topK,
    threshold,
    use_llm_enhancement: useLLMEnhancement,
    search_weights: searchWeights
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/search/enhanced`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(searchRequest),
    })

    if (!response.ok) {
      throw new Error(`Enhanced search failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    
    // Transform backend response to frontend format
    const transformedResults: VideoSegment[] = data.results?.map((result: any, index: number) => ({
      id: result.id || `${result.video_id}_${index}`,
      video_id: result.video_id,
      videoUrl: result.videoUrl || `/api/videos/${result.video_id}/stream`,
      thumbnailUrl: result.thumbnailUrl || result.thumbnail_url || `/api/thumbnails/${result.video_id}/${Math.floor(result.start_time)}.jpg`,
      startTime: result.start_time || result.startTime,
      endTime: result.end_time || result.endTime,
      title: result.title || `Video ${result.video_id}`,
      description: result.description || 'No description available',
      transcript: result.transcript,
      score: result.score || 0,
      enhancedQueryUsed: result.enhanced_query_used
    })) || []

    return {
      results: transformedResults,
      query_time_ms: data.query_time_ms || 0,
      total_results: data.total_results || transformedResults.length,
      originalQuery: data.original_query || query,
      enhancedQuery: data.enhanced_query,
      queryAnalysis: data.query_analysis
    }

  } catch (error) {
    console.error('Enhanced search API error:', error)
    throw new Error(`Failed to perform enhanced search: ${error.message}`)
  }
}

/**
 * Enhance a query for better search results
 */
export async function enhanceQuery(
  query: string,
  options: { context?: any, useLLM?: boolean } = {}
): Promise<{
  originalQuery: string,
  enhancedQuery: string,
  queryType: string,
  entities: string[],
  actions: string[],
  sceneContext: string[],
  temporalContext: string[],
  confidence: number
}> {
  const { context, useLLM = true } = options
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/enhance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query.trim(),
        context,
        use_llm: useLLM
      }),
    })

    if (!response.ok) {
      throw new Error(`Query enhancement failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    
    return {
      originalQuery: data.original_query,
      enhancedQuery: data.enhanced_query,
      queryType: data.query_type,
      entities: data.entities || [],
      actions: data.actions || [],
      sceneContext: data.scene_context || [],
      temporalContext: data.temporal_context || [],
      confidence: data.confidence || 0
    }

  } catch (error) {
    console.error('Query enhancement API error:', error)
    throw new Error(`Failed to enhance query: ${error.message}`)
  }
}

/**
 * Get query enhancement statistics
 */
export async function getQueryEnhancementStats(): Promise<{
  enhancementStats: any,
  status: string
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/query/stats`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to get stats: ${response.status} ${response.statusText}`)
    }

    return await response.json()

  } catch (error) {
    console.error('Query stats API error:', error)
    throw new Error(`Failed to get enhancement stats: ${error.message}`)
  }
}
