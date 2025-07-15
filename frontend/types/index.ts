/**
 * Type definitions for video segment retrieval system
 */

export interface VideoSegment {
  id: string
  video_id: string
  videoUrl: string
  thumbnailUrl: string
  startTime: number
  endTime: number
  title: string
  description: string
  transcript?: string
  score: number
  enhancedQueryUsed?: string
}

export interface SearchResult {
  results: VideoSegment[]
  query_time_ms: number
  total_results: number
  originalQuery?: string
  enhancedQuery?: string
  queryAnalysis?: QueryAnalysis
}

export interface SearchRequest {
  query: string
  top_k?: number
  threshold?: number
  use_query_enhancement?: boolean
}

export interface EnhancedSearchRequest {
  query: string
  top_k?: number
  threshold?: number
  use_llm_enhancement?: boolean
  search_weights?: SearchWeights
}

export interface SearchWeights {
  visual?: number
  audio?: number
  text?: number
  metadata?: number
}

export interface QueryAnalysis {
  type: string
  entities: string[]
  actions: string[]
  scene_context: string[]
  temporal_context: string[]
  confidence: number
}

export interface QueryEnhancement {
  originalQuery: string
  enhancedQuery: string
  queryType: string
  entities: string[]
  actions: string[]
  sceneContext: string[]
  temporalContext: string[]
  confidence: number
}

/**
 * Legacy types for backward compatibility
 */

export interface LegacySearchResult {
  video_id: string
  video_path: string
  segments: LegacyVideoSegment[]
  score: number
}

export interface LegacyVideoSegment {
  keyframe_id: string
  start_time: number
  end_time: number
  score: number
  thumbnail_path: string
  timestamp: number
}

export interface VideoPlayerSegment {
  id: string
  start: number
  end: number
  title: string
  thumbnail?: string
}

export interface LegacyVideoPlayerProps {
  src: string
  segments: VideoPlayerSegment[]
}

export interface VideoPlayerData {
  src: string
  segments: VideoPlayerSegment[]
}
