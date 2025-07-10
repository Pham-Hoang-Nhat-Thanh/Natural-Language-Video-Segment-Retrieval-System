export interface SearchResult {
  video_id: string
  start_time: number
  end_time: number
  score: number
  thumbnail_url: string
  title: string
  description?: string
  video_url?: string
  duration?: number
  channel?: string
}

export interface SearchResponse {
  results: SearchResult[]
  query_time_ms: number
  total_results: number
  cached?: boolean
}

export interface SearchRequest {
  query: string
  top_k?: number
  threshold?: number
}

export interface VideoMetadata {
  video_id: string
  title: string
  description: string
  duration: number
  fps: number
  width: number
  height: number
  file_size: number
  processed: boolean
  channel?: string
  tags?: string[]
}

export interface Shot {
  shot_id: number
  start_frame: number
  end_frame: number
  start_time: number
  end_time: number
  confidence: number
}

export interface Keyframe {
  keyframe_id: string
  shot_id: number
  frame_number: number
  timestamp: number
  image_path: string
  thumbnail_url?: string
}

export interface IngestionStatus {
  video_id: string
  status: 'processing' | 'completed' | 'failed' | 'pending'
  progress?: number
  error_message?: string
  created_at: string
  processing_start_time?: string
  processing_end_time?: string
  shots_detected?: number
  keyframes_extracted?: number
  embeddings_generated?: number
}

export interface AdminStats {
  total_videos: number
  completed_videos: number
  failed_videos: number
  processing_videos: number
  total_shots: number
  total_keyframes: number
  avg_processing_time: number
  avg_query_time: number
  cache_hit_rate: number
  system_health: {
    ingest_service: 'up' | 'down'
    search_service: 'up' | 'down'
    database: 'up' | 'down'
    redis: 'up' | 'down'
  }
}

export interface PerformanceMetrics {
  timestamp: string
  query_latency_ms: number
  throughput_rps: number
  cache_hit_rate: number
  error_rate: number
  active_users: number
}

export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'user'
  created_at: string
  last_login?: string
}

export interface APIError {
  error: string
  message?: string
  details?: any
}

// API Response wrapper
export interface APIResponse<T> {
  data?: T
  error?: APIError
  success: boolean
}
