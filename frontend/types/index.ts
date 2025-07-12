// Type definitions for the video retrieval system

export interface SearchResult {
  video_id: string
  video_path: string
  segments: VideoSegment[]
  score: number
}

export interface VideoSegment {
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

export interface VideoPlayerProps {
  src: string
  segments: VideoPlayerSegment[]
}

export interface VideoPlayerData {
  src: string
  segments: VideoPlayerSegment[]
}
