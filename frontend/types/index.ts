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
}

export interface SearchResult {
  results: VideoSegment[]
  query_time_ms: number
  total_results: number
}

export interface SearchRequest {
  query: string
  top_k?: number
  threshold?: number
}

export interface VideoPlayerProps {
  videoUrl?: string
  currentSegment?: VideoSegment | null
  autoplay?: boolean
  onTimeUpdate?: (currentTime: number) => void
  onLoadedMetadata?: (duration: number) => void
}

export interface SearchBarProps {
  onSearch: (query: string) => void
  isLoading?: boolean
  placeholder?: string
  autoFocus?: boolean
}

export interface SegmentItemProps {
  segment: VideoSegment
  isActive?: boolean
  onClick: (segment: VideoSegment) => void
  showScore?: boolean
}

export interface SegmentListProps {
  segments: VideoSegment[]
  activeSegmentId?: string | null
  onSegmentClick: (segment: VideoSegment) => void
  isLoading?: boolean
  showScores?: boolean
}

// Legacy types for backward compatibility
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
