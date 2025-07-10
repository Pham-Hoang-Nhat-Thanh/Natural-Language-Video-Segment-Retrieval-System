'use client'

import { motion } from 'framer-motion'
import { Play, Clock, ExternalLink } from 'lucide-react'
import { SearchResult } from '@/types'
import { formatDuration, formatScore } from '@/lib/utils'

interface ResultsListProps {
  results: SearchResult[]
  onResultSelect: (result: SearchResult) => void
  query: string
}

export default function ResultsList({ results, onResultSelect, query }: ResultsListProps) {
  const handleResultClick = (result: SearchResult) => {
    onResultSelect(result)
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {results.map((result, index) => (
        <motion.div
          key={`${result.video_id}-${result.start_time}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="result-card bg-white rounded-lg border border-border overflow-hidden cursor-pointer group"
          onClick={() => handleResultClick(result)}
        >
          {/* Thumbnail */}
          <div className="relative aspect-video bg-muted overflow-hidden">
            {result.thumbnail_url ? (
              <img
                src={result.thumbnail_url}
                alt={result.title || 'Video thumbnail'}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                loading="lazy"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                <Play className="w-8 h-8 text-muted-foreground" />
              </div>
            )}
            
            {/* Play Overlay */}
            <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
              <div className="w-12 h-12 bg-white/90 rounded-full flex items-center justify-center">
                <Play className="w-6 h-6 text-gray-900 ml-0.5" fill="currentColor" />
              </div>
            </div>
            
            {/* Score Badge */}
            <div className="absolute top-2 right-2">
              <span className={`px-2 py-1 rounded text-xs font-medium ${getScoreColor(result.score)}`}>
                {formatScore(result.score)}
              </span>
            </div>
            
            {/* Duration Badge */}
            <div className="absolute bottom-2 right-2 bg-black/80 text-white px-2 py-1 rounded text-xs flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatDuration(result.end_time - result.start_time)}
            </div>
          </div>

          {/* Content */}
          <div className="p-4">
            {/* Title */}
            <h3 className="font-medium text-foreground mb-2 line-clamp-2 group-hover:text-primary transition-colors">
              {result.title || `Video ${result.video_id}`}
            </h3>
            
            {/* Description */}
            {result.description && (
              <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                {result.description}
              </p>
            )}
            
            {/* Metadata */}
            <div className="space-y-2">
              {/* Time Range */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Clock className="w-3 h-3" />
                <span>
                  {formatDuration(result.start_time)} - {formatDuration(result.end_time)}
                </span>
              </div>
              
              {/* Channel */}
              {result.channel && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <ExternalLink className="w-3 h-3" />
                  <span>{result.channel}</span>
                </div>
              )}
              
              {/* Video ID */}
              <div className="text-xs text-muted-foreground font-mono">
                ID: {result.video_id}
              </div>
            </div>
            
            {/* Action Button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleResultClick(result)
              }}
              className="mt-3 w-full px-3 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4" />
              Play Segment
            </button>
          </div>
        </motion.div>
      ))}
    </div>
  )
}

// Utility component for line clamping (if not using Tailwind)
const LineClamp = ({ children, lines = 2, className = '' }: { 
  children: React.ReactNode
  lines?: number
  className?: string 
}) => (
  <div 
    className={className}
    style={{
      display: '-webkit-box',
      WebkitLineClamp: lines,
      WebkitBoxOrient: 'vertical',
      overflow: 'hidden'
    }}
  >
    {children}
  </div>
)
