/**
 * SegmentItem Component
 * 
 * Displays a single video segment result with:
 * - Thumbnail image
 * - Video title and segment timing
 * - Description/transcript snippet
 * - Hover/focus states
 * - Active state indication
 * - Optional relevance score display
 */

'use client'

import React from 'react'
import { SegmentItemProps } from '@/types'

/**
 * Formats seconds to MM:SS format
 */
function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}

/**
 * Truncates text to specified length with ellipsis
 */
function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength).trim() + '...'
}

export default function SegmentItem({
  segment,
  isActive = false,
  onClick,
  showScore = false
}: SegmentItemProps) {
  const handleClick = () => {
    onClick(segment)
  }

  const handleKeyDown = (e: any) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onClick(segment)
    }
  }

  return (
    <div
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="button"
      aria-label={`Play video segment: ${segment.title} at ${formatTime(segment.startTime)}`}
      className={`
        group relative bg-white rounded-lg shadow-sm border-2 
        cursor-pointer transition-all duration-200 ease-in-out
        hover:shadow-md hover:scale-[1.02] hover:border-blue-300
        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
        ${isActive 
          ? 'border-blue-500 bg-blue-50 shadow-md scale-[1.02]' 
          : 'border-gray-200 hover:border-gray-300'
        }
      `}
    >
      {/* Active Indicator Badge */}
      {isActive && (
        <div className="absolute top-2 right-2 z-10">
          <div className="flex items-center space-x-1 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
            <span>Playing</span>
          </div>
        </div>
      )}

      {/* Thumbnail Container */}
      <div className="relative aspect-video w-full overflow-hidden rounded-t-lg bg-gray-100">
        {segment.thumbnailUrl ? (
          <img
            src={segment.thumbnailUrl}
            alt={`Thumbnail for ${segment.title}`}
            className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
            loading="lazy"
          />
        ) : (
          // Placeholder thumbnail
          <div className="w-full h-full flex items-center justify-center bg-gray-200 text-gray-400">
            <span className="text-sm">No Preview</span>
          </div>
        )}

        {/* Play Button Overlay */}
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-white bg-opacity-90 rounded-full p-3 shadow-lg">
              <span className="text-blue-600 font-bold">▶</span>
            </div>
          </div>
        </div>

        {/* Time Badge */}
        <div className="absolute bottom-2 left-2">
          <div className="bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
            {formatTime(segment.startTime)}
          </div>
        </div>
      </div>

      {/* Content Section */}
      <div className="p-4">
        {/* Title */}
        <h3 className="font-semibold text-gray-900 text-sm mb-1 line-clamp-2">
          {segment.title}
        </h3>

        {/* Timing Info */}
        <div className="flex items-center text-xs text-gray-500 mb-2">
          <span>
            {formatTime(segment.startTime)} - {formatTime(segment.endTime)}
            <span className="text-gray-400 ml-1">
              ({Math.round(segment.endTime - segment.startTime)}s)
            </span>
          </span>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-700 line-clamp-3 mb-2">
          {truncateText(segment.description || segment.transcript || 'No description available', 120)}
        </p>

        {/* Score (if enabled) */}
        {showScore && (
          <div className="flex items-center justify-between">
            <div className="flex items-center text-xs text-gray-500">
              <span className="font-medium">Relevance:</span>
              <div className="ml-2 flex items-center">
                <div className="w-16 bg-gray-200 rounded-full h-1.5">
                  <div 
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(segment.score * 100, 100)}%` }}
                  />
                </div>
                <span className="ml-2 text-xs font-mono">
                  {(segment.score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
