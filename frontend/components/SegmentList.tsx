/**
 * SegmentList Component
 * 
 * Displays a grid/list of video segments with:
 * - Responsive grid layout
 * - Loading states
 * - Empty states
 * - Keyboard navigation support
 * - Sorting by relevance score
 */

'use client'

import React from 'react'
import SegmentItem from './SegmentItem'
import { SegmentListProps } from '@/types'

export default function SegmentList({
  segments,
  activeSegmentId = null,
  onSegmentClick,
  isLoading = false,
  showScores = false
}: SegmentListProps) {
  
  // Sort segments by score (highest first)
  const sortedSegments = [...segments].sort((a, b) => b.score - a.score)

  if (isLoading) {
    return (
      <div className="w-full">
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Searching video segments...</p>
          </div>
        </div>
        
        {/* Loading Skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, index) => (
            <div key={index} className="animate-pulse">
              <div className="bg-gray-200 rounded-lg aspect-video mb-4"></div>
              <div className="space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                <div className="h-3 bg-gray-200 rounded w-full"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (segments.length === 0) {
    return (
      <div className="w-full">
        <div className="text-center py-16">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            No video segments found
          </h3>
          <p className="text-gray-600 max-w-md mx-auto">
            Try adjusting your search query or check if videos have been processed and indexed.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      {/* Results Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Search Results
          </h2>
          <span className="bg-blue-100 text-blue-800 text-sm font-medium px-3 py-1 rounded-full">
            {segments.length} segment{segments.length !== 1 ? 's' : ''} found
          </span>
        </div>
        
        {showScores && (
          <div className="flex items-center text-sm text-gray-600">
            <span>Sorted by relevance</span>
          </div>
        )}
      </div>

      {/* Results Grid */}
      <div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
        role="list"
        aria-label={`${segments.length} video segment results`}
      >
        {sortedSegments.map((segment, index) => (
          <div 
            key={segment.id} 
            role="listitem"
            className="focus-within:ring-2 focus-within:ring-blue-500 rounded-lg"
          >
            <SegmentItem
              segment={segment}
              isActive={activeSegmentId === segment.id}
              onClick={onSegmentClick}
              showScore={showScores}
            />
            
            {/* Screen Reader Helper */}
            <span className="sr-only">
              Result {index + 1} of {segments.length}
            </span>
          </div>
        ))}
      </div>

      {/* Load More / Pagination could go here */}
      {segments.length > 12 && (
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-600">
            Showing {Math.min(segments.length, 12)} of {segments.length} results
          </p>
        </div>
      )}
    </div>
  )
}
