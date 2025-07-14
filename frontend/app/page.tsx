/**
 * Video Segment Retrieval Main Page
 * 
 * A modern interface for natural language video search with:
 * - Prominent search bar with real-time search
 * - Grid layout of video segment results
 * - Integrated video player with auto-seek
 * - Responsive design and accessibility features
 */

'use client'

import { useState, useCallback } from 'react'
import SearchBar from '@/components/SearchBar'
import SegmentList from '@/components/SegmentList'
import VideoPlayer from '@/components/VideoPlayer'
import { VideoSegment, SearchResult } from '@/types'
import { searchVideoSegments } from '@/lib/api'

// Mock data for development - replace with real API calls
const mockSegments: VideoSegment[] = [
  {
    id: '1',
    video_id: 'vid_001',
    videoUrl: '/api/videos/sample.mp4',
    thumbnailUrl: '/api/thumbnails/thumb_001.jpg',
    startTime: 25.5,
    endTime: 45.2,
    title: 'Introduction to Machine Learning',
    description: 'An overview of machine learning concepts including supervised and unsupervised learning paradigms.',
    transcript: 'Machine learning is a subset of artificial intelligence that focuses on algorithms...',
    score: 0.95
  },
  {
    id: '2',
    video_id: 'vid_001',
    videoUrl: '/api/videos/sample.mp4',
    thumbnailUrl: '/api/thumbnails/thumb_002.jpg',
    startTime: 120.8,
    endTime: 145.3,
    title: 'Neural Network Architectures',
    description: 'Deep dive into different neural network architectures and their applications.',
    transcript: 'Neural networks are computational models inspired by biological neural networks...',
    score: 0.87
  },
  {
    id: '3',
    video_id: 'vid_002',
    videoUrl: '/api/videos/sample2.mp4',
    thumbnailUrl: '/api/thumbnails/thumb_003.jpg',
    startTime: 67.2,
    endTime: 89.5,
    title: 'Data Preprocessing Techniques',
    description: 'Essential preprocessing steps for preparing data for machine learning models.',
    transcript: 'Data preprocessing is a crucial step in the machine learning pipeline...',
    score: 0.82
  }
]

export default function VideoRetrievalPage() {
  const [segments, setSegments] = useState([])
  const [currentSegment, setCurrentSegment] = useState(null)
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [showScores, setShowScores] = useState(false)

  /**
   * Handle search query submission
   * Calls the real search API backend
   */
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) return

    setIsSearching(true)
    setHasSearched(true)

    try {
      // Call the real search API
      const searchResult = await searchVideoSegments(query, {
        topK: 20,
        threshold: 0.3
      })

      setSegments(searchResult.results)

    } catch (error) {
      console.error('Search failed:', error)
      
      // Fallback to mock data for demo purposes
      const filteredSegments = mockSegments.filter(segment =>
        segment.title.toLowerCase().includes(query.toLowerCase()) ||
        segment.description.toLowerCase().includes(query.toLowerCase()) ||
        segment.transcript?.toLowerCase().includes(query.toLowerCase())
      )

      setSegments(filteredSegments)
    } finally {
      setIsSearching(false)
    }
  }, [])

  /**
   * Handle segment selection for video playback
   */
  const handleSegmentClick = useCallback((segment: VideoSegment) => {
    setCurrentSegment(segment)
  }, [])

  /**
   * Handle video time updates for segment tracking
   */
  const handleVideoTimeUpdate = useCallback((currentTime: number) => {
    // Optional: Update UI based on current playback time
    // Could be used to highlight current segment in timeline
  }, [])

  /**
   * Handle video metadata loading
   */
  const handleVideoLoadedMetadata = useCallback((duration: number) => {
    // Optional: Store video duration for UI enhancements
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Search Section */}
        <section className="mb-8">
          <div className="text-center mb-6">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Find Video Segments with Natural Language
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Search through video content using natural language queries. 
              Click any result to jump to that segment and start playing.
            </p>
            
            {/* Debug Toggle */}
            <div className="mt-4 flex justify-center">
              <label className="flex items-center text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={showScores}
                  onChange={(e) => setShowScores(e.target.checked)}
                  className="mr-2"
                />
                Show relevance scores (debug mode)
              </label>
            </div>
          </div>
          
          <SearchBar
            onSearch={handleSearch}
            isLoading={isSearching}
            placeholder="Search video segments... (e.g., 'machine learning concepts', 'data preprocessing')"
            autoFocus={true}
          />
        </section>

        {/* Video Player Section */}
        {currentSegment && (
          <section className="mb-8">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Now Playing
                </h3>
                <p className="text-gray-600">
                  {currentSegment.title} • Starting at {Math.floor(currentSegment.startTime / 60)}:{String(Math.floor(currentSegment.startTime % 60)).padStart(2, '0')}
                </p>
              </div>
              
              <VideoPlayer
                videoUrl={currentSegment.videoUrl}
                currentSegment={currentSegment}
                autoplay={true}
                onTimeUpdate={handleVideoTimeUpdate}
                onLoadedMetadata={handleVideoLoadedMetadata}
              />
            </div>
          </section>
        )}

        {/* Results Section */}
        {hasSearched && (
          <section>
            <SegmentList
              segments={segments}
              activeSegmentId={currentSegment?.id || null}
              onSegmentClick={handleSegmentClick}
              isLoading={isSearching}
              showScores={showScores}
            />
          </section>
        )}

        {/* Welcome State */}
        {!hasSearched && (
          <section className="text-center py-16">
            <div className="max-w-md mx-auto">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Start Your Search
              </h3>
              <p className="text-gray-600 mb-6">
                Enter a search query above to find relevant video segments. 
                You can search for topics, concepts, or specific content.
              </p>
              <div className="space-y-2 text-sm text-gray-500">
                <p><strong>Example queries:</strong></p>
                <p>"machine learning introduction"</p>
                <p>"data preprocessing steps"</p>
                <p>"neural network architecture"</p>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-500">
            <p>
              Video Segment Retrieval System • 
              <span className="ml-1">Powered by CLIP and FastAPI</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
