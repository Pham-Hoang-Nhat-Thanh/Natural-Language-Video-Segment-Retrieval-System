'use client'

import { useState } from 'react'

interface SearchResult {
  id: string
  video_path: string
  start_time: number
  end_time: number
  score: number
  thumbnail_path?: string
}

const EXAMPLE_QUERIES = ['dog playing fetch', 'person cooking', 'sunset scene', 'car driving']

export default function HomePage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedVideo, setSelectedVideo] = useState<SearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return
    
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`http://localhost:8090/search?query=${encodeURIComponent(query)}&top_k=20`)
      if (!response.ok) {
        throw new Error(`Search failed: ${response.status} ${response.statusText}`)
      }
      const data = await response.json()
      setResults(data.results || [])
    } catch (error) {
      console.error('Search failed:', error)
      setError(error instanceof Error ? error.message : 'Search failed. Please try again.')
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Video Search System</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Search Section */}
        <div className="mb-8">
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4 text-center">Search Videos</h2>
              
              {/* Search Input */}
              <div className="flex gap-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Describe what you're looking for..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={handleSearch}
                  disabled={isLoading || !query.trim()}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                >
                  {isLoading ? 'Searching...' : 'Search'}
                </button>
              </div>

              {/* Example Queries */}
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_QUERIES.map((example) => (
                    <button
                      key={example}
                      onClick={() => setQuery(example)}
                      className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="mb-8">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">Search Results ({results.length})</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {results.map((result, index) => (
                <div
                  key={`${result.id}-${index}`}
                  className="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => setSelectedVideo(result)}
                >
                  <div className="p-4">
                    <div className="aspect-video bg-gray-200 rounded mb-3 flex items-center justify-center">
                      <span className="text-gray-500 text-sm">Video Thumbnail</span>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Video Segment</p>
                      <p className="text-xs text-gray-600">
                        Time: {result.start_time}s - {result.end_time}s
                      </p>
                      <p className="text-xs text-gray-600">
                        Score: {(result.score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Video Player Modal */}
        {selectedVideo && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
              <div className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-semibold">Video Player</h3>
                  <button
                    onClick={() => setSelectedVideo(null)}
                    className="text-gray-500 hover:text-gray-700 text-2xl"
                  >
                    ×
                  </button>
                </div>
                
                <div className="aspect-video bg-gray-900 rounded mb-4 flex items-center justify-center">
                  <p className="text-white">Video Player Area</p>
                  <p className="text-white text-sm ml-2">
                    ({selectedVideo.start_time}s - {selectedVideo.end_time}s)
                  </p>
                </div>
                
                <div className="text-sm text-gray-600">
                  <p>File: {selectedVideo.video_path}</p>
                  <p>Segment: {selectedVideo.start_time}s - {selectedVideo.end_time}s</p>
                  <p>Relevance Score: {(selectedVideo.score * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Searching videos...</p>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && results.length === 0 && query && (
          <div className="text-center py-8">
            <p className="text-gray-600">No results found for "{query}"</p>
            <p className="text-sm text-gray-500 mt-1">Try different search terms</p>
          </div>
        )}
      </main>
    </div>
  )
}
