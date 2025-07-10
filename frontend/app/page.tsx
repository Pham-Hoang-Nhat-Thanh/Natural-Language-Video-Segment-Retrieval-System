'use client'

import { useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from 'react-query'
import SearchInterface from '@/components/SearchInterface'
import VideoPlayer from '@/components/VideoPlayer'
import Header from '@/components/Header'
import { SearchResult } from '@/types'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

export default function HomePage() {
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [playerKey, setPlayerKey] = useState(0)

  const handleResultSelect = (result: SearchResult) => {
    setSelectedResult(result)
    setPlayerKey(prev => prev + 1) // Force re-render of player
  }

  const handleClosePlayer = () => {
    setSelectedResult(null)
  }

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen">
        <Header />
        
        <main className="container mx-auto px-4 py-8">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 gradient-text">
              Find Any Video Moment
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
              Search through hours of video content using natural language. 
              Describe what you're looking for and get precise video segments instantly.
            </p>
          </div>

          {/* Search Interface */}
          <div className="max-w-4xl mx-auto mb-8">
            <SearchInterface onResultSelect={handleResultSelect} />
          </div>

          {/* Video Player */}
          {selectedResult && (
            <div className="max-w-6xl mx-auto">
              <VideoPlayer
                key={playerKey}
                result={selectedResult}
                onClose={handleClosePlayer}
              />
            </div>
          )}

          {/* Features Section */}
          <div className="mt-20 grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="text-center p-6 rounded-lg bg-white/50 backdrop-blur">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
              <p className="text-muted-foreground">
                Get results in under 50ms with our optimized search pipeline
              </p>
            </div>

            <div className="text-center p-6 rounded-lg bg-white/50 backdrop-blur">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">AI-Powered</h3>
              <p className="text-muted-foreground">
                Advanced CLIP models understand natural language queries
              </p>
            </div>

            <div className="text-center p-6 rounded-lg bg-white/50 backdrop-blur">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Precise Segments</h3>
              <p className="text-muted-foreground">
                Get exact timestamps for the moments you're looking for
              </p>
            </div>
          </div>
        </main>
      </div>
    </QueryClientProvider>
  )
}
