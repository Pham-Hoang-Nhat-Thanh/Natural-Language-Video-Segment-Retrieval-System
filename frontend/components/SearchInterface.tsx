'use client'

import React, { useState, ReactNode } from 'react'
import { useQuery } from 'react-query'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import { Search, Loader2, Clock, Play } from 'lucide-react'
import { SearchResult, SearchResponse } from '@/types'
import { searchVideos } from '@/lib/api'
import SearchBar from './SearchBar'
import ResultsList from './ResultsList'

interface SearchInterfaceProps {
  onResultSelect: (result: SearchResult) => void
}

export default function SearchInterface({ onResultSelect }: SearchInterfaceProps) {
  const [query, setQuery] = useState('')
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const [recentQueries] = useState<string[]>([
    'dog playing fetch',
    'person cooking pasta',
    'sunset over mountains',
    'children laughing',
    'car driving fast'
  ])

  const {
    data: searchResults,
    isLoading,
    error,
    refetch
  } = useQuery<SearchResponse>(
    ['search', query],
    () => searchVideos({ query, top_k: 20 }),
    {
      enabled: query.length > 2,
      onError: (error: any) => {
        toast.error(error.message || 'Search failed')
      },
      onSuccess: (data) => {
        if (data.results.length === 0) {
          toast('No results found. Try a different search term.', {
            icon: '🔍'
          })
        } else {
          toast.success(`Found ${data.results.length} results in ${data.query_time_ms}ms`)
        }
      }
    }
  )

  const handleSearch = (searchQuery: string) => {
    if (searchQuery.trim().length < 3) {
      toast.error('Please enter at least 3 characters')
      return
    }

    setQuery(searchQuery.trim())
    
    // Add to search history
    setSearchHistory(prev => {
      const updated = [searchQuery, ...prev.filter(q => q !== searchQuery)]
      return updated.slice(0, 10) // Keep last 10 searches
    })
  }

  const handleSuggestionClick = (suggestion: string) => {
    handleSearch(suggestion)
  }

  return (
    <div className="w-full">
      {/* Search Bar */}
      <SearchBar
        onSearch={handleSearch}
        isLoading={isLoading}
        placeholder="Describe what you're looking for... (e.g., 'dog playing in park')"
      />

      {/* Search Suggestions */}
      {/* @ts-ignore */}
      {!query && (
        <div className="mt-6">
          <div className="text-center mb-4">
            <h3 className="text-lg font-medium text-foreground mb-2">
              Try these examples:
            </h3>
          </div>
          
          <div className="flex flex-wrap justify-center gap-2">
            {recentQueries.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                className="px-4 py-2 text-sm bg-white border border-border rounded-full hover:border-primary hover:text-primary transition-colors focus-ring"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search History */}
      {/* @ts-ignore */}
      {searchHistory.length > 0 && !query && (
        <div className="mt-8">
          <div className="flex items-center gap-2 mb-4">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <h3 className="text-sm font-medium text-muted-foreground">
              Recent Searches
            </h3>
          </div>
          
          <div className="flex flex-wrap gap-2">
            {searchHistory.slice(0, 5).map((historyQuery, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(historyQuery)}
                className="px-3 py-1 text-sm text-muted-foreground bg-muted rounded-lg hover:bg-muted/80 transition-colors"
              >
                {historyQuery}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading State */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mt-8 text-center"
          >
            <div className="inline-flex items-center gap-2 text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Searching through videos...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error State */}
      {error && !isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 p-4 bg-destructive/10 border border-destructive/20 rounded-lg"
        >
          <div className="flex items-center gap-2 text-destructive">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 15.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span className="font-medium">Search Error</span>
          </div>
          <p className="mt-1 text-sm text-destructive/80">
            {(error as any)?.message || 'Failed to search videos. Please try again.'}
          </p>
          <button
            onClick={() => refetch()}
            className="mt-2 px-3 py-1 text-sm bg-destructive text-destructive-foreground rounded hover:bg-destructive/90 transition-colors"
          >
            Retry Search
          </button>
        </motion.div>
      )}

      {/* Search Results */}
      <AnimatePresence>
        {searchResults && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mt-8"
          >
            {/* Results Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Search className="w-5 h-5 text-primary" />
                <h2 className="text-xl font-semibold">
                  Search Results for "{query}"
                </h2>
              </div>
              
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <span>
                  {searchResults.total_results} results
                </span>
                <span>
                  {searchResults.query_time_ms}ms
                  {searchResults.cached && (
                    <span className="ml-1 text-primary">(cached)</span>
                  )}
                </span>
              </div>
            </div>

            {/* Results List */}
            <ResultsList
              results={searchResults.results}
              onResultSelect={onResultSelect}
              query={query}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {searchResults && searchResults.results.length === 0 && !isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 text-center py-12"
        >
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
            <Search className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium mb-2">No results found</h3>
          <p className="text-muted-foreground mb-4">
            Try adjusting your search terms or browse our example queries above.
          </p>
          <button
            onClick={() => setQuery('')}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Clear Search
          </button>
        </motion.div>
      )}
    </div>
  )
}
