/**
 * SearchBar Component
 * 
 * A prominent text input for video segment search with:
 * - Debounced search on typing
 * - Keyboard submission support (Enter key)
 * - Loading state indication
 * - Accessibility features
 */

'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { SearchBarProps } from '@/types'

// Custom debounce hook
function useDebounce(value: string, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [value, delay])

  return debouncedValue
}

export default function SearchBar({
  onSearch,
  isLoading = false,
  placeholder = "Search video segments...",
  autoFocus = true
}: SearchBarProps) {
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounce(query, 300) // 300ms debounce

  // Trigger search when debounced query changes
  useEffect(() => {
    if (debouncedQuery.trim()) {
      onSearch(debouncedQuery.trim())
    }
  }, [debouncedQuery, onSearch])

  const handleSubmit = useCallback((e: any) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query.trim())
    }
  }, [query, onSearch])

  const handleInputChange = useCallback((e: any) => {
    setQuery(e.target.value)
  }, [])

  const handleClear = useCallback(() => {
    setQuery('')
  }, [])

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          {/* Main Input */}
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            placeholder={placeholder}
            autoFocus={autoFocus}
            disabled={isLoading}
            className={`
              w-full h-14 pl-4 pr-20 text-lg 
              border-2 border-gray-300 rounded-xl
              focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none
              placeholder-gray-500 bg-white shadow-sm
              transition-all duration-200 ease-in-out
              ${isLoading ? 'opacity-75 cursor-not-allowed' : 'hover:border-gray-400'}
            `}
            aria-label="Search video segments"
            aria-describedby="search-help"
          />

          {/* Clear Button */}
          {query && !isLoading && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute inset-y-0 right-12 flex items-center pr-2 text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Clear search"
            >
              ×
            </button>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="absolute inset-y-0 right-4 flex items-center">
              <span className="text-sm text-blue-500">Searching...</span>
            </div>
          )}

          {/* Search Button (optional, mainly for accessibility) */}
          {!isLoading && (
            <button
              type="submit"
              disabled={!query.trim()}
              className={`
                absolute inset-y-0 right-2 px-4 
                text-sm font-medium rounded-lg
                transition-all duration-200 ease-in-out
                ${query.trim() 
                  ? 'text-blue-600 hover:bg-blue-50 hover:text-blue-700' 
                  : 'text-gray-400 cursor-not-allowed'
                }
              `}
              aria-label="Submit search"
            >
              Search
            </button>
          )}
        </div>
      </form>

      {/* Help Text */}
      <p id="search-help" className="mt-2 text-sm text-gray-600 text-center">
        Type to search video segments in real-time, or press Enter to search
      </p>
    </div>
  )
}
