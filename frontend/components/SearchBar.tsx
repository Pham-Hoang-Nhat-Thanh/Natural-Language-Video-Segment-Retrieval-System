'use client'

import { useState, useRef, useEffect } from 'react'
import { Search, Loader2, X } from 'lucide-react'
import { motion } from 'framer-motion'

interface SearchBarProps {
  onSearch: (query: string) => void
  isLoading?: boolean
  placeholder?: string
  initialValue?: string
}

export default function SearchBar({ 
  onSearch, 
  isLoading = false, 
  placeholder = "Search videos...",
  initialValue = ""
}: SearchBarProps) {
  const [query, setQuery] = useState(initialValue)
  const [isFocused, setIsFocused] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setQuery(initialValue)
  }, [initialValue])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isLoading) {
      onSearch(query.trim())
    }
  }

  const handleClear = () => {
    setQuery('')
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      inputRef.current?.blur()
    }
  }

  return (
    <div className="relative w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <motion.div
          animate={{
            scale: isFocused ? 1.02 : 1,
            boxShadow: isFocused 
              ? '0 10px 25px rgba(0,0,0,0.1)' 
              : '0 4px 6px rgba(0,0,0,0.05)'
          }}
          transition={{ duration: 0.2 }}
          className={`
            relative bg-white rounded-2xl border-2 transition-colors duration-200
            ${isFocused ? 'border-primary' : 'border-border'}
          `}
        >
          {/* Search Icon */}
          <div className="absolute left-4 top-1/2 transform -translate-y-1/2 z-10">
            {isLoading ? (
              <Loader2 className="w-5 h-5 text-primary animate-spin" />
            ) : (
              <Search className="w-5 h-5 text-muted-foreground" />
            )}
          </div>

          {/* Input Field */}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            className={`
              w-full h-14 pl-12 pr-20 text-lg bg-transparent border-none outline-none
              placeholder:text-muted-foreground
              disabled:cursor-not-allowed disabled:opacity-50
            `}
            autoComplete="off"
            spellCheck={false}
          />

          {/* Clear Button */}
          {query && !isLoading && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute right-12 top-1/2 transform -translate-y-1/2 p-1 rounded-full hover:bg-muted transition-colors"
              aria-label="Clear search"
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </button>
          )}

          {/* Search Button */}
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className={`
              absolute right-2 top-1/2 transform -translate-y-1/2 
              w-10 h-10 rounded-xl flex items-center justify-center
              transition-all duration-200
              ${query.trim() && !isLoading
                ? 'bg-primary text-white hover:bg-primary/90 shadow-lg' 
                : 'bg-muted text-muted-foreground cursor-not-allowed'
              }
            `}
            aria-label="Search"
          >
            <Search className="w-5 h-5" />
          </button>
        </motion.div>

        {/* Search Hints */}
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ 
            opacity: isFocused && !query ? 1 : 0,
            y: isFocused && !query ? 0 : 5
          }}
          className="absolute top-full mt-2 left-0 right-0 bg-white rounded-lg border shadow-lg p-3 z-20"
        >
          <div className="text-xs text-muted-foreground space-y-1">
            <p className="font-medium">Search tips:</p>
            <ul className="space-y-0.5">
              <li>• Describe actions: "person riding bicycle"</li>
              <li>• Mention objects: "red car in parking lot"</li>
              <li>• Include emotions: "happy children playing"</li>
              <li>• Add locations: "sunset over ocean"</li>
            </ul>
          </div>
        </motion.div>
      </form>

      {/* Loading Bar */}
      {isLoading && (
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: '100%' }}
          transition={{ duration: 2, ease: 'easeInOut' }}
          className="absolute bottom-0 left-0 h-1 bg-primary rounded-full"
        />
      )}
    </div>
  )
}
