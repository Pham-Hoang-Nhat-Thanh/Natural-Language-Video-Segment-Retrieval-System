'use client'

import React, { useState, useRef, useEffect } from 'react'

export default function VideoPlayer({ src, segments }) {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [selectedSegment, setSelectedSegment] = useState(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime)
    }

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)

    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
    }
  }, [])

  const jumpToSegment = (segmentIndex) => {
    const video = videoRef.current
    if (!video || !segments[segmentIndex]) return

    const segment = segments[segmentIndex]
    video.currentTime = segment.start
    setSelectedSegment(segmentIndex)
    
    // Auto-play the segment
    video.play()
    
    // Stop at segment end
    const checkSegmentEnd = () => {
      if (video.currentTime >= segment.end) {
        video.pause()
        video.removeEventListener('timeupdate', checkSegmentEnd)
      }
    }
    
    video.addEventListener('timeupdate', checkSegmentEnd)
  }

  const togglePlayPause = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play()
    }
  }

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`
  }

  if (!segments || segments.length === 0) {
    return (
      <div className="text-center text-gray-500 p-8">
        No video segments available
      </div>
    )
  }

  return (
    <div className="w-full">
      {/* Video Player */}
      <div className="relative bg-black rounded-lg overflow-hidden mb-4">
        <video
          ref={videoRef}
          src={src}
          className="w-full h-64 object-contain"
          controls={false}
        />
        
        {/* Custom Controls Overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4">
          <div className="flex items-center justify-between text-white">
            <div className="text-sm">
              Segment {selectedSegment + 1}/{segments.length}: {formatTime(segments[selectedSegment]?.start)} - {formatTime(segments[selectedSegment]?.end)}
            </div>
            <button
              onClick={togglePlayPause}
              className="bg-white/20 hover:bg-white/30 rounded-full p-2"
            >
              {isPlaying ? '⏸️' : '▶️'}
            </button>
          </div>
        </div>
      </div>

      {/* Segment List */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="font-medium text-gray-900 mb-3">Video Segments</h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {segments.map((segment, index) => (
            <div
              key={segment.id || index}
              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                selectedSegment === index
                  ? 'bg-blue-100 border-blue-300 border'
                  : 'bg-white hover:bg-gray-100 border border-gray-200'
              }`}
              onClick={() => jumpToSegment(index)}
            >
              <div className="flex justify-between items-start mb-1">
                <span className="font-medium text-gray-900">
                  Segment {index + 1}
                </span>
                <span className="text-sm text-gray-500">
                  {formatTime(segment.start)} - {formatTime(segment.end)}
                </span>
              </div>
              
              {segment.title && (
                <p className="text-sm text-gray-600 mb-1">
                  {segment.title}
                </p>
              )}

              <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded inline-block">
                Click to jump
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
