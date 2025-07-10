'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Play, Pause, Volume2, VolumeX, Maximize, SkipBack, SkipForward } from 'lucide-react'
import { SearchResult } from '@/types'
import { formatDuration } from '@/lib/utils'

interface VideoPlayerProps {
  result: SearchResult
  onClose: () => void
}

export default function VideoPlayer({ result, onClose }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const controlsTimeoutRef = useRef<NodeJS.Timeout>()

  // Auto-hide controls
  useEffect(() => {
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
    
    controlsTimeoutRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false)
      }
    }, 3000)

    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current)
      }
    }
  }, [isPlaying, showControls])

  // Video event handlers
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedData = () => {
      setIsLoaded(true)
      setDuration(video.duration)
      // Jump to start time
      video.currentTime = result.start_time
    }

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime)
      
      // Auto-pause at end time
      if (video.currentTime >= result.end_time) {
        video.pause()
        setIsPlaying(false)
      }
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleVolumeChange = () => setIsMuted(video.muted)

    video.addEventListener('loadeddata', handleLoadedData)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('volumechange', handleVolumeChange)

    return () => {
      video.removeEventListener('loadeddata', handleLoadedData)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('volumechange', handleVolumeChange)
    }
  }, [result.start_time, result.end_time])

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!videoRef.current) return

      switch (e.code) {
        case 'Space':
          e.preventDefault()
          togglePlayPause()
          break
        case 'ArrowLeft':
          e.preventDefault()
          skipBackward()
          break
        case 'ArrowRight':
          e.preventDefault()
          skipForward()
          break
        case 'KeyM':
          e.preventDefault()
          toggleMute()
          break
        case 'KeyF':
          e.preventDefault()
          toggleFullscreen()
          break
        case 'Escape':
          if (isFullscreen) {
            exitFullscreen()
          } else {
            onClose()
          }
          break
      }
    }

    document.addEventListener('keydown', handleKeyPress)
    return () => document.removeEventListener('keydown', handleKeyPress)
  }, [isFullscreen])

  const togglePlayPause = () => {
    if (!videoRef.current) return
    
    if (isPlaying) {
      videoRef.current.pause()
    } else {
      videoRef.current.play()
    }
  }

  const toggleMute = () => {
    if (!videoRef.current) return
    videoRef.current.muted = !videoRef.current.muted
  }

  const skipBackward = () => {
    if (!videoRef.current) return
    const newTime = Math.max(result.start_time, videoRef.current.currentTime - 10)
    videoRef.current.currentTime = newTime
  }

  const skipForward = () => {
    if (!videoRef.current) return
    const newTime = Math.min(result.end_time, videoRef.current.currentTime + 10)
    videoRef.current.currentTime = newTime
  }

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current) return
    
    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percentage = clickX / rect.width
    const segmentDuration = result.end_time - result.start_time
    const newTime = result.start_time + (percentage * segmentDuration)
    
    videoRef.current.currentTime = newTime
  }

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      videoRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  const exitFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  const getProgressPercentage = () => {
    const segmentDuration = result.end_time - result.start_time
    const playedDuration = currentTime - result.start_time
    return Math.max(0, Math.min(100, (playedDuration / segmentDuration) * 100))
  }

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  const videoUrl = result.video_url || `/static/videos/${result.video_id}.mp4`

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
        onClick={handleBackdropClick}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-4xl bg-black rounded-lg overflow-hidden video-segment-highlight"
          onMouseMove={() => setShowControls(true)}
        >
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 z-10 w-10 h-10 bg-black/50 hover:bg-black/70 rounded-full flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>

          {/* Video Element */}
          <video
            ref={videoRef}
            className="w-full aspect-video"
            src={videoUrl}
            preload="metadata"
            onClick={() => setShowControls(!showControls)}
          />

          {/* Loading Overlay */}
          {!isLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-white text-center">
                <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-2" />
                <p>Loading video...</p>
              </div>
            </div>
          )}

          {/* Controls Overlay */}
          <AnimatePresence>
            {showControls && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30"
              >
                {/* Top Info Bar */}
                <div className="absolute top-0 left-0 right-0 p-4">
                  <div className="flex items-center justify-between text-white">
                    <div>
                      <h3 className="font-medium">{result.title || 'Video Segment'}</h3>
                      <p className="text-sm text-white/80">
                        {formatDuration(result.start_time)} - {formatDuration(result.end_time)}
                      </p>
                    </div>
                    <div className="text-sm bg-black/50 px-2 py-1 rounded">
                      Score: {(result.score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {/* Center Play Button */}
                {!isPlaying && isLoaded && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <button
                      onClick={togglePlayPause}
                      className="w-20 h-20 bg-white/20 hover:bg-white/30 rounded-full flex items-center justify-center transition-colors"
                    >
                      <Play className="w-10 h-10 text-white ml-1" fill="currentColor" />
                    </button>
                  </div>
                )}

                {/* Bottom Controls */}
                <div className="absolute bottom-0 left-0 right-0 p-4">
                  {/* Progress Bar */}
                  <div
                    className="w-full h-2 bg-white/20 rounded-full cursor-pointer mb-4"
                    onClick={handleSeek}
                  >
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-150"
                      style={{ width: `${getProgressPercentage()}%` }}
                    />
                  </div>

                  {/* Control Buttons */}
                  <div className="flex items-center justify-between text-white">
                    <div className="flex items-center gap-4">
                      <button
                        onClick={skipBackward}
                        className="p-2 hover:bg-white/20 rounded transition-colors"
                      >
                        <SkipBack className="w-5 h-5" />
                      </button>
                      
                      <button
                        onClick={togglePlayPause}
                        className="p-2 hover:bg-white/20 rounded transition-colors"
                      >
                        {isPlaying ? (
                          <Pause className="w-6 h-6" />
                        ) : (
                          <Play className="w-6 h-6" fill="currentColor" />
                        )}
                      </button>
                      
                      <button
                        onClick={skipForward}
                        className="p-2 hover:bg-white/20 rounded transition-colors"
                      >
                        <SkipForward className="w-5 h-5" />
                      </button>
                      
                      <button
                        onClick={toggleMute}
                        className="p-2 hover:bg-white/20 rounded transition-colors"
                      >
                        {isMuted ? (
                          <VolumeX className="w-5 h-5" />
                        ) : (
                          <Volume2 className="w-5 h-5" />
                        )}
                      </button>
                    </div>

                    <div className="flex items-center gap-4">
                      <span className="text-sm">
                        {formatDuration(currentTime)} / {formatDuration(result.end_time)}
                      </span>
                      
                      <button
                        onClick={toggleFullscreen}
                        className="p-2 hover:bg-white/20 rounded transition-colors"
                      >
                        <Maximize className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
