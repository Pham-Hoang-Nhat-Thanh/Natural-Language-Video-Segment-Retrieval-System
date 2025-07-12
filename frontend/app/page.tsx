// @ts-nocheck
'use client'

import { useState, useEffect } from 'react'

export default function VideosPage() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [processingIds, setProcessingIds] = useState(new Set()) // track per-video processing
  const [deletingIds, setDeletingIds] = useState(new Set())   // track per-video deletion

  const fetchVideos = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/videos')
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setVideos(data.videos || data)
    } catch (e) {
      setError(`Failed to fetch videos: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchVideos()
  }, [])

  const handleProcess = async (video_id) => {
    // mark as processing
    setProcessingIds(prev => new Set(prev).add(video_id))
    try {
      await fetch(`/api/videos/${video_id}/process`, { method: 'POST' })
      await fetchVideos()
    } catch (e) {
      setError(`Failed to process video: ${e.message}`)
    } finally {
      setProcessingIds(prev => { const s = new Set(prev); s.delete(video_id); return s })
    }
  }

  const handleProcessAll = async () => {
    try {
      await fetch('/api/videos/process-all', { method: 'POST' })
      fetchVideos()
    } catch (e) {
      setError(`Failed to process all videos: ${e.message}`)
    }
  }

  const handleDelete = async (video_id) => {
    if (!confirm('Are you sure you want to delete this video?')) return
    try {
      // mark as deleting
      setDeletingIds(prev => new Set(prev).add(video_id))
      await fetch(`/api/videos/${video_id}`, { method: 'DELETE' })
      setVideos(videos.filter(video => video.video_id !== video_id))
    } catch (e) {
      setError(`Failed to delete video: ${e.message}`)
    } finally {
      setDeletingIds(prev => { const s = new Set(prev); s.delete(video_id); return s })
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Videos</h1>
      {error && <div className="text-red-600 mb-4">{error}</div>}

      <div className="mb-4">
        <button
          onClick={handleProcessAll}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Process All Videos'}
        </button>
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : (
        <table className="min-w-full bg-white border">
          <thead>
            <tr>
              <th className="px-4 py-2 border">Video ID</th>
              <th className="px-4 py-2 border">Filename</th>
              <th className="px-4 py-2 border">Processed</th>
              <th className="px-4 py-2 border">Actions</th>
            </tr>
          </thead>
          <tbody>
            {videos.map((video) => (
              <tr key={video.video_id}>
                <td className="border px-4 py-2">{video.video_id}</td>
                <td className="border px-4 py-2">{video.filename}</td>
                <td className="border px-4 py-2">
                  {video.processed ? '✅' : '❌'}
                </td>
                <td className="border px-4 py-2 space-x-2">
                  {!video.processed && (
                    <button
                      onClick={() => handleProcess(video.video_id)}
                      className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Process
                    </button>
                  )}
                  {video.processed && (
                    <button
                      onClick={() => handleDelete(video.video_id)}
                      className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                    >
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
