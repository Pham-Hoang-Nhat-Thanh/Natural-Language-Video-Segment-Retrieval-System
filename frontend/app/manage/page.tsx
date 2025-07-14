/**
 * Video Management Page
 * 
 * Administrative interface for:
 * - Uploading new videos
 * - Processing videos for indexing
 * - Managing video library
 * - Monitoring system health
 */

'use client'

import { useState, useEffect } from 'react'
import { getVideos, processVideo, deleteVideo, uploadVideo, checkApiHealth } from '@/lib/api'

export default function ManagementPage() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [processingIds, setProcessingIds] = useState(new Set())
  const [deletingIds, setDeletingIds] = useState(new Set())
  const [uploadProgress, setUploadProgress] = useState(null)
  const [healthStatus, setHealthStatus] = useState({ ingest: false, search: false })

  const fetchVideos = async () => {
    setLoading(true)
    setError(null)
    try {
      const videoData = await getVideos()
      setVideos(videoData)
    } catch (e) {
      setError(`Failed to fetch videos: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const checkHealth = async () => {
    try {
      const health = await checkApiHealth()
      setHealthStatus(health)
    } catch (e) {
      console.error('Health check failed:', e)
    }
  }

  useEffect(() => {
    fetchVideos()
    checkHealth()
    
    // Check health every 30 seconds
    const healthInterval = setInterval(checkHealth, 30000)
    return () => clearInterval(healthInterval)
  }, [])

  const handleProcess = async (video_id) => {
    setProcessingIds(prev => new Set(prev).add(video_id))
    try {
      await processVideo(video_id)
      await fetchVideos()
    } catch (e) {
      setError(`Failed to process video: ${e.message}`)
    } finally {
      setProcessingIds(prev => { const s = new Set(prev); s.delete(video_id); return s })
    }
  }

  const handleDelete = async (video_id) => {
    if (!confirm('Are you sure you want to delete this video?')) return
    
    setDeletingIds(prev => new Set(prev).add(video_id))
    try {
      await deleteVideo(video_id)
      setVideos(videos.filter(video => video.video_id !== video_id))
    } catch (e) {
      setError(`Failed to delete video: ${e.message}`)
    } finally {
      setDeletingIds(prev => { const s = new Set(prev); s.delete(video_id); return s })
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setUploadProgress({ name: file.name, progress: 0 })
    try {
      await uploadVideo(file)
      await fetchVideos()
      setUploadProgress(null)
    } catch (e) {
      setError(`Failed to upload video: ${e.message}`)
      setUploadProgress(null)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Header Section with Health Status */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Video Management</h1>
            <p className="text-gray-600 mt-1">Upload, process, and manage your video library</p>
          </div>
          
          {/* Health Status */}
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-700">System Status:</span>
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1">
                  <div className={`w-3 h-3 rounded-full ${healthStatus.ingest ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm text-gray-600">Ingest</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className={`w-3 h-3 rounded-full ${healthStatus.search ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm text-gray-600">Search</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Upload Section */}
        <section className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload New Video</h2>
          
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              className="hidden"
              id="video-upload"
            />
            <label htmlFor="video-upload" className="cursor-pointer">
              <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <div className="mt-4">
                <p className="text-gray-600">Click to upload a video file</p>
                <p className="text-sm text-gray-500 mt-1">MP4, AVI, MOV up to 2GB</p>
              </div>
            </label>
          </div>

          {uploadProgress && (
            <div className="mt-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Uploading: {uploadProgress.name}</span>
                <span className="text-gray-600">{uploadProgress.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress.progress}%` }}
                />
              </div>
            </div>
          )}
        </section>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex">
              <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
              <button 
                onClick={() => setError(null)}
                className="ml-auto text-red-400 hover:text-red-600"
              >
                <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Videos Table */}
        <section className="bg-white rounded-lg shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900">Video Library</h2>
              <button
                onClick={fetchVideos}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>

          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading videos...</p>
            </div>
          ) : videos.length === 0 ? (
            <div className="p-8 text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <h3 className="mt-4 text-lg font-medium text-gray-900">No videos found</h3>
              <p className="mt-2 text-gray-600">Upload your first video to get started.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Video
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {videos.map((video) => (
                    <tr key={video.video_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10">
                            <div className="h-10 w-10 rounded-lg bg-gray-200 flex items-center justify-center">
                              <svg className="h-6 w-6 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                              </svg>
                            </div>
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {video.filename || video.title || `Video ${video.video_id}`}
                            </div>
                            <div className="text-sm text-gray-500">
                              ID: {video.video_id}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          video.processed 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {video.processed ? 'Processed' : 'Pending'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                        {!video.processed && (
                          <button
                            onClick={() => handleProcess(video.video_id)}
                            disabled={processingIds.has(video.video_id)}
                            className="inline-flex items-center px-3 py-1 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {processingIds.has(video.video_id) ? (
                              <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                                Processing...
                              </>
                            ) : (
                              'Process'
                            )}
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(video.video_id)}
                          disabled={deletingIds.has(video.video_id)}
                          className="inline-flex items-center px-3 py-1 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {deletingIds.has(video.video_id) ? (
                            <>
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                              Deleting...
                            </>
                          ) : (
                            'Delete'
                          )}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
