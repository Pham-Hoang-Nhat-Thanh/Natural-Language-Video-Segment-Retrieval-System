import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios'
import { SearchRequest, SearchResponse, AdminStats } from '@/types'

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for auth tokens (if needed)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.response?.status === 401) {
      // Handle unauthorized
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token')
        window.location.href = '/login'
      }
    }
    
    // Enhance error message
    const message = error.response?.data?.error || error.message || 'An error occurred'
    error.message = message
    
    return Promise.reject(error)
  }
)

// Search API
export const searchVideos = async (request: SearchRequest): Promise<SearchResponse> => {
  const response = await api.post('/api/search', request)
  return response.data
}

export const embedText = async (text: string): Promise<{ embedding: number[] }> => {
  const response = await api.post('/api/query/embed', { text })
  return response.data
}

export const rerankResults = async (query: string, candidates: any[]): Promise<any[]> => {
  const response = await api.post('/api/query/rerank', { query, candidates })
  return response.data
}

export const regressBoundaries = async (data: any): Promise<any> => {
  const response = await api.post('/api/query/regress', data)
  return response.data
}

// Video management API (no uploads - videos are manually added to dataset directory)
export const listVideos = async (): Promise<{ videos: any[] }> => {
  const response = await api.get('/api/videos')
  return response.data
}

export const processVideo = async (videoId: string): Promise<any> => {
  const response = await api.post(`/api/videos/${videoId}/process`)
  return response.data
}

export const processAllVideos = async (): Promise<any> => {
  const response = await api.post('/api/videos/process-all')
  return response.data
}

export const getVideoStatus = async (videoId: string): Promise<any> => {
  const response = await api.get(`/api/videos/${videoId}/status`)
  return response.data
}

export const deleteVideoData = async (videoId: string): Promise<void> => {
  await api.delete(`/api/videos/${videoId}`)
}

// Admin API
export const getAdminStats = async (): Promise<AdminStats> => {
  const response = await api.get('/api/admin/stats')
  return response.data
}

export const getSystemHealth = async (): Promise<any> => {
  const response = await api.get('/health')
  return response.data
}

export const getPerformanceMetrics = async (timeRange: string = '24h'): Promise<any> => {
  const response = await api.get(`/api/admin/metrics?range=${timeRange}`)
  return response.data
}

// User management API (if needed)
export const login = async (email: string, password: string): Promise<{ token: string, user: any }> => {
  const response = await api.post('/api/auth/login', { email, password })
  return response.data
}

export const logout = async (): Promise<void> => {
  await api.post('/api/auth/logout')
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token')
  }
}

export const getCurrentUser = async (): Promise<any> => {
  const response = await api.get('/api/auth/me')
  return response.data
}

// Video metadata API
export const getVideoMetadata = async (videoId: string): Promise<any> => {
  const response = await api.get(`/api/videos/${videoId}/metadata`)
  return response.data
}

export const getVideoShots = async (videoId: string): Promise<any[]> => {
  const response = await api.get(`/api/videos/${videoId}/shots`)
  return response.data
}

export const getVideoKeyframes = async (videoId: string): Promise<any[]> => {
  const response = await api.get(`/api/videos/${videoId}/keyframes`)
  return response.data
}

// Export api instance for direct use
export default api
