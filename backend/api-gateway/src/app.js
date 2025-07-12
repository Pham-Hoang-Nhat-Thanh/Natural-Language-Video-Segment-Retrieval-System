const fastify = require('fastify')({
  logger: {
    level: process.env.LOG_LEVEL || 'info',
    transport: process.env.NODE_ENV === 'development' ? {
      target: 'pino-pretty'
    } : undefined
  }
});

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const Redis = require('ioredis');

// Configuration
const config = {
  port: process.env.PORT || 8000,
  host: process.env.HOST || '0.0.0.0',
  ingestServiceUrl: process.env.INGEST_SERVICE_URL || 'http://localhost:8001',
  searchServiceUrl: process.env.SEARCH_SERVICE_URL || 'http://localhost:8002',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
  jwtSecret: process.env.JWT_SECRET || 'your-secret-key',
  enableAuth: process.env.ENABLE_AUTH === 'true'
};

// Redis client
const redis = new Redis(config.redisUrl);

// Rate limiting
const rateLimitCache = new Map();

function rateLimit(limit, windowMs) {
  return async (request, reply) => {
    const key = request.ip;
    const now = Date.now();
    const windowStart = now - windowMs;
    
    // Clean old entries
    if (rateLimitCache.has(key)) {
      const timestamps = rateLimitCache.get(key).filter(time => time > windowStart);
      rateLimitCache.set(key, timestamps);
    }
    
    const requests = rateLimitCache.get(key) || [];
    
    if (requests.length >= limit) {
      return reply.code(429).send({ 
        error: 'Too many requests',
        retryAfter: Math.ceil((requests[0] + windowMs - now) / 1000)
      });
    }
    
    requests.push(now);
    rateLimitCache.set(key, requests);
  };
}

// JWT Authentication (simple implementation)
async function authenticate(request, reply) {
  if (!config.enableAuth) return; // Skip auth if disabled
  
  const token = request.headers.authorization?.replace('Bearer ', '');
  if (!token) {
    return reply.code(401).send({ error: 'Authentication required' });
  }
  
  try {
    // In production, use proper JWT validation
    // For now, just check if token exists
    if (token !== 'valid-token') {
      return reply.code(401).send({ error: 'Invalid token' });
    }
  } catch (error) {
    return reply.code(401).send({ error: 'Invalid token' });
  }
}

// Register plugins
async function registerPlugins() {
  // Rate limiting plugin
  await fastify.register(require('@fastify/rate-limit'), {
    max: 100,
    timeWindow: '1 minute',
    redis: redis
  });

  // CORS
  await fastify.register(require('@fastify/cors'), {
    origin: true,
    credentials: true
  });

  // Static file serving - videos and thumbnails (no copying)
  await fastify.register(require('@fastify/static'), {
    root: path.join(__dirname, '../../../data'),
    prefix: '/static/',
    serve: true,
    // Prevent listing directories for security
    list: false
  });

  // Swagger documentation
  await fastify.register(require('@fastify/swagger'), {
    swagger: {
      info: {
        title: 'Video Segment Retrieval API',
        description: 'API for natural language video segment retrieval',
        version: '1.0.0'
      },
      host: 'localhost:8000',
      schemes: ['http'],
      consumes: ['application/json'],
      produces: ['application/json']
    }
  });

  await fastify.register(require('@fastify/swagger-ui'), {
    routePrefix: '/docs',
    uiConfig: {
      docExpansion: 'full',
      deepLinking: false
    }
  });

  // Security headers
  await fastify.register(require('@fastify/helmet'));
}

// Health check endpoint
fastify.get('/health', async (request, reply) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {}
  };

  try {
    // Check ingest service
    const ingestResponse = await axios.get(`${config.ingestServiceUrl}/health`, { timeout: 5000 });
    health.services.ingest = ingestResponse.data.status === 'healthy' ? 'up' : 'down';
  } catch (error) {
    health.services.ingest = 'down';
  }

  try {
    // Check search service
    const searchResponse = await axios.get(`${config.searchServiceUrl}/health`, { timeout: 5000 });
    health.services.search = searchResponse.data.status === 'healthy' ? 'up' : 'down';
  } catch (error) {
    health.services.search = 'down';
  }

  try {
    // Check Redis
    await redis.ping();
    health.services.redis = 'up';
  } catch (error) {
    health.services.redis = 'down';
  }

  const allServicesUp = Object.values(health.services).every(status => status === 'up');
  reply.code(allServicesUp ? 200 : 503).send(health);
});

// Video management endpoints (proxy to ingestion service)
fastify.get('/api/videos', {
  preHandler: [rateLimit(50, 60000)], // 50 requests per minute
  schema: {
    description: 'List all videos in the dataset with processing status',
    response: {
      200: {
        type: 'object',
        properties: {
          videos: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                video_id: { type: 'string' },
                filename: { type: 'string' },
                processed: { type: 'boolean' },
                file_size: { type: 'number' },
                path: { type: 'string' }
              }
            }
          }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.get(`${config.ingestServiceUrl}/api/videos`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error listing videos:', error);
    return reply.code(500).send({ error: 'Failed to list videos' });
  }
});

fastify.post('/api/videos/:video_id/process', {
  preHandler: [authenticate, rateLimit(10, 60000)], // Auth + 10 requests per minute
  schema: {
    description: 'Process a specific video for search indexing',
    params: {
      type: 'object',
      properties: {
        video_id: { type: 'string', description: 'Video identifier' }
      },
      required: ['video_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          video_id: { type: 'string' },
          status: { type: 'string' },
          shots_detected: { type: 'number' },
          keyframes_extracted: { type: 'number' },
          embeddings_generated: { type: 'number' },
          processing_time_seconds: { type: 'number' },
          message: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const { video_id } = request.params;
    const response = await axios.post(`${config.ingestServiceUrl}/api/videos/${video_id}/process`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error processing video:', error);
    if (error.response?.status === 404) {
      return reply.code(404).send({ error: `Video ${request.params.video_id} not found` });
    }
    return reply.code(500).send({ error: 'Failed to process video' });
  }
});

fastify.post('/api/videos/process-all', {
  preHandler: [authenticate, rateLimit(2, 300000)], // Auth + 2 requests per 5 minutes
  schema: {
    description: 'Process all unprocessed videos in the dataset',
    response: {
      200: {
        type: 'object',
        properties: {
          status: { type: 'string' },
          processed_count: { type: 'number' },
          failed_videos: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                video_id: { type: 'string' },
                error: { type: 'string' }
              }
            }
          },
          message: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.post(`${config.ingestServiceUrl}/api/videos/process-all`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error processing all videos:', error);
    return reply.code(500).send({ error: 'Failed to process videos' });
  }
});

fastify.get('/api/videos/:video_id/status', {
  preHandler: [rateLimit(100, 60000)], // 100 requests per minute
  schema: {
    description: 'Get processing status for a specific video',
    params: {
      type: 'object',
      properties: {
        video_id: { type: 'string', description: 'Video identifier' }
      },
      required: ['video_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          video_id: { type: 'string' },
          processed: { type: 'boolean' },
          shots_count: { type: 'number' },
          keyframes_count: { type: 'number' },
          embeddings_count: { type: 'number' },
          processed_at: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const { video_id } = request.params;
    const response = await axios.get(`${config.ingestServiceUrl}/api/videos/${video_id}/status`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error getting video status:', error);
    return reply.code(500).send({ error: 'Failed to get video status' });
  }
});

fastify.delete('/api/videos/:video_id', {
  preHandler: [authenticate, rateLimit(10, 60000)], // Auth + 10 requests per minute
  schema: {
    description: 'Delete processed video data (preserves original MP4 file)',
    params: {
      type: 'object',
      properties: {
        video_id: { type: 'string', description: 'Video identifier' }
      },
      required: ['video_id']
    },
    response: {
      200: {
        type: 'object',
        properties: {
          video_id: { type: 'string' },
          status: { type: 'string' },
          note: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const { video_id } = request.params;
    const response = await axios.delete(`${config.ingestServiceUrl}/api/videos/${video_id}`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error deleting video data:', error);
    return reply.code(500).send({ error: 'Failed to delete video data' });
  }
});

// Search endpoints
fastify.post('/api/search', {
  schema: {
    description: 'Search for video segments using natural language',
    body: {
      type: 'object',
      required: ['query'],
      properties: {
        query: { type: 'string', description: 'Natural language search query' },
        top_k: { type: 'integer', default: 10, description: 'Number of results to return' },
        threshold: { type: 'number', default: 0.5, description: 'Minimum similarity threshold' }
      }
    },
    response: {
      200: {
        type: 'object',
        properties: {
          results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                video_id: { type: 'string' },
                start_time: { type: 'number' },
                end_time: { type: 'number' },
                score: { type: 'number' },
                thumbnail_url: { type: 'string' },
                title: { type: 'string' }
              }
            }
          },
          query_time_ms: { type: 'number' },
          total_results: { type: 'integer' }
        }
      }
    }
  }
}, async (request, reply) => {
  const startTime = Date.now();
  const { query, top_k = 10, threshold = 0.5 } = request.body;

  try {
    // Check cache first
    const cacheKey = `search:${Buffer.from(query).toString('base64')}:${top_k}:${threshold}`;
    const cachedResult = await redis.get(cacheKey);
    
    if (cachedResult) {
      const result = JSON.parse(cachedResult);
      result.query_time_ms = Date.now() - startTime;
      result.cached = true;
      return result;
    }

    // Forward to search service
    const response = await axios.post(`${config.searchServiceUrl}/api/search`, {
      query,
      top_k,
      threshold
    });

    const result = response.data;
    result.query_time_ms = Date.now() - startTime;

    // Cache result for 5 minutes
    await redis.setex(cacheKey, 300, JSON.stringify(result));

    return result;
  } catch (error) {
    fastify.log.error('Search error:', error);
    return reply.code(500).send({ 
      error: 'Search failed',
      query_time_ms: Date.now() - startTime
    });
  }
});

// Text embedding endpoint
fastify.post('/api/embed/text', {
  schema: {
    description: 'Generate text embeddings using CLIP model',
    body: {
      type: 'object',
      required: ['text'],
      properties: {
        text: { type: 'string', description: 'Text to embed' }
      }
    },
    response: {
      200: {
        type: 'object',
        properties: {
          embedding: {
            type: 'array',
            items: { type: 'number' },
            description: 'Text embedding vector'
          },
          dimension: { type: 'number' },
          model: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.post(`${config.searchServiceUrl}/api/embed/text`, request.body);
    return response.data;
  } catch (error) {
    fastify.log.error('Text embedding error:', error);
    return reply.code(500).send({ error: 'Failed to generate embeddings' });
  }
});

// Video embedding endpoint
fastify.post('/api/embed/video', {
  schema: {
    description: 'Generate video embeddings from keyframes',
    body: {
      type: 'object',
      required: ['video_id'],
      properties: {
        video_id: { type: 'string' },
        keyframes: {
          type: 'array',
          items: { type: 'string' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.post(`${config.searchServiceUrl}/api/embed/video`, request.body);
    return response.data;
  } catch (error) {
    fastify.log.error('Video embedding error:', error);
    return reply.code(500).send({ error: 'Failed to generate video embeddings' });
  }
});

// Reranking endpoint
fastify.post('/api/rerank', {
  schema: {
    description: 'Rerank search results using cross-encoder',
    body: {
      type: 'object',
      required: ['query', 'candidates'],
      properties: {
        query: { type: 'string' },
        candidates: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              video_id: { type: 'string' },
              start_time: { type: 'number' },
              end_time: { type: 'number' },
              score: { type: 'number' }
            }
          }
        }
      }
    },
    response: {
      200: {
        type: 'object',
        properties: {
          reranked_results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                video_id: { type: 'string' },
                start_time: { type: 'number' },
                end_time: { type: 'number' },
                score: { type: 'number' }
              }
            }
          }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.post(`${config.searchServiceUrl}/api/rerank`, request.body);
    return response.data;
  } catch (error) {
    fastify.log.error('Reranking error:', error);
    return reply.code(500).send({ error: 'Failed to rerank results' });
  }
});

// Boundary regression endpoint
fastify.post('/api/regress', {
  schema: {
    description: 'Refine segment boundaries using regression model',
    body: {
      type: 'object',
      required: ['video_segments', 'query_embedding'],
      properties: {
        video_segments: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              video_id: { type: 'string' },
              start_time: { type: 'number' },
              end_time: { type: 'number' },
              score: { type: 'number' }
            }
          }
        },
        query_embedding: {
          type: 'array',
          items: { type: 'number' }
        }
      }
    }
  }
}, async (request, reply) => {
  try {
    const response = await axios.post(`${config.searchServiceUrl}/api/regress`, request.body);
    return response.data;
  } catch (error) {
    fastify.log.error('Boundary regression error:', error);
    return reply.code(500).send({ error: 'Failed to regress boundaries' });
  }
});

// Admin endpoints
fastify.get('/api/admin/stats', async (request, reply) => {
  try {
    const [ingestStats, searchStats] = await Promise.all([
      axios.get(`${config.ingestServiceUrl}/api/stats`),
      axios.get(`${config.searchServiceUrl}/api/stats`)
    ]);

    return {
      ingest: ingestStats.data,
      search: searchStats.data,
      cache: {
        connected: redis.status === 'ready',
        memory_usage: await redis.memory('usage')
      }
    };
  } catch (error) {
    fastify.log.error('Stats error:', error);
    return reply.code(500).send({ error: 'Failed to get stats' });
  }
});

// Error handler
fastify.setErrorHandler((error, request, reply) => {
  fastify.log.error(error);
  reply.status(500).send({ error: 'Internal Server Error' });
});

// Video streaming endpoint
fastify.get('/api/videos/:videoId/stream', async (request, reply) => {
  try {
    const { videoId } = request.params;
    
    // Forward to ingestion service for video streaming
    const videoPath = path.join('/workspace/competitions/AIC_2025/SIU_Unicorn/Natural-Language-Video-Segment-Retrieval-System/data/videos/datasets/custom', `${videoId}.mp4`);
    
    // Check if file exists
    if (!fs.existsSync(videoPath)) {
      return reply.code(404).send({ error: 'Video not found' });
    }
    
    // Get file stats for range requests
    const stats = fs.statSync(videoPath);
    const fileSize = stats.size;
    
    // Handle range requests for video streaming
    const range = request.headers.range;
    
    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunksize = (end - start) + 1;
      
      const stream = fs.createReadStream(videoPath, { start, end });
      
      reply.code(206)
        .header('Content-Range', `bytes ${start}-${end}/${fileSize}`)
        .header('Accept-Ranges', 'bytes')
        .header('Content-Length', chunksize)
        .header('Content-Type', 'video/mp4')
        .send(stream);
    } else {
      reply
        .header('Content-Length', fileSize)
        .header('Content-Type', 'video/mp4')
        .send(fs.createReadStream(videoPath));
    }
  } catch (error) {
    fastify.log.error('Video streaming error:', error);
    reply.code(500).send({ error: 'Video streaming failed' });
  }
});

// Start server
async function start() {
  try {
    await registerPlugins();
    
    await fastify.listen({
      port: config.port,
      host: config.host
    });
    
    fastify.log.info(`API Gateway started on http://${config.host}:${config.port}`);
    fastify.log.info(`Swagger docs available at http://${config.host}:${config.port}/docs`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  try {
    await redis.quit();
    await fastify.close();
    fastify.log.info('Server shut down gracefully');
  } catch (err) {
    fastify.log.error('Error during shutdown:', err);
    process.exit(1);
  }
});

if (require.main === module) {
  start();
}

module.exports = fastify;
