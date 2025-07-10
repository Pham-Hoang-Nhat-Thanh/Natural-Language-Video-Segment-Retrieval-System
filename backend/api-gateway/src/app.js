const fastify = require('fastify')({
  logger: {
    level: process.env.LOG_LEVEL || 'info',
    prettyPrint: process.env.NODE_ENV === 'development'
  }
});

const axios = require('axios');
const Redis = require('ioredis');
const path = require('path');

// Configuration
const config = {
  port: process.env.PORT || 8000,
  host: process.env.HOST || '0.0.0.0',
  ingestServiceUrl: process.env.INGEST_SERVICE_URL || 'http://localhost:8001',
  searchServiceUrl: process.env.SEARCH_SERVICE_URL || 'http://localhost:8002',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379'
};

// Redis client
const redis = new Redis(config.redisUrl);

// Register plugins
async function registerPlugins() {
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
fastify.get('/api/videos', async (request, reply) => {
  try {
    const response = await axios.get(`${config.ingestServiceUrl}/api/videos`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error listing videos:', error);
    return reply.code(500).send({ error: 'Failed to list videos' });
  }
});

fastify.post('/api/videos/:video_id/process', async (request, reply) => {
  try {
    const { video_id } = request.params;
    const response = await axios.post(`${config.ingestServiceUrl}/api/process/${video_id}`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error processing video:', error);
    if (error.response?.status === 404) {
      return reply.code(404).send({ error: `Video ${request.params.video_id} not found` });
    }
    return reply.code(500).send({ error: 'Failed to process video' });
  }
});

fastify.post('/api/videos/process-all', async (request, reply) => {
  try {
    const response = await axios.post(`${config.ingestServiceUrl}/api/process-all`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error processing all videos:', error);
    return reply.code(500).send({ error: 'Failed to process videos' });
  }
});

fastify.get('/api/videos/:video_id/status', async (request, reply) => {
  try {
    const { video_id } = request.params;
    const response = await axios.get(`${config.ingestServiceUrl}/api/videos/${video_id}/status`);
    return response.data;
  } catch (error) {
    fastify.log.error('Error getting video status:', error);
    return reply.code(500).send({ error: 'Failed to get video status' });
  }
});

fastify.delete('/api/videos/:video_id', async (request, reply) => {
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
fastify.post('/api/query/embed', {
  schema: {
    description: 'Generate text embeddings',
    body: {
      type: 'object',
      required: ['text'],
      properties: {
        text: { type: 'string' }
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

// Reranking endpoint
fastify.post('/api/query/rerank', {
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
fastify.post('/api/query/regress', {
  schema: {
    description: 'Refine segment boundaries using regression model'
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
