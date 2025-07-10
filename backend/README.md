# Backend Services

This directory contains all backend microservices for the Natural Language-Driven Video Segment Retrieval system.

## 🏗️ Architecture

```
backend/
├── api-gateway/     # Node.js + Fastify - API Gateway & Auth
├── ingest/         # Python + FastAPI - Video Processing & Ingestion  
└── search/         # Python + FastAPI - ML-Powered Search
```

## 🚀 Services Overview

### API Gateway (`api-gateway/`)
- **Technology**: Node.js + Fastify
- **Port**: 8000
- **Purpose**: 
  - Single entry point for all API requests
  - Authentication and authorization
  - Rate limiting and caching
  - Request routing to microservices
  - Swagger documentation

### Ingestion Service (`ingest/`)
- **Technology**: Python + FastAPI
- **Port**: 8001
- **Purpose**:
  - Video upload and validation
  - Shot boundary detection (TransNetV2)
  - Keyframe extraction
  - Multimodal embedding generation (CLIP)
  - Metadata storage

### Search Service (`search/`)
- **Technology**: Python + FastAPI
- **Port**: 8002
- **Purpose**:
  - Text encoding for queries
  - FAISS-based approximate nearest neighbor search
  - Cross-encoder reranking
  - Neural boundary regression
  - Result ranking and filtering

## 🛠️ Development

### Quick Start
```bash
# Start all services with Docker Compose (recommended)
docker-compose up -d

# Or start individual services for development
cd api-gateway && npm run dev
cd ingest && python main.py
cd search && python main.py
```

### Environment Setup
Each service has its own `.env.example` file. Copy and customize:
```bash
cp api-gateway/.env.example api-gateway/.env
cp ingest/.env.example ingest/.env
cp search/.env.example search/.env
```

### Health Checks
All services provide health endpoints:
- API Gateway: http://localhost:8000/health
- Ingestion: http://localhost:8001/health
- Search: http://localhost:8002/health

## 📋 API Documentation

### API Gateway
- **Swagger UI**: http://localhost:8000/docs
- **Main Routes**:
  - `GET /health` - Service health
  - `POST /api/search` - Search videos
  - `POST /api/ingest/video` - Upload video
  - `GET /api/videos` - List videos

### Service Communication
```
Frontend → API Gateway → [Ingest Service | Search Service]
                      ↓
                   Database + Redis + Storage
```

## 🔧 Configuration

### Common Environment Variables
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection  
- `MODEL_PATH` - ML model directory
- `DATA_PATH` - Data storage directory

### Service-Specific
- **API Gateway**: JWT secrets, CORS origins, rate limits
- **Ingest**: Video processing parameters, model configs
- **Search**: FAISS index settings, search parameters

## 🧪 Testing

```bash
# Run all backend tests
python -m pytest tests/ -v

# Test specific services
npm test                    # API Gateway
python -m pytest tests/test_ingest* # Ingestion
python -m pytest tests/test_search* # Search
```

## 📊 Monitoring

All services include:
- Prometheus metrics at `/metrics`
- Structured logging with correlation IDs
- Distributed tracing with Jaeger
- Health check endpoints

## 🚀 Deployment

### Docker
Each service has both development and production Dockerfiles:
- `Dockerfile.dev` - Development with hot reload
- `Dockerfile` - Production optimized

### Kubernetes
See `../infra/k8s/` for deployment manifests.

## 📚 Additional Documentation

- See individual service README files for detailed information
- API documentation available at each service's `/docs` endpoint
- Architecture diagrams in `../docs/architecture/`
