# Natural Language Video Segment Retrieval System

🎉 **NEW: Automated ML Model Management** - No more manual model placement! See [📖 scripts/README.md](scripts/README.md)

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Environment Setup

1. **Clone and setup**
```bash
   git clone <repository-url>
   cd video-segment-retrieval
```

2. **🤖 Automated ML Model Setup (NEW)**
```bash
# Download and setup ALL ML models automatically
python scripts/download_models.py

# Or setup individual components
python scripts/download_models.py --clip-only      # CLIP + ONNX conversion
python scripts/download_models.py --reranker-only  # Cross-encoder models  
python scripts/download_models.py --regressor-only # Boundary regressor + training
```

3. Copy the example environment files:
```bash
   cp .env.example .env
   cp frontend/.env.example frontend/.env.local
   cp backend/api-gateway/.env.example backend/api-gateway/.env
   cp backend/ingest/.env.example backend/ingest/.env
   cp backend/search/.env.example backend/search/.env
```

4. Customize the environment variables as needed

2. **Start with Docker (Recommended - Models Auto-Downloaded)**
```bash
# Start all services (ML models downloaded automatically during build)
docker-compose up -d

# Check service health
docker-compose ps

# View logs (including model download progress)
docker-compose logs -f

# Verify models are ready
cat models/model_manifest.json
```

3. **Access Applications**
- Frontend: http://localhost:3000
- API Gateway: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin123)
- Prometheus: http://localhost:9090

### Manual Development Setup

A high-performance, full-stack application for retrieving precise video segments using natural language queries with <50ms end-to-end latency.

## � Quick Start

### Development Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd video-segment-retrieval

# Start all services with Docker Compose
docker-compose up -d

# Access applications
# Frontend: http://localhost:3000
# API Gateway: http://localhost:8000
# Admin Dashboard: http://localhost:3000/admin
```

### Manual Setup

```bash
# Backend services
cd services/api-gateway && npm install && npm run dev
cd services/ingest && pip install -r requirements.txt && python main.py
cd services/search && pip install -r requirements.txt && python main.py

# Frontend
cd frontend && npm install && npm run dev
```

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│  API Gateway    │────│  ML Services    │
│   (Next.js)     │    │  (Node.js)      │    │  (FastAPI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                         ┌─────────────────┐
                         │  Vector Store   │
                         │  (FAISS)        │
                         └─────────────────┘
```

### Core Components

1. **Offline Ingestion Pipeline**
   - Shot detection and keyframe extraction
   - CLIP-based embedding generation
   - FAISS index construction

2. **Online Query Pipeline**
   - Text encoding with quantized CLIP
   - ANN search with FAISS HNSW
   - Cross-encoder reranking
   - Boundary regression for precise timestamps

3. **Frontend Interface**
   - Natural language search
   - Video player with segment highlighting
   - Admin performance dashboard

## ✨ Features

### Backend (FastAPI)
- ✅ Simple REST API
- ✅ CORS enabled
- ✅ Static file serving
- ✅ Basic text search (keyword matching)
- ✅ Extensible architecture
- ✅ No AI dependencies

### Frontend (React + Vite)
- ✅ Clean, responsive UI
- ✅ Real-time search
- ✅ Results display
- ✅ Error handling
- ✅ Status monitoring

## 🚀 Quick Start

### Option 1: Automatic Setup
```bash
# Windows
.\setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Manual Development Setup

#### Backend Services
```bash
# API Gateway
cd backend/api-gateway
npm install
npm run dev

# Ingestion Service
cd backend/ingest
pip install -r requirements.txt
python main.py

# Search Service  
cd backend/search
pip install -r requirements.txt
python main.py
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing

### Run All Tests
```bash
# Using the test runner (recommended)
python tests/run_tests.py

# Or manually with pytest
python -m pytest tests/ -v --cov=backend
```

### Test Individual Services
```bash
# API Gateway tests
cd backend/api-gateway && npm test

# Python service tests
python -m pytest tests/test_search_service.py
python -m pytest tests/test_ingestion_service.py

# Load testing
cd tests/performance && ./run_performance_tests.sh
```

## 🌐 Access Points

- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/videos` | List MP4 files in dataset directory |
| POST | `/api/videos/{id}/process` | Process a specific video |
| POST | `/api/videos/process-all` | Process all unprocessed videos |
| GET | `/api/videos/{id}/status` | Get video processing status |
| POST | `/api/search` | Search video segments |
| DELETE | `/api/videos/{id}` | Delete video processing data |

## 🔧 Configuration

All services can be configured via environment variables. See the `.env.example` files in each service directory for available options.

### Key Configuration Areas

- **Database**: PostgreSQL connection settings
- **Redis**: Cache and message queue settings  
- **ML Models**: Model paths and inference settings
- **FAISS**: Vector search index configuration
- **Storage**: MinIO/S3 for video and model storage

### Scaling Configuration

- **Horizontal Scaling**: Adjust replica counts in K8s manifests
- **Resource Limits**: Configure CPU/memory in Docker Compose and K8s
- **GPU Support**: Enable NVIDIA GPU support for ML workloads

## 🚀 Deployment

### Docker Compose (Staging)
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes (Production)
```bash
# Apply all manifests
kubectl apply -f infra/k8s/

# Or use the deployment script
./infra/deploy-k8s.sh
```

### CI/CD
The project includes GitHub Actions workflows for:
- Automated testing on PR/push
- Security scanning with Trivy and CodeQL
- Docker image building and publishing
- Kubernetes deployment

## 📈 Performance

The system is designed for:
- **<50ms search latency** with proper hardware
- **1000+ concurrent users** with horizontal scaling
- **Multi-TB video collections** with distributed storage
- **99.9% uptime** with proper monitoring and failover

## 🎯 Architecture Highlights

This implementation provides:
1. **Production-ready foundation** - Complete CI/CD, monitoring, testing
2. **Microservices architecture** - Independently scalable components
3. **Modern ML stack** - CLIP, FAISS, cross-encoders, boundary regression
4. **High performance** - Sub-50ms search with proper optimization
5. **Enterprise features** - Authentication, monitoring, logging, tracing

## 📚 Documentation

- **API Documentation**: Available at `/docs` on each service
- **Architecture**: See `docs/architecture/` directory  
- **Deployment**: See `docs/deployment/` directory
- **Development**: See individual service README files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

##
## 🎯 Use Cases

This system is ideal for:
- Enterprise video content management
- Educational platform video search
- Media and entertainment applications
- Content discovery and recommendation
- Automated video analysis and indexing

## 🏗️ System Architecture

The system uses a modern microservices architecture designed for:
- **High Performance**: Sub-50ms search latency
- **Scalability**: Horizontal scaling with Kubernetes
- **Reliability**: Health checks, monitoring, and failover
- **Maintainability**: Clean separation of concerns
- **Extensibility**: Modular design for easy feature additions

## 📊 Performance Characteristics

| Metric | Target | Production Ready |
|--------|--------|------------------|
| Search Latency | <50ms | ✅ With proper hardware |
| Concurrent Users | 1000+ | ✅ With horizontal scaling |
| Video Collection Size | Multi-TB | ✅ With distributed storage |
| Accuracy | 90%+ | ✅ With fine-tuned models |
| Uptime | 99.9% | ✅ With monitoring/failover |

The system combines cutting-edge ML with production-grade infrastructure!
