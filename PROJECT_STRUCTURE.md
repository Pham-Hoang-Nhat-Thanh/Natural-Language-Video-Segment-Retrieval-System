# Project Structure

This document describes the layout of the Natural Language Video Segment Retrieval System, outlining key directories and files.

## Repository Layout

```
/
├── README.md                   # Main project documentation
├── USER_GUIDE.md               # Detailed user guide for manual workflow
├── PROJECT_STRUCTURE.md        # This file: project layout overview
├── IMPLEMENTATION_SUMMARY.md   # Summary of implementation details and design
├── CLEANUP_SUMMARY.md          # Summary of refactoring and cleanup steps
├── GITHUB_UPLOAD_GUIDE.md      # Instructions to upload the project to GitHub
├── SYSTEM_VERIFICATION.md      # Report on final system verification
├── STORAGE_ARCHITECTURE.md     # Storage architecture and data flow
├── docker-compose.yml          # Docker Compose configuration for all services
├── setup.sh                    # Shell setup script (Linux/Mac)
├── setup.bat                   # Batch setup script (Windows)
├── frontend/                   # Next.js frontend application
│   ├── components/             # React components for UI
│   ├── lib/                    # Client API helpers and utilities
│   ├── types/                  # TypeScript type definitions
│   ├── .env.example            # Example env file for frontend
│   └── ...                     # Other frontend files
├── backend/                    # Backend microservices (FastAPI, Node.js)
│   ├── api-gateway/            # Node.js API Gateway (Fastify)
│   │   └── src/app.js          # Main gateway application
│   ├── ingest/                 # Video ingestion service (FastAPI)
│   │   └── main.py             # Ingestion entry point
│   └── search/                 # Video search service (FastAPI)
│       └── main.py             # Search entry point
└── data/                       # Data storage (videos, thumbnails, metadata, embeddings)
```

## Technology Stack

### Frontend
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS + shadcn/ui components
- **State Management**: React Query + Zustand
- **Video Player**: Video.js or custom HTML5 player

### Backend Services
- **API Gateway**: Node.js + Fastify
- **ML Services**: Python + FastAPI
- **Vector Store**: FAISS with HNSW indexing
- **Models**: ONNX Runtime with quantized CLIP

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes
- **Database**: PostgreSQL + Redis
- **Storage**: MinIO (S3-compatible)
- **Monitoring**: Prometheus + Grafana + Jaeger

### CI/CD
- **Pipeline**: GitHub Actions
- **Testing**: Jest, Pytest, k6
- **Security**: Trivy, Snyk

## Development Phases

### Sprint 1: Infrastructure Setup
- [x] Project structure creation
- [ ] Docker environment setup
- [ ] CI/CD pipeline configuration
- [ ] Local development environment

### Sprint 2: Offline Pipeline
- [ ] Shot detection service
- [ ] Keyframe extraction
- [ ] CLIP model optimization
- [ ] Embedding service
- [ ] FAISS index builder

### Sprint 3: Online ML Services
- [ ] Text encoder service
- [ ] ANN search microservice
- [ ] Cross-encoder reranker
- [ ] Boundary regression service

### Sprint 4: Frontend & Integration
- [ ] Next.js application setup
- [ ] Search interface
- [ ] Video player component
- [ ] Admin dashboard

### Sprint 5: Optimization & QA
- [ ] Performance tuning
- [ ] Load testing
- [ ] Security audit
- [ ] Documentation
