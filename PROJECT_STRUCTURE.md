# Natural Language-Driven Video Segment Retrieval - Project Structure

## Overview
Production-ready, full-stack application for natural language video segment retrieval with <50ms query latency, built with modern microservices architecture and ML-powered search capabilities.

## Repository Structure

```
/
├── README.md                          # Main project documentation
├── IMPLEMENTATION_SUMMARY.md          # Complete implementation overview
├── PROJECT_STRUCTURE.md              # This file
├── docker-compose.yml                # Complete development environment
├── setup.sh / setup.bat              # Cross-platform setup scripts
├── .github/
│   └── workflows/                     # CI/CD pipelines
│       ├── ci-cd.yml                 # Main CI/CD pipeline
│       └── security.yml              # Security scanning
├── frontend/                          # Next.js 14 frontend application
│   ├── app/                          # Next.js App Router
│   │   ├── globals.css               # Global styles
│   │   ├── layout.tsx                # Root layout
│   │   ├── page.tsx                  # Main search page
│   │   ├── manage/                   # Video management interface
│   │   └── admin/                    # Admin dashboard
│   ├── components/                    # React components
│   │   ├── Header.tsx                # Site header
│   │   ├── SearchInterface.tsx       # Main search interface
│   │   ├── SearchBar.tsx            # Search input component
│   │   ├── ResultsList.tsx          # Search results display
│   │   └── VideoPlayer.tsx          # Video player with segments
│   ├── lib/                          # Utility functions
│   │   ├── api.ts                    # API client
│   │   └── utils.ts                  # Helper functions
│   ├── types/                        # TypeScript definitions
│   │   └── index.ts                  # Shared type definitions
│   ├── package.json
│   ├── next.config.js
│   └── tailwind.config.js
├── backend/                           # Backend microservices
│   ├── api-gateway/                   # Node.js API Gateway (Fastify)
│   │   ├── src/
│   │   │   └── app.js                # Main gateway application
│   │   ├── package.json
│   │   ├── Dockerfile
│   │   └── Dockerfile.dev
│   ├── ingest/                        # Python video ingestion service
│   │   ├── main.py                   # FastAPI ingestion service
│   │   ├── shot_detector.py          # Shot boundary detection
│   │   ├── keyframe_extractor.py     # Keyframe extraction
│   │   ├── embedding_service.py      # Multimodal embeddings
│   │   ├── config.py                 # Configuration management
│   │   ├── database.py               # Database operations
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── search/                        # Python search service
│       ├── main.py                   # FastAPI search service
│       ├── text_encoder.py           # Text encoding with caching
│       ├── ann_search.py             # FAISS ANN search engine
│       ├── reranker.py               # Cross-encoder reranking
│       ├── boundary_regressor.py     # Neural boundary refinement
│       ├── database.py               # Search database operations
│       ├── config.py                 # Configuration management
│       ├── requirements.txt
│       ├── Dockerfile
│       └── Dockerfile.dev
├── models/                            # ML models and configurations
│   ├── clip/                          # CLIP model files
│   ├── reranker/                      # Cross-encoder model
│   ├── regressor/                     # Boundary regression model
│   └── onnx/                          # ONNX optimized models
├── infra/                             # Infrastructure and deployment
│   ├── k8s/                           # Kubernetes manifests
│   │   ├── 00-namespace-config.yml    # Namespace and RBAC
│   │   ├── 01-frontend-api.yml        # Frontend and API Gateway
│   │   ├── 02-ml-services.yml         # ML microservices
│   │   ├── 03-infrastructure.yml      # Database, Redis, Storage
│   │   └── 04-monitoring.yml          # Monitoring stack
│   ├── helm/                          # Helm charts
│   ├── terraform/                     # Infrastructure as code
│   └── monitoring/                    # Prometheus, Grafana configs
├── data/                              # Optimized storage architecture
│   ├── videos/                        # Video storage (bandwidth optimized)
│   │   ├── datasets/                 # Pre-built datasets (READ-ONLY)
│   │   │   └── custom/              # User's MP4 files (manually added)
│   ├── thumbnails/                   # Generated keyframe thumbnails
│   │   └── datasets/                # Thumbnails organized by dataset
│   │       └── custom/              # Thumbnails for user's videos
│   ├── embeddings/                   # FAISS vector indices (fast access)
│   │   └── custom.faiss             # Index for user videos
│   ├── metadata/                     # Video metadata and processing results
│   │   └── custom/                  # Processing metadata for user videos
│   ├── cache/                        # Local performance cache
│   ├── README.md                     # Data directory documentation
│   └── STORAGE_ARCHITECTURE.md      # Technical storage details
├── tests/                             # Test suites
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── load/                          # Load testing (k6)
└── docs/                              # Documentation
    ├── api/                           # API documentation
    ├── architecture/                  # Architecture diagrams
    └── deployment/                    # Deployment guides
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
