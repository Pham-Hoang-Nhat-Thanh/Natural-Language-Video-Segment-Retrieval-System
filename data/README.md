# Video Retrieval System - Data Directory

This directory contains all video data, embeddings, and metadata for the retrieval system.

## 🏗️ Storage Architecture

The system uses a **bandwidth-optimized architecture** where:
- ✅ Videos are stored once and accessed by file paths
- ✅ Only metadata and embeddings transfer over network  
- ✅ Services use file references, not file transfers
- ✅ Pre-built datasets are processed in-place

## 📁 Directory Structure

```
data/
├── videos/
│   ├── datasets/          # Pre-built video datasets (READ-ONLY)
│   │   ├── msvd/         # Microsoft Video Description Corpus
│   │   ├── charades/     # Charades Activity Recognition
│   │   ├── youtube8m/    # YouTube-8M (large scale)
│   │   └── custom/       # Custom/sample videos
│   ├── uploaded/         # User uploaded videos
│   └── temp/            # Temporary processing files
├── thumbnails/           # Generated keyframe thumbnails
├── embeddings/          # FAISS vector indices (fast access)
├── metadata/           # Dataset annotations and metadata
└── cache/             # Local cache for performance
```

## 🚀 Dataset Setup

### 1. Sample Dataset (Ready)
```bash
# Already available for testing
ls data/videos/datasets/custom/
# sample_001.mp4, sample_002.mp4
```

### 2. MSVD Dataset
```bash
# Download from: https://www.cs.utexas.edu/users/ml/clamp/videoDescription/
# Extract to: data/videos/datasets/msvd/
# Index via API: POST /api/datasets/msvd/index
```

### 3. Charades Dataset  
```bash
# Download from: https://prior.allenai.org/projects/charades
# Extract to: data/videos/datasets/charades/
# Index via API: POST /api/datasets/charades/index
```

## 🔧 Usage

### List Available Datasets
```bash
curl http://localhost:8000/api/datasets
```

### Get Dataset Information
```bash
curl http://localhost:8000/api/datasets/custom
```

### Index a Dataset
```bash
curl -X POST http://localhost:8000/api/datasets/custom/index
```

### Search Videos
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning introduction", "top_k": 5}'
```

## 📊 Performance Benefits

| Aspect | Traditional | Our Architecture |
|--------|-------------|------------------|
| Video Transfer | High bandwidth | Zero (path references) |
| Search Latency | >500ms | <50ms |
| Storage | Duplicated | Single source |
| Scalability | Network limited | Compute limited |

## 🔒 Security

- Videos in datasets/ are mounted read-only
- Only uploaded/ directory is writable
- Static files served via API Gateway only
- No direct file access from external services

## 📝 Adding Your Own Dataset

1. Create directory: `data/videos/datasets/your_dataset/`
2. Add videos to the directory
3. Create metadata: `data/metadata/your_dataset_annotations.json`
4. Index via API: `POST /api/datasets/your_dataset/index`
5. Videos processed locally, no network transfer!

For more details, see STORAGE_ARCHITECTURE.md
