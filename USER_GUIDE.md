# User Guide

This guide explains how to use the Natural Language Video Segment Retrieval System with your own MP4 videos. The system uses a manual workflow: you place MP4 files in a dataset directory, services process them, and you can search video segments via web UI or API.

## Prerequisites
- Docker and Docker Compose installed (for one-command startup)
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

## Directory Structure
All data is stored under the `data/` directory:

```
data/
├── videos/
│   └── datasets/
│       └── custom/           # Place your MP4 files here (read-only)
├── thumbnails/
│   └── datasets/
│       └── custom/           # Auto-generated thumbnails per video
├── metadata/                 # Auto-generated processing metadata
└── embeddings/               # Auto-generated vector indices
```

## 1. Add Videos
1. Copy your MP4 files (max 1GB each) into `data/videos/datasets/custom/`:
   ```bash
   cp /path/to/my_video.mp4 data/videos/datasets/custom/
   ```
2. Each filename (without `.mp4`) is treated as its video ID.

## 2. Start the System
Use Docker Compose for all services:
```bash
# At project root
docker-compose up -d
```
This command starts:
- API Gateway (port 8000)
- Ingestion Service (port 8001)
- Search Service (port 8002)
- Redis and PostgreSQL

## 3. Process Videos
The ingestion service does not auto-scan on startup. To process videos:

- **Process all unprocessed videos**:
  ```bash
  curl -X POST http://localhost:8000/api/videos/process-all
  ```
- **Process a single video**:
  ```bash
  curl -X POST http://localhost:8000/api/videos/{video_id}/process
  ```

The endpoint returns a 202 Accepted and begins asynchronous processing:
- Shot detection with enhanced visual features
- Keyframe extraction with multi-modal analysis
- **🧠 Enhanced Feature Extraction (NEW)**:
  - Object detection (YOLO/DETR models when available)
  - Scene classification with visual context
  - OCR text extraction from video frames
  - Color histogram and visual feature analysis
- Thumbnail generation
- **Enhanced embedding creation** with multi-modal features
- Metadata storage in PostgreSQL with rich feature data

## 4. Check Processing Status
- **All videos**: `GET http://localhost:8000/api/videos/status` returns status of each video
- **Single video**: `GET http://localhost:8000/api/videos/{video_id}/status`

## 5. Search Video Segments
Once videos are processed and embeddings are in place, search via web UI or API:

### 🧠 Enhanced Search Features (NEW)
The system now includes advanced AI-powered search capabilities:

#### Web Interface
- Open `http://localhost:3000` in your browser
- **Enhanced Search**: Use the improved search with automatic query enhancement
- **Query Suggestions**: Get real-time query improvement suggestions
- **Advanced Results**: View enhanced metadata including objects, scenes, and confidence scores
- Enter natural language queries like:
  - "dog playing fetch in the park" → Enhanced to include scene context
  - "sunset on beach" → Expanded with related visual and temporal elements
  - "car driving through city" → Enhanced with action and location understanding

#### Standard Search API
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "beach sunset", "limit": 5}'
```

#### 🚀 Enhanced Search API (NEW)
```bash
curl -X POST http://localhost:8000/api/search/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "beach sunset", 
    "top_k": 5, 
    "use_llm_enhancement": true,
    "threshold": 0.5
  }'
```

#### Query Enhancement API (NEW)
```bash
curl -X POST http://localhost:8000/api/query/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "query": "beach sunset",
    "use_llm": true
  }'
```

#### Enhanced Response Format
All enhanced search responses now include:
- `video_id`, `start_time`, `end_time` (standard)
- `thumbnail_url` (standard)
- `enhanced_query_used` - The improved query that was actually searched
- `query_analysis` - Detected entities, actions, scene context, and confidence
- `enhanced_features` - Objects, scenes, and text detected in the video segment
Response contains top matching segments with:
- `video_id`
- `start_time` / `end_time`
- `thumbnail_url`

## 6. Delete Video Data
To remove processed data for a video:
```bash
curl -X DELETE http://localhost:8000/api/videos/{video_id}
```
This deletes thumbnails, metadata, embeddings for that video.

## Troubleshooting
- Ensure `.env` files are configured correctly:
  - API_GATEWAY_URL, INGEST_SERVICE_URL, SEARCH_SERVICE_URL
- Check service logs:
  ```bash
docker-compose logs -f api-gateway ingest search
```
- Verify data directories have correct permissions (read-only for MP4s, read-write for others)

## Advanced: Local Development (without Docker)
1. **API Gateway**
   ```bash
   cd backend/api-gateway
   npm install
   npm run dev
   ```
2. **Ingestion Service**
   ```bash
   cd backend/ingest
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8001
   ```
3. **Search Service**
   ```bash
   cd backend/search
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8002
   ```
4. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

Your system is now ready for everyday use: add MP4 files, process them, and search video segments with natural language!
