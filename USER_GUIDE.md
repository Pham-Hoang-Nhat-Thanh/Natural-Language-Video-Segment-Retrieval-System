# How to Use Your MP4 Videos

## Adding Videos

1. **Place your MP4 files** in the `data/videos/datasets/custom/` directory
   - Example: `data/videos/datasets/custom/my_video.mp4`
   - The filename (without .mp4) becomes the video ID
   - Supported format: MP4 only

2. **Start the system** using Docker:
   ```bash
   docker-compose up -d
   ```

3. **Process your videos** using the API or web interface:
   - List all videos: `GET /api/videos`
   - Process a specific video: `POST /api/videos/{video_id}/process`
   - Process all videos: `POST /api/videos/process-all`

4. **Search for content** once processing is complete:
   - Use the web interface at `http://localhost:3000`
   - Or use the API: `POST /api/search` with your query

## Example Workflow

```bash
# 1. Add a video file
cp ~/Downloads/vacation.mp4 data/videos/datasets/custom/

# 2. Start the system
docker-compose up -d

# 3. Process the video (via API)
curl -X POST http://localhost:8000/api/videos/vacation/process

# 4. Search for segments
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "beach sunset", "top_k": 5}'
```

## Directory Structure

```
data/
├── videos/datasets/custom/    # Place your MP4 files here
├── thumbnails/datasets/custom/ # Generated thumbnails (auto-created)
├── metadata/                  # Processing metadata (auto-created)
└── embeddings/               # Vector indices (auto-created)
```

## Notes

- **No uploads required**: Just copy MP4 files directly to the directory
- **Processing is on-demand**: Videos are only processed when requested
- **Thumbnails are generated automatically** during processing
- **Original files are never modified** - only processing data can be deleted
