@echo off
echo 🚀 Starting Video Retrieval Backend Services...

echo 📊 Starting with Docker Compose...
echo 📡 API Gateway will be available at: http://localhost:8000
echo 🔍 Search Service will be available at: http://localhost:8002
echo � Ingestion Service will be available at: http://localhost:8001
echo 📖 API docs will be available at: http://localhost:8000/docs
echo 📈 Grafana will be available at: http://localhost:3001
echo.
echo Press Ctrl+C to stop all services

docker-compose up --build
