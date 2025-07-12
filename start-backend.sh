#!/bin/bash
echo "🚀 Starting Video Retrieval Backend Services..."

echo "📊 Starting with Docker Compose..."
echo "📡 API Gateway will be available at: http://localhost:8090"
echo "🔍 Search Service will be available at: http://localhost:8052"
echo "📥 Ingestion Service will be available at: http://localhost:8051"
echo "📖 API docs will be available at: http://localhost:8090/docs"
echo "📈 Grafana will be available at: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all services"

docker compose up --build
