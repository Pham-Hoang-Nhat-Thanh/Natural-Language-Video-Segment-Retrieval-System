@echo off
echo 🚀 Starting Complete Video Retrieval System...
echo ==============================================

echo.
echo 🐳 Using Docker Compose for full system startup
echo.
echo 📡 Services that will be started:
echo    - API Gateway (Port 8000)
echo    - Ingestion Service (Port 8001) 
echo    - Search Service (Port 8002)
echo    - Frontend (Port 3000)
echo    - PostgreSQL Database (Port 5432)
echo    - Redis Cache (Port 6379)
echo    - MinIO Storage (Port 9000)
echo    - Prometheus (Port 9090)
echo    - Grafana (Port 3001)
echo    - Jaeger (Port 16686)
echo.
echo 🌐 Access points after startup:
echo    - Frontend: http://localhost:3000
echo    - API Docs: http://localhost:8000/docs
echo    - Grafana: http://localhost:3001 (admin/admin123)
echo    - Prometheus: http://localhost:9090
echo    - Jaeger: http://localhost:16686
echo.
echo ⚠️  First startup may take several minutes to download images
echo 📋 Use 'docker-compose logs -f' to view real-time logs
echo 🛑 Press Ctrl+C to stop all services
echo.

docker-compose up --build
