@echo off
setlocal enabledelayedexpansion

echo 🚀 Setting up Natural Language Video Segment Retrieval System
echo ==============================================================

REM Function to print colored output (simplified for Windows)
goto :main

:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:check_docker
call :print_status "Checking Docker..."
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker and try again."
    exit /b 1
)
call :print_success "Docker is running"
goto :eof

:check_docker_compose
call :print_status "Checking Docker Compose..."
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        call :print_error "Docker Compose is not installed"
        exit /b 1
    )
)
call :print_success "Docker Compose is available"
goto :eof

:create_directories
call :print_status "Creating data directories..."

if not exist "data\videos" mkdir "data\videos"
if not exist "data\thumbnails" mkdir "data\thumbnails"
if not exist "data\embeddings" mkdir "data\embeddings"
if not exist "models\clip" mkdir "models\clip"
if not exist "models\onnx" mkdir "models\onnx"
if not exist "logs" mkdir "logs"

call :print_success "Directories created"
goto :eof

:create_env_file
call :print_status "Creating environment configuration..."

if not exist ".env" (
    (
        echo # Database Configuration
        echo DATABASE_URL=postgresql://postgres:password@postgres:5432/video_retrieval
        echo REDIS_URL=redis://redis:6379
        echo.
        echo # API Configuration
        echo NEXT_PUBLIC_API_URL=http://localhost:8000
        echo API_GATEWAY_PORT=8000
        echo INGEST_SERVICE_PORT=8001
        echo SEARCH_SERVICE_PORT=8002
        echo.
        echo # Model Configuration
        echo MODEL_PATH=/app/models
        echo CLIP_MODEL_NAME=ViT-B/32
        echo USE_ONNX=true
        echo DEVICE=auto
        echo BATCH_SIZE=32
        echo.
        echo # Storage Configuration
        echo DATA_PATH=/app/data
        echo VIDEO_STORAGE_PATH=/app/data/videos
        echo THUMBNAIL_STORAGE_PATH=/app/data/thumbnails
        echo.
        echo # Performance Configuration
        echo MAX_VIDEO_SIZE_MB=1000
        echo MAX_VIDEO_DURATION_MINUTES=120
        echo PROCESSING_TIMEOUT_SECONDS=3600
        echo.
        echo # Development
        echo NODE_ENV=development
        echo DEBUG=true
        echo LOG_LEVEL=INFO
    ) > .env
    call :print_success "Environment file created (.env)"
) else (
    call :print_warning "Environment file already exists"
)
goto :eof

:download_models
call :print_status "Setting up ML models..."

if not exist "models\clip" mkdir "models\clip"
if not exist "models\onnx" mkdir "models\onnx"
if not exist "models\reranker" mkdir "models\reranker"
if not exist "models\regressor" mkdir "models\regressor"

echo # CLIP model will be downloaded automatically on first run > "models\clip\README.md"
echo # ONNX models will be generated from PyTorch models > "models\onnx\README.md"
echo # Cross-encoder reranker model > "models\reranker\README.md"
echo # Boundary regression model > "models\regressor\README.md"

call :print_success "Model directories prepared"
goto :eof

:init_database
call :print_status "Creating database initialization script..."

if not exist "infra\postgres" mkdir "infra\postgres"

(
    echo -- Create database if not exists
    echo CREATE DATABASE IF NOT EXISTS video_retrieval;
    echo.
    echo -- Create user if not exists
    echo DO $$
    echo BEGIN
    echo    IF NOT EXISTS ^(SELECT FROM pg_catalog.pg_roles WHERE rolname = 'video_user'^) THEN
    echo       CREATE USER video_user WITH PASSWORD 'video_password';
    echo    END IF;
    echo END
    echo $$;
    echo.
    echo -- Grant privileges
    echo GRANT ALL PRIVILEGES ON DATABASE video_retrieval TO video_user;
    echo.
    echo -- Switch to video_retrieval database
    echo \c video_retrieval;
    echo.
    echo -- Enable extensions
    echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    echo CREATE EXTENSION IF NOT EXISTS "pg_trgm";
) > "infra\postgres\init.sql"

call :print_success "Database initialization script created"
goto :eof

:create_monitoring_config
call :print_status "Setting up monitoring configuration..."

if not exist "infra\monitoring" mkdir "infra\monitoring"
if not exist "infra\monitoring\grafana\datasources" mkdir "infra\monitoring\grafana\datasources"
if not exist "infra\monitoring\grafana\dashboards" mkdir "infra\monitoring\grafana\dashboards"

REM Prometheus configuration
(
    echo global:
    echo   scrape_interval: 15s
    echo   evaluation_interval: 15s
    echo.
    echo scrape_configs:
    echo   - job_name: 'api-gateway'
    echo     static_configs:
    echo       - targets: ['api-gateway:8000']
    echo     metrics_path: '/metrics'
    echo     scrape_interval: 5s
    echo.
    echo   - job_name: 'ingest-service'
    echo     static_configs:
    echo       - targets: ['ingest-service:8001']
    echo     metrics_path: '/metrics'
    echo     scrape_interval: 10s
    echo.
    echo   - job_name: 'search-service'
    echo     static_configs:
    echo       - targets: ['search-service:8002']
    echo     metrics_path: '/metrics'
    echo     scrape_interval: 5s
) > "infra\monitoring\prometheus.yml"

call :print_success "Monitoring configuration created"
goto :eof

:start_services
call :print_status "Building and starting services..."

call :print_status "Building Docker images..."
docker-compose build

call :print_status "Starting services..."
docker-compose up -d

call :print_success "Services started"
goto :eof

:wait_for_services
call :print_status "Waiting for services to be ready..."

call :print_status "Waiting for PostgreSQL..."
:wait_postgres
docker-compose exec postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_postgres
)
call :print_success "PostgreSQL is ready"

call :print_status "Waiting for Redis..."
:wait_redis
docker-compose exec redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_redis
)
call :print_success "Redis is ready"

call :print_status "Waiting for API Gateway..."
:wait_api
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    timeout /t 5 /nobreak >nul
    goto wait_api
)
call :print_success "API Gateway is ready"

call :print_success "All services are ready!"
goto :eof

:show_service_info
echo.
echo 🎉 Natural Language Video Segment Retrieval System is running!
echo ==============================================================
echo.
echo 📊 Service URLs:
echo   Frontend:          http://localhost:3000
echo   API Gateway:       http://localhost:8000
echo   API Documentation: http://localhost:8000/docs
echo   Grafana Dashboard: http://localhost:3001 (admin/admin123)
echo   Prometheus:        http://localhost:9090
echo   Jaeger Tracing:    http://localhost:16686
echo   MinIO Console:     http://localhost:9001 (minioadmin/minioadmin123)
echo.
echo 💾 Database Access:
echo   PostgreSQL:        localhost:5432 (postgres/password)
echo   Redis:             localhost:6379
echo.
echo 📁 Useful Commands:
echo   View logs:         docker-compose logs -f
echo   Stop services:     docker-compose down
echo   Restart services:  docker-compose restart
echo   Update services:   docker-compose pull ^&^& docker-compose up -d
echo.
echo 🔧 Next Steps:
echo   1. Place MP4 videos in data\videos\datasets\custom\ directory
echo   2. Process videos through the ingestion service
echo   3. Try searching with natural language queries
echo   4. Check the admin dashboard for system metrics
echo.
goto :eof

:main
call :print_status "Starting setup process..."

call :check_docker
if errorlevel 1 exit /b 1

call :check_docker_compose
if errorlevel 1 exit /b 1

call :create_directories
call :create_env_file
call :download_models
call :init_database
call :create_monitoring_config
call :start_services
call :wait_for_services
call :show_service_info

call :print_success "Setup completed successfully! 🚀"
goto :eof
