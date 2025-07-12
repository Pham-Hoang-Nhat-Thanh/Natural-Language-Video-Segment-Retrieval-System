#!/bin/bash
set -e

echo "🚀 Setting up Natural Language Video Segment Retrieval System"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Set the compose command based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    print_success "Docker Compose is available"
}

# Create necessary directories
create_directories() {
    print_status "Creating data directories..."
    
    mkdir -p data/videos
    mkdir -p data/thumbnails
    mkdir -p data/embeddings
    mkdir -p models/clip
    mkdir -p models/onnx
    mkdir -p logs
    
    print_success "Directories created"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f .env ]; then
        cat > .env << EOL
# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/video_retrieval
REDIS_URL=redis://redis:6379

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
API_GATEWAY_PORT=8000
INGEST_SERVICE_PORT=8001
SEARCH_SERVICE_PORT=8002

# Model Configuration
MODEL_PATH=/app/models
CLIP_MODEL_NAME=ViT-B/32
USE_ONNX=true
DEVICE=auto
BATCH_SIZE=32

# Storage Configuration
DATA_PATH=/app/data
VIDEO_STORAGE_PATH=/app/data/videos
THUMBNAIL_STORAGE_PATH=/app/data/thumbnails

# Performance Configuration
MAX_VIDEO_SIZE_MB=1000
MAX_VIDEO_DURATION_MINUTES=120
PROCESSING_TIMEOUT_SECONDS=3600

# Development
NODE_ENV=development
DEBUG=true
LOG_LEVEL=INFO
EOL
        print_success "Environment file created (.env)"
    else
        print_warning "Environment file already exists"
    fi
}

# Download sample models (placeholder)
download_models() {
    print_status "Setting up ML models..."
    
    # In a real implementation, you would download pre-trained models here
    # For now, we'll create placeholder files
    
    mkdir -p models/clip
    mkdir -p models/onnx
    mkdir -p models/reranker
    mkdir -p models/regressor
    
    # Create model placeholders
    echo "# CLIP model will be downloaded automatically on first run" > models/clip/README.md
    echo "# ONNX models will be generated from PyTorch models" > models/onnx/README.md
    echo "# Cross-encoder reranker model" > models/reranker/README.md
    echo "# Boundary regression model" > models/regressor/README.md
    
    print_success "Model directories prepared"
}

# Initialize database
init_database() {
    print_status "Creating database initialization script..."
    
    mkdir -p infra/postgres
    
    cat > infra/postgres/init.sql << EOL
-- Create database if not exists
CREATE DATABASE IF NOT EXISTS video_retrieval;

-- Create user if not exists
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'video_user') THEN
      CREATE USER video_user WITH PASSWORD 'video_password';
   END IF;
END
\$\$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE video_retrieval TO video_user;

-- Switch to video_retrieval database
\c video_retrieval;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create tables will be handled by the application
EOL

    print_success "Database initialization script created"
}

# Create monitoring configuration
create_monitoring_config() {
    print_status "Setting up monitoring configuration..."
    
    mkdir -p infra/monitoring
    
    # Prometheus configuration
    cat > infra/monitoring/prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'ingest-service'
    static_configs:
      - targets: ['ingest-service:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'search-service'
    static_configs:
      - targets: ['search-service:8002']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOL

    # Grafana datasources
    mkdir -p infra/monitoring/grafana/datasources
    cat > infra/monitoring/grafana/datasources/prometheus.yml << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOL

    # Grafana dashboards
    mkdir -p infra/monitoring/grafana/dashboards
    cat > infra/monitoring/grafana/dashboards/dashboard.yml << EOL
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOL

    print_success "Monitoring configuration created"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Build images
    print_status "Building Docker images..."
    $COMPOSE_CMD build
    
    # Start services
    print_status "Starting services..."
    $COMPOSE_CMD up -d
    
    print_success "Services started"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for PostgreSQL..."
    until $COMPOSE_CMD exec postgres pg_isready -U postgres > /dev/null 2>&1; do
        sleep 2
    done
    print_success "PostgreSQL is ready"
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    until $COMPOSE_CMD exec redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Redis is ready"
    
    # Wait for API Gateway
    print_status "Waiting for API Gateway..."
    until curl -s http://localhost:8000/health > /dev/null; do
        sleep 5
    done
    print_success "API Gateway is ready"
    
    print_success "All services are ready!"
}

# Display service information
show_service_info() {
    echo ""
    echo "🎉 Natural Language Video Segment Retrieval System is running!"
    echo "=============================================================="
    echo ""
    echo "📊 Service URLs:"
    echo "  Frontend:          http://localhost:3000"
    echo "  API Gateway:       http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Grafana Dashboard: http://localhost:3001 (admin/admin123)"
    echo "  Prometheus:        http://localhost:9090"
    echo "  Jaeger Tracing:    http://localhost:16686"
    echo "  MinIO Console:     http://localhost:9001 (minioadmin/minioadmin123)"
    echo ""
    echo "💾 Database Access:"
    echo "  PostgreSQL:        localhost:5432 (postgres/password)"
    echo "  Redis:             localhost:6379"
    echo ""
    echo "📁 Useful Commands:"
    echo "  View logs:         $COMPOSE_CMD logs -f"
    echo "  Stop services:     $COMPOSE_CMD down"
    echo "  Restart services:  $COMPOSE_CMD restart"
    echo "  Update services:   $COMPOSE_CMD pull && $COMPOSE_CMD up -d"
    echo ""
    echo "🔧 Next Steps:"
    echo "  1. Place MP4 videos in data/videos/datasets/custom/ directory"
    echo "  2. Process videos through the ingestion service"
    echo "  3. Try searching with natural language queries"
    echo "  4. Check the admin dashboard for system metrics"
    echo ""
}

# Main execution
main() {
    print_status "Starting setup process..."
    
    check_docker
    check_docker_compose
    create_directories
    create_env_file
    download_models
    init_database
    create_monitoring_config
    start_services
    wait_for_services
    show_service_info
    
    print_success "Setup completed successfully! 🚀"
}

# Run main function
main "$@"
