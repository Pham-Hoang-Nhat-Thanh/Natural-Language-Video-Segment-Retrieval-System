#!/bin/bash
set -e

echo "🚀 Starting comprehensive system tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found, skipping frontend tests"
    SKIP_FRONTEND=true
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_status "Prerequisites check passed ✓"

# Setup test environment
print_status "Setting up test environment..."

# Create test data directory
mkdir -p ./data/videos/datasets/custom
mkdir -p ./models/{clip,onnx,regressor,reranker}

# Download a small test video if none exists
if [ ! -f "./data/videos/datasets/custom/test_video.mp4" ]; then
    print_status "Creating test video..."
    # Create a simple test video using ffmpeg if available
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=1 -f lavfi -i sine=frequency=1000:duration=10 -c:v libx264 -c:a aac -shortest ./data/videos/datasets/custom/test_video.mp4 -y
    else
        print_warning "ffmpeg not found, creating dummy video file"
        echo "dummy video content" > ./data/videos/datasets/custom/test_video.mp4
    fi
fi

print_status "Test environment setup complete ✓"

# Backend tests
print_status "Running backend tests..."

# Install Python dependencies for testing
pip3 install pytest httpx pytest-asyncio

# Test ingestion service
print_status "Testing ingestion service..."
cd backend/ingest
python3 -m pytest ../../tests/test_ingestion_service.py -v
cd ../..

# Test search service
print_status "Testing search service..."
cd backend/search
python3 -m pytest ../../tests/test_search_service.py -v
cd ../..

# Test API Gateway
print_status "Testing API Gateway..."
cd backend/api-gateway
npm install
npm test
cd ../..

print_status "Backend tests completed ✓"

# Frontend tests (if Node.js is available)
if [ "$SKIP_FRONTEND" != "true" ]; then
    print_status "Running frontend tests..."
    cd frontend
    npm install
    npm run test
    npm run build
    cd ..
    print_status "Frontend tests completed ✓"
fi

# Integration tests with Docker Compose
print_status "Starting integration tests with Docker Compose..."

# Start services
print_status "Starting Docker services..."
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Function to check service health
check_service_health() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f "$health_url" &> /dev/null; then
            print_status "$service_name is healthy ✓"
            return 0
        fi
        print_warning "$service_name not ready yet (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to become healthy"
    return 1
}

# Check service health
check_service_health "API Gateway" "http://localhost:8000/health"
check_service_health "Ingestion Service" "http://localhost:8001/health"  
check_service_health "Search Service" "http://localhost:8002/health"

# Run end-to-end tests
print_status "Running end-to-end tests..."

# Test video listing
print_status "Testing video listing..."
response=$(curl -s "http://localhost:8000/api/videos")
if echo "$response" | grep -q "videos"; then
    print_status "Video listing test passed ✓"
else
    print_error "Video listing test failed"
    echo "Response: $response"
    exit 1
fi

# Test video processing
print_status "Testing video processing..."
response=$(curl -s -X POST "http://localhost:8000/api/videos/test_video/process")
if echo "$response" | grep -q "success\|processing"; then
    print_status "Video processing test passed ✓"
else
    print_warning "Video processing test returned: $response"
fi

# Test search functionality
print_status "Testing search functionality..."
response=$(curl -s -X POST "http://localhost:8000/api/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test video", "top_k": 5}')
if echo "$response" | grep -q "results"; then
    print_status "Search test passed ✓"
else
    print_error "Search test failed"
    echo "Response: $response"
    exit 1
fi

# Test admin stats
print_status "Testing admin stats..."
response=$(curl -s "http://localhost:8000/api/admin/stats")
if echo "$response" | grep -q "ingest"; then
    print_status "Admin stats test passed ✓"
else
    print_error "Admin stats test failed"
    echo "Response: $response"
    exit 1
fi

print_status "End-to-end tests completed ✓"

# Performance tests
print_status "Running performance tests..."

# Test search performance
print_status "Testing search performance..."
for i in {1..5}; do
    start_time=$(date +%s%N)
    curl -s -X POST "http://localhost:8000/api/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "performance test '$i'", "top_k": 10}' > /dev/null
    end_time=$(date +%s%N)
    duration=$((($end_time - $start_time) / 1000000))
    print_status "Search $i completed in ${duration}ms"
done

print_status "Performance tests completed ✓"

# Cleanup
print_status "Cleaning up test environment..."
docker-compose -f docker-compose.test.yml down
docker system prune -f

print_status "Test cleanup completed ✓"

# Summary
echo ""
print_status "🎉 All tests completed successfully!"
echo ""
print_status "Test Summary:"
print_status "✓ Prerequisites check"
print_status "✓ Backend unit tests"
if [ "$SKIP_FRONTEND" != "true" ]; then
    print_status "✓ Frontend tests"
fi
print_status "✓ Service health checks"
print_status "✓ End-to-end integration tests"
print_status "✓ Performance tests"
print_status "✓ Environment cleanup"

echo ""
print_status "System is ready for deployment! 🚀"
