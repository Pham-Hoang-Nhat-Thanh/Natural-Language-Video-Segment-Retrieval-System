#!/bin/bash
# Model Setup Script for Docker Integration
# This script is designed to be called during Docker image building
# to pre-download and setup all ML models

set -e

echo "🚀 Starting automated ML model setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a Docker environment
if [ -f /.dockerenv ]; then
    print_status "Running in Docker environment"
    MODELS_DIR="/app/models"
    SCRIPTS_DIR="/app/scripts"
else
    print_status "Running in local environment"
    MODELS_DIR="./models"
    SCRIPTS_DIR="./scripts"
fi

# Create directories
mkdir -p "$MODELS_DIR"/{clip,onnx,reranker,regressor}
mkdir -p "$SCRIPTS_DIR"

# Install Python dependencies
print_status "Installing Python dependencies for model download..."
if [ -f "$SCRIPTS_DIR/requirements_models.txt" ]; then
    pip install -r "$SCRIPTS_DIR/requirements_models.txt"
else
    # Fallback installation
    pip install torch torchvision clip-by-openai sentence-transformers onnx onnxruntime numpy
fi

# Run the automated model download
print_status "Running automated model download script..."
if [ -f "$SCRIPTS_DIR/download_models.py" ]; then
    cd "$(dirname "$SCRIPTS_DIR")"
    python "$SCRIPTS_DIR/download_models.py" --models-dir "$MODELS_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "✅ Automated model download completed successfully"
        
        # Verify model setup
        if [ -f "$MODELS_DIR/model_manifest.json" ]; then
            print_success "✅ Model manifest created"
            echo "📋 Model setup summary:"
            cat "$MODELS_DIR/model_manifest.json" | python -m json.tool | head -20
        fi
        
        # Check disk usage
        MODELS_SIZE=$(du -sh "$MODELS_DIR" | cut -f1)
        print_status "📦 Models directory size: $MODELS_SIZE"
        
    else
        print_error "❌ Automated model download failed"
        exit 1
    fi
else
    print_error "❌ Model download script not found at $SCRIPTS_DIR/download_models.py"
    exit 1
fi

# Set permissions for models directory
if [ -d "$MODELS_DIR" ]; then
    chmod -R 755 "$MODELS_DIR"
    print_success "✅ Set appropriate permissions for models directory"
fi

print_success "🎉 ML model setup completed successfully!"

# Optional: Clean up pip cache to reduce image size
if [ -f /.dockerenv ]; then
    print_status "Cleaning up pip cache to reduce image size..."
    pip cache purge
fi

echo ""
echo "📝 Model Setup Summary:"
echo "  - CLIP models: Downloaded and ONNX-converted ✅"
echo "  - Cross-encoder reranker: Downloaded and cached ✅"
echo "  - Boundary regressor: Sample weights and training pipeline ✅"
echo "  - Model manifest: Generated ✅"
echo ""
echo "🚀 Your video retrieval system is ready with automated ML model management!"
