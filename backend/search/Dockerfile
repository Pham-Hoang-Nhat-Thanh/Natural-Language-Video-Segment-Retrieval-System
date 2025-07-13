FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Skip pip upgrade to avoid filesystem conflicts and use existing pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies first
RUN pip install --no-cache-dir --retries 3 --timeout 60 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy source code
COPY . .

# Create directories for models and index
RUN mkdir -p /app/models /app/faiss_index /app/scripts

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Start production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
