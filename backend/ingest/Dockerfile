FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy model setup scripts from parent directory
COPY ../../scripts/download_models.py /app/scripts/download_models.py
COPY ../../scripts/requirements_models.txt /app/scripts/requirements_models.txt
COPY ../../scripts/setup_models_docker.sh /app/scripts/setup_models_docker.sh

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/scripts

# Setup ML models automatically during build
RUN chmod +x /app/scripts/setup_models_docker.sh && \
    /app/scripts/setup_models_docker.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash ingest
RUN chown -R ingest:ingest /app
USER ingest

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
