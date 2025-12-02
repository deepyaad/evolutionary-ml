FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow (CPU version for now, can switch to GPU if needed)
RUN pip install tensorflow

# Install RunPod serverless
RUN pip install runpod

# Copy application code
COPY . .

# Copy dataset to a fixed location in the container
# This ensures the dataset is always available at /workspace/dataset.npz
# Note: When RunPod builds, it clones the repo, so the path should be relative to repo root
RUN mkdir -p /workspace
# The dataset should be copied during build - adjust path if needed based on your repo structure
# If the dataset isn't in the image, you'll need to mount it as a volume or download it

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set working directory for RunPod (RunPod uses /workspace as default)
WORKDIR /workspace

# Default command
CMD ["python", "/app/runpod_handler.py"]

