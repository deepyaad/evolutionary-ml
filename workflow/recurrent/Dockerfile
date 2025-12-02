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
RUN mkdir -p /workspace
# Copy dataset from the repo (relative to build context)
# The build context should be set to the repo root, so this path should work
COPY ../../datasets/spotify_dataset.npz /workspace/dataset.npz

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set working directory for RunPod (RunPod uses /workspace as default)
WORKDIR /workspace

# Default command
CMD ["python", "/app/runpod_handler.py"]

