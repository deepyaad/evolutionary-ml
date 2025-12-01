# Quick Start: Parallel Evolution with RunPod

## What Was Created

1. **`runpod_handler.py`** - Serverless function that trains a single solution
2. **`parallel_evolution.py`** - Client for managing parallel training jobs
3. **`main_parallel.py`** - Main script for parallel evolution
4. **`evo_v5.py`** - Added `evolve_parallel()` method
5. **`Dockerfile`** - Container image for RunPod
6. **`runpod_config.yaml`** - RunPod deployment configuration
7. **`requirements.txt`** - Python dependencies

## Setup Steps

### 1. Install Dependencies

```bash
pip install runpod
```

### 2. Get RunPod Credentials

1. Sign up at https://www.runpod.io
2. Get API key from dashboard
3. Set environment variables:

```bash
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"  # You'll get this after deployment
```

### 3. Deploy Handler

```bash
cd workflow/recurrent
runpod build
runpod deploy
```

**Note**: The RunPod CLI commands may vary. Check RunPod documentation for the latest deployment process.

### 4. Run Parallel Evolution

```bash
python main_parallel.py
```

## Expected Performance

With **50 parallel workers**:
- **30 minutes**: ~310 solutions (10 initial + 300 from 6 generations)
- **8 hours**: ~4,960 solutions (10 initial + 4,950 from 100 generations)

## Adjusting Batch Size

Edit `main_parallel.py`:

```python
env.evolve_parallel(
    batch_size=50,  # Change this: 20, 50, 100, 200
    ...
)
```

## Important Notes

1. **RunPod API**: The `parallel_evolution.py` uses a generic RunPod client interface. You may need to adjust the API calls based on the actual RunPod Python SDK version you're using.

2. **Dataset Path**: Ensure the dataset path is accessible in the RunPod container. You may need to:
   - Mount a volume with the dataset
   - Or upload the dataset to cloud storage and download it in the handler

3. **Cost Monitoring**: Monitor your RunPod usage in the dashboard to avoid unexpected costs.

4. **Testing**: Start with `batch_size=20` and `time_limit=1800` (30 min) to test before running full 8-hour runs.

## Troubleshooting

- **Jobs not starting**: Check RunPod endpoint status and logs
- **Import errors**: Ensure all dependencies are in `requirements.txt`
- **Dataset not found**: Verify dataset path and volume mounts
- **API errors**: Check RunPod SDK documentation for correct API usage

