# Parallel Evolution with RunPod

This guide explains how to set up and run parallel evolution using RunPod serverless.

## Overview

The parallel evolution system allows you to train multiple neural network solutions simultaneously on RunPod, dramatically increasing throughput from ~117 solutions in 8 hours to **300+ solutions in 30 minutes** (or **10,000+ in 8 hours** with sufficient capacity).

## Architecture

1. **RunPod Handler** (`runpod_handler.py`): Serverless function that trains a single solution
2. **Parallel Evolution Client** (`parallel_evolution.py`): Manages batch dispatch and result collection
3. **Modified Evolution** (`evo_v5.py`): New `evolve_parallel()` method for generation-based evolution
4. **Main Script** (`main_parallel.py`): Entry point for parallel evolution

## Setup Instructions

### 1. Install RunPod CLI

```bash
pip install runpod
```

### 2. Set Up RunPod Account

1. Create account at https://www.runpod.io
2. Get your API key from the dashboard
3. Set environment variables:

```bash
export RUNPOD_API_KEY="your-api-key-here"
export RUNPOD_ENDPOINT_ID="your-endpoint-id-here"  # You'll get this after creating endpoint
```

### 3. Build and Deploy Handler

```bash
cd workflow/recurrent

# Build Docker image
runpod build

# Deploy to RunPod
runpod deploy
```

This will:
- Build a Docker image with your code
- Deploy it as a serverless endpoint
- Return an endpoint ID (save this!)

### 4. Configure Endpoint

After deployment, configure your endpoint:
- **GPU Type**: Choose based on your needs (RTX 3090, A100, etc.)
- **Max Workers**: Set to your desired parallelism (50-200 recommended)
- **Idle Timeout**: 60 seconds (to minimize costs)
- **Volume Mounts**: Mount your dataset if needed

### 5. Run Parallel Evolution

```bash
# Set environment variables
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

# Run parallel evolution
python main_parallel.py
```

## Configuration

### Batch Size

The `batch_size` parameter controls how many solutions are generated and trained in parallel per generation:

- **Small (20)**: Lower cost, ~120 solutions in 30 minutes
- **Medium (50)**: Balanced, ~300 solutions in 30 minutes  
- **Large (100-200)**: Maximum throughput, 600-1200 solutions in 30 minutes

### Time Limits

- **30 minutes**: `time_limit=1800` (for testing)
- **8 hours**: `time_limit=28800` (for full runs)

### Cost Estimation

RunPod charges per GPU-second. Example:
- RTX 3090: ~$0.0003/second
- 50 workers × 4.5 minutes = 225 GPU-minutes = 13,500 GPU-seconds
- Cost per generation: ~$4.05
- 6 generations (30 min): ~$24.30
- 100 generations (8 hours): ~$405

## Performance Targets

| Configuration | Solutions/30min | Solutions/8hr | Cost/8hr |
|--------------|----------------|---------------|----------|
| 20 workers   | ~130           | ~2,080        | ~$162    |
| 50 workers   | ~310           | ~4,960        | ~$405    |
| 100 workers  | ~610           | ~9,760        | ~$810    |
| 200 workers  | ~1,210         | ~19,360       | ~$1,620  |

## Troubleshooting

### Jobs Failing

1. Check RunPod logs in dashboard
2. Verify dataset path is correct
3. Ensure Docker image has all dependencies

### Slow Performance

1. Increase `max_workers` on endpoint
2. Use faster GPU types (A100 > RTX 3090)
3. Reduce `batch_size` if hitting rate limits

### High Costs

1. Reduce `max_workers`
2. Use cheaper GPU types
3. Set shorter `idle_timeout`
4. Monitor usage in RunPod dashboard

## Monitoring

- **RunPod Dashboard**: Real-time job status and logs
- **Console Output**: Generation progress and statistics
- **Solution Archive**: `outputs/recurrent/all_solutions.jsonl`

## Comparison: Sequential vs Parallel

| Metric | Sequential | Parallel (50 workers) |
|--------|-----------|----------------------|
| Solutions/8hr | ~117 | ~4,960 |
| Speedup | 1x | **42x** |
| Cost | $0 (local) | ~$405 |

## Next Steps

1. Start with small batch size (20) to test
2. Monitor costs and performance
3. Scale up based on results
4. Optimize GPU selection for cost/performance

