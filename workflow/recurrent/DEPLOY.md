# How to Deploy RunPod Serverless Endpoint

## Method 1: Using RunPod CLI (Recommended)

### Step 1: Configure API Key

```bash
# Set your RunPod API key
runpod config YOUR_API_KEY_HERE
```

Or set it as an environment variable:
```bash
export RUNPOD_API_KEY="your-api-key-here"
```

### Step 2: Deploy the Endpoint

Since your Docker image was already built and pushed (from the build logs), you have two options:

#### Option A: Deploy via CLI (if you have a runpod.toml project file)
```bash
cd workflow/recurrent
runpod project deploy
```

#### Option B: Deploy via RunPod Web Dashboard (Recommended)

1. Go to https://www.runpod.io/console/serverless
2. Click **"Create Endpoint"** or **"New Endpoint"** button
3. Look for the image configuration field. It might be labeled as:
   - **"Container Image"** (most common)
   - **"Docker Image"**
   - **"Image Name"** or **"Image"**
   - **"Custom Image"** (if there's a template selector)
   - **"Image URL"** or **"Image Path"**

4. **Where to find it:**
   - If you see a dropdown with templates, look for **"Custom"** or **"Use Custom Image"** option
   - If you see a text input field, paste your image name there
   - The field might be in a section called **"Container Settings"** or **"Image Configuration"**

5. **Enter your image name:**
   ```
   registry.runpod.net/deepyaad-evolutionary-ml-rnn-cloud-workflow-recurrent-dockerfile:5b9d28f40
   ```
   
   **OR** if there's a dropdown to browse images:
   - Click **"Browse Images"** or **"Select Image"**
   - Look for: `deepyaad-evolutionary-ml-rnn-cloud-workflow-recurrent-dockerfile`
   - Select the tag: `5b9d28f40` (or latest)

6. **Configure other settings:**
   - **Name**: `evolutionary-ml-handler` (or any name you prefer)
   - **GPU Type**: Choose based on your needs:
     - RTX 3090 (cheaper, good performance)
     - A100 (faster, more expensive)
   - **Max Workers**: Set to your desired parallelism:
     - Start with **20-50** for testing
     - Scale up to **100-200** for production
   - **Idle Timeout**: `60` seconds (to minimize costs)
   - **Container Disk**: `10 GB` (should be enough)
   - **Volume Mounts**: 
     - Mount your dataset if needed (optional, since dataset is in the image)
   - **Environment Variables**: 
     - `PYTHONUNBUFFERED=1` (already in Dockerfile)

7. Click **"Create"** or **"Deploy"**

8. **Save the Endpoint ID** - You'll need this for `main_parallel.py`

**If you still can't find the image field**, try:
- Look for a **"Templates"** tab and switch to **"Custom"**
- Check if there's an **"Advanced"** or **"Show Advanced Options"** toggle
- The interface might have changed - use Method 2 (Python SDK) below instead

### Step 3: Get Your Endpoint ID

After deployment, you'll see an endpoint ID like: `abc123def456...`

Copy this ID and set it as an environment variable:

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id-here"
```

Or add it to your `main_parallel.py` or create a config file.

## Method 2: Using RunPod Python SDK (Alternative)

If the CLI doesn't work, you can deploy programmatically:

```python
import runpod

runpod.api_key = "your-api-key"

endpoint = runpod.create_endpoint(
    name="evolutionary-ml-handler",
    image_name="registry.runpod.net/deepyaad-evolutionary-ml-rnn-cloud-workflow-recurrent-dockerfile:5b9d28f40",
    gpu_type_id="RTX 3090",  # or "A100" etc.
    max_workers=50,
    idle_timeout=60,
    container_disk=10,
)

print(f"Endpoint ID: {endpoint['id']}")
```

## Verify Deployment

1. Go to RunPod dashboard → Serverless → Your Endpoint
2. Click **"Test"** to send a test job
3. Check logs to ensure it's working

## Next Steps

Once deployed, update your environment variables and run:

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

cd workflow/recurrent
python main_parallel.py
```

## Troubleshooting

- **Image not found**: Make sure the image was successfully pushed (check build logs)
- **Endpoint creation fails**: Check your API key and account credits
- **Jobs not starting**: Verify GPU type availability and max workers setting
- **Import errors**: Check that all dependencies are in the Docker image

