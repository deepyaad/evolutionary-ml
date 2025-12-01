# Setting Up RunPod API Key

You have **three options** for providing your RunPod API key:

## Option 1: Environment Variable (Recommended)

Set it in your shell before running:

```bash
export RUNPOD_API_KEY="your-api-key-here"
export RUNPOD_ENDPOINT_ID="your-endpoint-id-here"
python main_parallel.py
```

Or add to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export RUNPOD_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export RUNPOD_ENDPOINT_ID="your-endpoint-id-here"' >> ~/.bashrc
source ~/.bashrc
```

## Option 2: api-key.txt File (Already Set Up)

The code will automatically read from `api-key.txt` in the project root if the environment variable is not set.

**Your API key is already in `api-key.txt`** (this file is in `.gitignore` so it won't be committed).

You just need to:
1. Make sure `api-key.txt` is in the project root directory
2. The code will automatically use it

## Option 3: Pass Directly in Code (Not Recommended)

You can modify `main_parallel.py` to hardcode it, but this is **not recommended** for security reasons.

## Getting Your API Key

1. Go to https://www.runpod.io
2. Sign in to your account
3. Navigate to **Settings** → **API Keys**
4. Copy your API key

## Getting Your Endpoint ID

1. Deploy your handler: `runpod build && runpod deploy`
2. The endpoint ID will be shown after deployment
3. Or find it in the RunPod dashboard under **Serverless** → **Endpoints**

## Verification

To verify your API key is loaded correctly, you can add this to `main_parallel.py` temporarily:

```python
print(f"Using endpoint: {runpod_endpoint_id}")
print(f"API key loaded: {'Yes' if runpod_api_key else 'No'}")
```

## Security Notes

- ✅ `api-key.txt` is already in `.gitignore` - it won't be committed
- ✅ Never commit API keys to git
- ✅ Use environment variables in production
- ✅ Rotate keys if accidentally exposed

