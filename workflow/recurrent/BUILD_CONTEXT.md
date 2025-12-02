# Build Context Explanation

## What is Build Context?

The **Build Context** is the directory that Docker uses as the "root" when building your image. All `COPY` commands in your Dockerfile are relative to this directory.

## For Your Project

Your build context should be:

```
/Users/anandafrancis/ananda/c-resources/d-leverage/credentials/accreditation/degrees/northeastern/fa25/thesis/evolutionary-ml/workflow/recurrent
```

Or relative to your project root:

```
workflow/recurrent/
```

## Why This Matters

Your Dockerfile contains:
```dockerfile
COPY requirements.txt .
COPY . .
```

These commands copy files **from the build context** into the Docker image. So:
- `COPY requirements.txt .` copies `workflow/recurrent/requirements.txt`
- `COPY . .` copies everything in `workflow/recurrent/` into `/app` in the container

## Setting Build Context in RunPod

When deploying via RunPod CLI or web interface:

### Option 1: RunPod CLI (from the build context directory)
```bash
cd workflow/recurrent
runpod build
```

The CLI will automatically use the current directory as the build context.

### Option 2: RunPod Web Interface
If using the web interface, specify:
- **Build Context**: `/full/path/to/workflow/recurrent`
- **Dockerfile Path**: `./Dockerfile` (relative to build context)

### Option 3: RunPod Config File
Some RunPod setups allow specifying in config. The build context is typically:
- The directory where you run `runpod build`
- Or specified via `--context` flag: `runpod build --context ./workflow/recurrent`

## Quick Check

To verify your build context is correct:

1. **From the build context directory**, these files should exist:
   - `Dockerfile` ✓
   - `requirements.txt` ✓
   - `runpod_handler.py` ✓
   - `solution.py` ✓
   - All other Python files ✓

2. **Test locally** (optional):
   ```bash
   cd workflow/recurrent
   docker build -t test-image .
   ```

## Common Issues

❌ **Wrong**: Build context = project root
- Dockerfile can't find `requirements.txt` (it's in a subdirectory)
- `COPY . .` copies too much (entire project instead of just `workflow/recurrent/`)

✅ **Correct**: Build context = `workflow/recurrent/`
- All files are in the right place
- Dockerfile commands work as expected

## Summary

**Your build context is**: `workflow/recurrent/`

When deploying, make sure RunPod knows to use this directory as the build context.

