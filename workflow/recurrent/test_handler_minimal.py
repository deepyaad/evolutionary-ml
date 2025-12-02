#!/usr/bin/env python3
"""
Minimal test to see if the handler can even start.
This tests the handler code locally before deploying.
"""
import sys
import os
from pathlib import Path

# Add /app to path (simulating Docker environment)
sys.path.insert(0, str(Path(__file__).parent))

print("Testing handler imports...")
try:
    from solution import Solution
    print("✅ solution.py imported")
except Exception as e:
    print(f"❌ Failed to import solution: {e}")
    sys.exit(1)

try:
    from layer_builder import rebuild_layers_from_config
    print("✅ layer_builder.py imported")
except Exception as e:
    print(f"❌ Failed to import layer_builder: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print(f"❌ Failed to import numpy: {e}")
    sys.exit(1)

try:
    import runpod
    print("✅ runpod imported")
except Exception as e:
    print(f"❌ Failed to import runpod: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All imports successful!")
print("="*60)
print("\nThe handler should be able to start.")
print("If jobs are still timing out, check:")
print("1. Job logs in RunPod dashboard")
print("2. Dataset path is correct")
print("3. All files are in the Docker image")

