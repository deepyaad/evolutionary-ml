#!/usr/bin/env python3
"""
Check the status and logs of recent RunPod jobs.
"""
import os
import runpod
import sys
import time

def main():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        api_key_paths = ['../../api-key.txt', 'api-key.txt']
        for path in api_key_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    api_key = f.read().strip()
                break
    
    if not api_key:
        print("ERROR: Need RUNPOD_API_KEY")
        sys.exit(1)
    
    runpod.api_key = api_key
    
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID', 's5boifmseronhh')
    endpoint = runpod.Endpoint(endpoint_id)
    
    print(f"Checking endpoint: {endpoint_id}")
    print("="*60)
    
    # Try to get endpoint info
    try:
        # Get recent jobs - we'll need to use the RunPod API directly
        print("\nTo check job logs:")
        print("1. Go to: https://www.runpod.io/console/serverless")
        print(f"2. Click on endpoint: {endpoint_id}")
        print("3. Go to 'Jobs' tab")
        print("4. Click on any job (especially failed/timed out ones)")
        print("5. Check the 'Logs' tab")
        print("\n" + "="*60)
        print("\nCommon issues to look for in logs:")
        print("  - Import errors (can't find solution.py, layer_builder.py)")
        print("  - Dataset not found errors")
        print("  - Python syntax errors")
        print("  - Handler startup errors")
        print("  - Missing dependencies")
        print("\n" + "="*60)
        print("\nIf you see errors, we need to:")
        print("1. Fix the handler code")
        print("2. Rebuild the Docker image: runpod build")
        print("3. Redeploy the endpoint with the new image")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

