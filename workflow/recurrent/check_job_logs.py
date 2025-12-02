#!/usr/bin/env python3
"""
Check the logs from a failed job to see what's actually happening.
"""
import os
import runpod
import sys

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
    
    # Check endpoint health
    try:
        health = endpoint.health()
        print("\nEndpoint Health:")
        print("-" * 60)
        print(f"Workers: {health}")
        
        # Try to get job status from one of the queued jobs
        # You'll need to get a job ID from the dashboard
        print("\n" + "="*60)
        print("To see job logs:")
        print("1. Go to RunPod dashboard: https://www.runpod.io/console/serverless")
        print(f"2. Click on endpoint: {endpoint_id}")
        print("3. Go to 'Jobs' tab")
        print("4. Click on any job (especially a failed one)")
        print("5. Check the 'Logs' tab to see error messages")
        print("\nCommon issues to look for:")
        print("  - Import errors (can't find solution.py, layer_builder.py, etc.)")
        print("  - Dataset not found errors")
        print("  - Handler startup errors")
        print("  - Python syntax errors")
        
    except Exception as e:
        print(f"Error checking endpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

