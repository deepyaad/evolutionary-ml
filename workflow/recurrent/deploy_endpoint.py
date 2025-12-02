#!/usr/bin/env python3
"""
Script to deploy RunPod serverless endpoint programmatically.
This is easier than navigating the web UI.
"""
import os
import sys
import runpod

def main():
    # Get API key from environment or file
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        # Try to read from api-key.txt
        api_key_paths = [
            '../../api-key.txt',
            'api-key.txt',
            os.path.expanduser('~/.runpod/api_key.txt')
        ]
        for path in api_key_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    api_key = f.read().strip()
                print(f"Loaded API key from {path}")
                break
    
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not found!")
        print("Set it as an environment variable or create api-key.txt file")
        print("\nExample:")
        print("  export RUNPOD_API_KEY='your-key-here'")
        print("  OR")
        print("  echo 'your-key-here' > api-key.txt")
        sys.exit(1)
    
    # Set API key
    runpod.api_key = api_key
    
    # Image name from your build
    image_name = "registry.runpod.net/deepyaad-evolutionary-ml-rnn-cloud-workflow-recurrent-dockerfile:5b9d28f40"
    
    print(f"\nDeploying endpoint with image: {image_name}")
    print("This may take a few minutes...\n")
    
    try:
        # Step 1: Create or reuse template
        print("Step 1: Creating template...")
        
        # Option 1: Use existing template ID if you have one (from previous run)
        # Uncomment and set this if you want to reuse an existing template:
        # template_id = "sx8an13dpd"  # Your existing template ID from previous run
        
        # Option 2: Create new template with unique name (to avoid conflicts)
        import time
        template_name = f"evolutionary-ml-handler-template-{int(time.time())}"
        print(f"Creating template with unique name: {template_name}")
        
        template = runpod.create_template(
            name=template_name,
            image_name=image_name,
            docker_start_cmd="python /app/runpod_handler.py",
            container_disk_in_gb=10,
            env={
                "PYTHONUNBUFFERED": "1"
            },
            is_serverless=True  # This is a serverless endpoint
        )
        
        template_id = template.get('id') or template.get('template_id')
        if not template_id:
            raise ValueError(f"Failed to create template. Response: {template}")
        
        print(f"✅ Template created: {template_id}\n")
        
        # Step 2: Create the endpoint from the template
        print("Step 2: Creating endpoint from template...")
        # Check if user has a workers quota limit
        # Default to 2 workers for accounts with limited quota (can be increased later)
        # You can increase this if you have more quota available
        workers_max = 2  # Start with 2 workers (fits within 5 worker quota)
        
        # Allow override via environment variable
        if os.getenv('RUNPOD_MAX_WORKERS'):
            workers_max = int(os.getenv('RUNPOD_MAX_WORKERS'))
        
        print(f"Creating endpoint with {workers_max} max workers (adjust via RUNPOD_MAX_WORKERS env var)")
        
        endpoint = runpod.create_endpoint(
            name="evolutionary-ml-handler",
            template_id=template_id,
            gpu_ids="AMPERE_16",  # RTX 3090/4090 - adjust if needed
            # Alternative GPU options: "AMPERE_16", "AMPERE_24", "AMPERE_40", "AMPERE_48", "AMPERE_80"
            workers_min=0,  # Minimum workers (0 = scale to zero)
            workers_max=workers_max,  # Maximum parallel workers (limited by your quota)
            idle_timeout=60,  # Seconds before shutting down idle workers
            scaler_type="QUEUE_DELAY",  # Scale based on queue delay
            scaler_value=4,  # Scale when queue delay > 4 seconds
            gpu_count=1
        )
        
        endpoint_id = endpoint.get('id') or endpoint.get('endpoint_id')
        if not endpoint_id:
            raise ValueError(f"Failed to create endpoint. Response: {endpoint}")
        
        print("\n" + "="*60)
        print("✅ ENDPOINT DEPLOYED SUCCESSFULLY!")
        print("="*60)
        print(f"\nTemplate ID: {template_id}")
        print(f"Endpoint ID: {endpoint_id}")
        print(f"Max Workers: {workers_max} (limited by your RunPod quota)")
        print(f"\n⚠️  NOTE: You have a quota of 5 workers total.")
        print(f"   This endpoint uses {workers_max} workers.")
        print(f"   To increase workers, upgrade your RunPod plan or reduce other endpoints.")
        print(f"\nSet these environment variables:")
        print(f"  export RUNPOD_ENDPOINT_ID='{endpoint_id}'")
        print(f"  export RUNPOD_TEMPLATE_ID='{template_id}'")
        print(f"\nOr add them to your .bashrc/.zshrc:")
        print(f"  echo \"export RUNPOD_ENDPOINT_ID='{endpoint_id}'\" >> ~/.zshrc")
        print(f"  echo \"export RUNPOD_TEMPLATE_ID='{template_id}'\" >> ~/.zshrc")
        print("\n" + "="*60)
        
        # Save to a file for convenience
        with open('endpoint_id.txt', 'w') as f:
            f.write(f"ENDPOINT_ID={endpoint_id}\n")
            f.write(f"TEMPLATE_ID={template_id}\n")
        print("\nEndpoint and Template IDs also saved to: endpoint_id.txt")
        
    except Exception as e:
        print(f"\n❌ ERROR deploying endpoint: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Verify you have credits in your RunPod account")
        print("3. Check if the GPU type is available")
        print("4. Try a different GPU type (e.g., 'RTX 4090' or 'A40')")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

