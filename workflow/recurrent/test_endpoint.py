#!/usr/bin/env python3
"""
Quick test script to verify the RunPod endpoint is working.
Tests with a single simple job first.
"""
import os
import runpod
import json

# Get endpoint ID
endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID', 's5boifmseronhh')
api_key = os.getenv('RUNPOD_API_KEY')

if not api_key:
    # Try to read from file
    api_key_paths = ['../../api-key.txt', 'api-key.txt']
    for path in api_key_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                api_key = f.read().strip()
            break

if not api_key:
    print("ERROR: Need RUNPOD_API_KEY")
    exit(1)

runpod.api_key = api_key

# Create endpoint
endpoint = runpod.Endpoint(endpoint_id)

# Test with a minimal solution config
# We'll use a very simple config to test if the handler works
test_input = {
    "solution_config": {
        "id": "test-123",
        "hidden_layer_count": 1,
        "layer_names": ["SimpleRNN"],
        "layer_specifications": [{"return_sequences": False, "units": 10}],
        "neurons_per_layer": [10],
        "feature_shape": [431, 187],
        "class_count": 6,
        "epochs": 3,  # Very short for testing
        "batch_size": 32,
        "loss_function": "categorical_crossentropy",
        "optimizer": "adam",
        "input_size": [431, 187],
        "output_size": 6,
        "labels_inorder": ["english", "hindi", "mandarin", "patois", "pidgin", "spanish"]
    },
    "data_path": "/workspace/dataset.npz"
}

print(f"Testing endpoint {endpoint_id}...")
print("Sending test job...")

try:
    # Use run_sync for easier debugging
    result = endpoint.run_sync(test_input, timeout=600)  # 10 minute timeout
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))
    print("="*60)
    
    if isinstance(result, dict) and result.get('success'):
        print("✅ Test job completed successfully!")
    else:
        print("❌ Test job failed or returned unexpected result")
        if isinstance(result, dict) and 'error' in result:
            print(f"Error: {result['error']}")
            
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

