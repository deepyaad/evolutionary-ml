"""
RunPod serverless handler for training solutions in parallel.
This handler receives a solution configuration, trains it, and returns metrics.
"""
import runpod
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add /app to Python path so we can import our modules
sys.path.insert(0, '/app')

from solution import Solution
from layer_builder import rebuild_layers_from_config

# Dataset path in the Docker container
DATASET_PATH = "/workspace/dataset.npz"

def handler(event):
    """
    RunPod serverless handler function.
    
    Args:
        event: Dictionary containing:
            - 'input': Dictionary with:
                - 'solution_config': Solution configuration dictionary
                - 'data_path': Optional - path to dataset (defaults to /workspace/dataset.npz)
                - 'data': Optional - dataset dictionary (if not provided, loads from data_path)
    
    Returns:
        Dictionary with:
            - 'id': Solution ID
            - 'metrics': Training metrics
            - 'success': Boolean indicating if training succeeded
            - 'error': Error message if training failed
    """
    solution_config = None
    try:
        input_data = event.get('input', {})
        solution_config = input_data.get('solution_config')
        
        if solution_config is None:
            raise ValueError("solution_config is required in event['input']")
        
        # Load dataset
        data_path = input_data.get('data_path', DATASET_PATH)
        data_dict = input_data.get('data')
        
        if data_dict is None:
            # Load from file - try multiple possible paths
            possible_paths = [
                data_path,  # User-specified path
                DATASET_PATH,  # /workspace/dataset.npz
                "/workspace/datasets/spotify_dataset.npz",  # If volume mounted at /workspace/datasets
                "/datasets/spotify_dataset.npz",  # If volume mounted at /datasets
                "/app/datasets/spotify_dataset.npz",  # Alternative location
            ]
            
            data = None
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Loading dataset from {path}...")
                    data = np.load(path, allow_pickle=True)
                    print(f"Dataset loaded: train={data['train_features'].shape}, test={data['test_features'].shape}")
                    break
            
            if data is None:
                raise FileNotFoundError(
                    f"Dataset not found at any of these paths: {possible_paths}. "
                    "Make sure dataset is mounted as a volume or copied to the image."
                )
        else:
            # Convert data_dict back to numpy arrays
            data = {
                'train_features': np.array(data_dict['train_features']),
                'train_labels': np.array(data_dict['train_labels']),
                'val_features': np.array(data_dict['val_features']),
                'val_labels': np.array(data_dict['val_labels']),
                'test_features': np.array(data_dict['test_features']),
                'test_labels': np.array(data_dict['test_labels']),
                'labels_inorder': np.array(data_dict['labels_inorder'])
            }
        
        # Rebuild hidden_layers from configuration
        # (hidden_layers are Keras objects that can't be serialized, so we rebuild them)
        if 'hidden_layers' not in solution_config or solution_config.get('hidden_layers') is None:
            print("Rebuilding hidden_layers from layer_names and layer_specifications...")
            hidden_layers = rebuild_layers_from_config(solution_config)
            solution_config['hidden_layers'] = hidden_layers
            print(f"Rebuilt {len(hidden_layers)} layers")
        
        # Create solution from configuration
        print(f"Creating solution for ID: {solution_config.get('id', 'unknown')}")
        sol = Solution(solution_config)
        
        # Train the model
        print("Starting model training...")
        sol.develop_model(data)
        print("Model training completed")
        
        # Return results
        result = {
            'id': str(sol.configuration['id']),
            'configuration': sol.to_dict()['configuration'],
            'metrics': sol.to_dict()['metrics'],
            'success': True,
            'error': None
        }
        print(f"Returning results for solution {result['id']}")
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"ERROR in handler: {error_msg}")
        print(f"Traceback:\n{traceback_str}")
        return {
            'id': str(solution_config.get('id', 'unknown')) if solution_config else 'unknown',
            'configuration': solution_config if solution_config else None,
            'metrics': None,
            'success': False,
            'error': error_msg,
            'traceback': traceback_str
        }

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

