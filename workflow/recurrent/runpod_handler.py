"""
RunPod serverless handler for training solutions in parallel.
This handler receives a solution configuration, trains it, and returns metrics.
"""
import runpod
import numpy as np
import json
import os
from solution import Solution

def handler(event):
    """
    RunPod serverless handler function.
    
    Args:
        event: Dictionary containing:
            - 'input': Dictionary with:
                - 'solution_config': Solution configuration dictionary
                - 'data_path': Path to dataset file (or None if data is in event)
                - 'data': Optional - dataset dictionary (if not provided, loads from data_path)
    
    Returns:
        Dictionary with:
            - 'id': Solution ID
            - 'metrics': Training metrics
            - 'success': Boolean indicating if training succeeded
            - 'error': Error message if training failed
    """
    try:
        input_data = event.get('input', {})
        solution_config = input_data.get('solution_config')
        data_path = input_data.get('data_path', '../../datasets/spotify_dataset.npz')
        data_dict = input_data.get('data')
        
        # Load dataset if not provided
        if data_dict is None:
            if not os.path.exists(data_path):
                # Try relative to handler location
                data_path = os.path.join(os.path.dirname(__file__), data_path)
            data = np.load(data_path, allow_pickle=True)
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
        
        # Create solution from configuration
        sol = Solution(solution_config)
        
        # Train the model
        sol.develop_model(data)
        
        # Return results
        return {
            'id': str(sol.configuration['id']),
            'configuration': sol.to_dict()['configuration'],
            'metrics': sol.to_dict()['metrics'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        return {
            'id': solution_config.get('id', 'unknown') if 'solution_config' in locals() else 'unknown',
            'configuration': solution_config if 'solution_config' in locals() else None,
            'metrics': None,
            'success': False,
            'error': error_msg,
            'traceback': traceback_str
        }

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

