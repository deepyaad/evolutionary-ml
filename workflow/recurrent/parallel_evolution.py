"""
Parallel evolution support using RunPod serverless.
This module provides functions to dispatch training jobs to RunPod and collect results.
"""
import runpod
import time
import json
import numpy as np
from solution import Solution
from typing import List, Dict, Any, Optional
import os


class RunPodEvolutionClient:
    """
    Client for managing parallel evolution via RunPod serverless.
    """
    
    def __init__(self, endpoint_id: str, api_key: Optional[str] = None):
        """
        Initialize RunPod client.
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key (if None, uses environment variable)
        """
        self.endpoint_id = endpoint_id
        
        # Set API key
        if api_key:
            os.environ['RUNPOD_API_KEY'] = api_key
        elif not os.getenv('RUNPOD_API_KEY'):
            raise ValueError("RunPod API key must be provided or set in RUNPOD_API_KEY environment variable")
        
        # Initialize RunPod client
        self.client = runpod.Client()
    
    def prepare_solution_for_dispatch(self, sol: Solution, data_path: str = '../../datasets/spotify_dataset.npz') -> Dict[str, Any]:
        """
        Convert a Solution object to a dictionary suitable for RunPod dispatch.
        
        Args:
            sol: Solution object
            data_path: Path to dataset file (will be used on RunPod)
        
        Returns:
            Dictionary with solution configuration and data path
        """
        # Convert solution to dict (excludes model and hidden_layers)
        sol_dict = sol.to_dict()
        
        # Prepare data - we'll send a small sample or just the path
        # For efficiency, we can either:
        # 1. Send data path (requires volume mount)
        # 2. Send serialized data (larger payload)
        # We'll use option 1 for now (data_path)
        
        return {
            'solution_config': sol_dict['configuration'],
            'data_path': data_path
        }
    
    def dispatch_training_jobs(self, solutions: List[Solution], data_path: str = '../../datasets/spotify_dataset.npz') -> List[str]:
        """
        Dispatch multiple training jobs to RunPod in parallel.
        
        Args:
            solutions: List of Solution objects to train
            data_path: Path to dataset file
        
        Returns:
            List of job IDs
        """
        job_ids = []
        
        for sol in solutions:
            # Prepare input
            input_data = self.prepare_solution_for_dispatch(sol, data_path)
            
            # Dispatch job using RunPod API
            try:
                job = self.client.run(self.endpoint_id, input_data)
                job_id = job.get('id') if isinstance(job, dict) else str(job)
                job_ids.append(job_id)
                print(f"Dispatched job {job_id} for solution {sol.configuration['id']}")
            except Exception as e:
                print(f"Error dispatching job for solution {sol.configuration['id']}: {e}")
                import traceback
                traceback.print_exc()
                job_ids.append(None)
        
        return job_ids
    
    def wait_for_jobs(self, job_ids: List[str], timeout: int = 3600, poll_interval: int = 5) -> List[Dict[str, Any]]:
        """
        Wait for all jobs to complete and collect results.
        
        Args:
            job_ids: List of job IDs
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)
        
        Returns:
            List of result dictionaries (one per job)
        """
        results = []
        start_time = time.time()
        
        # Filter out None job_ids (failed dispatches)
        valid_job_ids = [jid for jid in job_ids if jid is not None]
        
        print(f"Waiting for {len(valid_job_ids)} jobs to complete...")
        
        while valid_job_ids and (time.time() - start_time) < timeout:
            completed = []
            
            for job_id in valid_job_ids:
                try:
                    # Get job status
                    status_response = self.client.get_job_status(self.endpoint_id, job_id)
                    status = status_response.get('status', 'UNKNOWN')
                    
                    if status == 'COMPLETED':
                        # Get the result
                        result_response = self.client.get_job_output(self.endpoint_id, job_id)
                        result = result_response.get('output', {})
                        results.append(result)
                        completed.append(job_id)
                        print(f"Job {job_id} completed successfully")
                    
                    elif status == 'FAILED':
                        error_msg = status_response.get('error', 'Unknown error')
                        print(f"Job {job_id} failed: {error_msg}")
                        results.append({
                            'success': False,
                            'error': error_msg,
                            'id': None
                        })
                        completed.append(job_id)
                
                except Exception as e:
                    print(f"Error checking status for job {job_id}: {e}")
            
            # Remove completed jobs
            valid_job_ids = [jid for jid in valid_job_ids if jid not in completed]
            
            if valid_job_ids:
                print(f"Still waiting for {len(valid_job_ids)} jobs... ({len(results)}/{len(job_ids)} completed)")
                time.sleep(poll_interval)
        
        if valid_job_ids:
            print(f"Warning: {len(valid_job_ids)} jobs did not complete within timeout")
            # Add None results for incomplete jobs
            for job_id in valid_job_ids:
                results.append({
                    'success': False,
                    'error': 'Timeout',
                    'id': None
                })
        
        return results
    
    def train_solutions_parallel(self, solutions: List[Solution], data_path: str = '../../datasets/spotify_dataset.npz', 
                                  timeout: int = 3600) -> List[Dict[str, Any]]:
        """
        Dispatch solutions for training and wait for results.
        
        Args:
            solutions: List of Solution objects to train
            data_path: Path to dataset file
            timeout: Maximum time to wait for completion (seconds)
        
        Returns:
            List of result dictionaries
        """
        # Dispatch all jobs
        job_ids = self.dispatch_training_jobs(solutions, data_path)
        
        # Wait for completion
        results = self.wait_for_jobs(job_ids, timeout)
        
        return results


def convert_result_to_solution(result: Dict[str, Any]) -> Optional[Solution]:
    """
    Convert a RunPod result dictionary back to a Solution object.
    
    Args:
        result: Result dictionary from RunPod
    
    Returns:
        Solution object or None if conversion failed
    """
    if not result.get('success', False):
        print(f"Result indicates failure: {result.get('error', 'Unknown error')}")
        return None
    
    try:
        # Extract configuration and metrics
        config = result.get('configuration', {})
        metrics = result.get('metrics', {})
        
        # Create solution
        sol = Solution(config, metrics=metrics)
        return sol
    
    except Exception as e:
        print(f"Error converting result to solution: {e}")
        return None

