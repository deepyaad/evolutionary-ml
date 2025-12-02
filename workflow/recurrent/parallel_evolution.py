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
            runpod.api_key = api_key
        elif not os.getenv('RUNPOD_API_KEY') and not runpod.api_key:
            raise ValueError("RunPod API key must be provided or set in RUNPOD_API_KEY environment variable")
        
        # Initialize RunPod Endpoint
        self.endpoint = runpod.Endpoint(endpoint_id)
    
    def prepare_solution_for_dispatch(self, sol: Solution, data_path: str = '/workspace/dataset.npz') -> Dict[str, Any]:
        """
        Convert a Solution object to a dictionary suitable for RunPod dispatch.
        
        Args:
            sol: Solution object
            data_path: Path to dataset file in RunPod container (default: /workspace/dataset.npz)
        
        Returns:
            Dictionary with solution configuration and data path
        """
        # Convert solution to dict (excludes model and hidden_layers)
        sol_dict = sol.to_dict()
        
        # The dataset is embedded in the Docker image at /workspace/dataset.npz
        # So we just need to pass the path, not the data itself
        
        return {
            'solution_config': sol_dict['configuration'],
            'data_path': data_path  # This is the path in the RunPod container
        }
    
    def dispatch_training_jobs(self, solutions: List[Solution], data_path: str = '/workspace/dataset.npz', use_sync: bool = False) -> List[Any]:
        """
        Dispatch multiple training jobs to RunPod in parallel.
        
        Args:
            solutions: List of Solution objects to train
            data_path: Path to dataset file
            use_sync: If True, use run_sync() for synchronous execution (waits for completion)
        
        Returns:
            List of Job objects or results (depending on use_sync)
        """
        jobs = []
        
        for sol in solutions:
            # Prepare input
            input_data = self.prepare_solution_for_dispatch(sol, data_path)
            
            # Dispatch job using RunPod API
            try:
                if use_sync:
                    # Use synchronous execution - waits for completion
                    print(f"Running job synchronously for solution {sol.configuration['id']}...")
                    result = self.endpoint.run_sync(input_data, timeout=3600)
                    jobs.append(result)
                    print(f"✅ Job completed for solution {sol.configuration['id']}")
                else:
                    # Use asynchronous execution - returns Job object
                    job = self.endpoint.run(input_data)
                    jobs.append(job)
                    job_id = job.id if hasattr(job, 'id') else str(job)
                    print(f"Dispatched job {job_id} for solution {sol.configuration['id']}")
            except Exception as e:
                print(f"❌ Error dispatching job for solution {sol.configuration['id']}: {e}")
                import traceback
                traceback.print_exc()
                jobs.append(None)
        
        return jobs
    
    def wait_for_jobs(self, jobs: List[Any], timeout: int = 3600, poll_interval: int = 5) -> List[Dict[str, Any]]:
        """
        Wait for all jobs to complete and collect results.
        
        Args:
            jobs: List of Job objects (from endpoint.run())
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)
        
        Returns:
            List of result dictionaries (one per job)
        """
        results = []
        start_time = time.time()
        
        # Filter out None jobs (failed dispatches)
        valid_jobs = [job for job in jobs if job is not None]
        
        print(f"Waiting for {len(valid_jobs)} jobs to complete...")
        
        while valid_jobs and (time.time() - start_time) < timeout:
            completed = []
            
            for job in valid_jobs:
                try:
                    # Get job status using the Job object's status() method
                    status = job.status()
                    job_id = job.id if hasattr(job, 'id') else str(job)
                    
                    # Debug: print status every 10th check to see what's happening
                    if len(results) == 0 or len(valid_jobs) <= 2:
                        print(f"  Job {job_id[:8]}... status: {status}")
                    
                    # Check for completion (status might be a dict or string)
                    if isinstance(status, dict):
                        status_str = status.get('status', '').upper()
                    else:
                        status_str = str(status).upper()
                    
                    if 'COMPLETED' in status_str or status_str == 'COMPLETED':
                        # Get the result using the Job object's output() method
                        try:
                            result = job.output()
                            results.append(result)
                            completed.append(job)
                            print(f"✅ Job {job_id} completed successfully")
                        except Exception as e:
                            print(f"⚠️  Job {job_id} marked completed but error getting output: {e}")
                            results.append({
                                'success': False,
                                'error': f'Error getting output: {e}',
                                'id': None
                            })
                            completed.append(job)
                    
                    elif 'FAILED' in status_str or status_str == 'FAILED' or 'ERROR' in status_str:
                        # Try to get error message
                        try:
                            output = job.output()
                            error_msg = output.get('error', 'Unknown error') if isinstance(output, dict) else str(output)
                        except Exception as e:
                            error_msg = f'Job failed: {e}'
                        
                        results.append({
                            'success': False,
                            'error': error_msg,
                            'id': None
                        })
                        completed.append(job)
                        print(f"❌ Job {job_id} failed: {error_msg}")
                    
                    # Status is IN_QUEUE, IN_PROGRESS, etc. - keep waiting
                
                except Exception as e:
                    job_id = job.id if hasattr(job, 'id') else str(job)
                    print(f"⚠️  Error checking status for job {job_id}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Remove completed jobs
            valid_jobs = [job for job in valid_jobs if job not in completed]
            
            if valid_jobs:
                print(f"Still waiting for {len(valid_jobs)} jobs... ({len(results)}/{len(jobs)} completed)")
                time.sleep(poll_interval)
        
        if valid_jobs:
            print(f"Warning: {len(valid_jobs)} jobs did not complete within timeout")
            # Add None results for incomplete jobs
            for job in valid_jobs:
                results.append({
                    'success': False,
                    'error': 'Timeout',
                    'id': None
                })
        
        return results
    
    def train_solutions_parallel(self, solutions: List[Solution], data_path: str = '/workspace/dataset.npz', 
                                  timeout: int = 3600, use_sync: bool = False) -> List[Dict[str, Any]]:
        """
        Dispatch solutions for training and wait for results.
        
        Args:
            solutions: List of Solution objects to train
            data_path: Path to dataset file
            timeout: Maximum time to wait for completion (seconds)
            use_sync: If True, use synchronous execution (simpler, but blocks per job)
        
        Returns:
            List of result dictionaries
        """
        if use_sync:
            # Use synchronous execution - each job waits for completion
            print("Using synchronous execution (each job waits for completion)...")
            results = self.dispatch_training_jobs(solutions, data_path, use_sync=True)
            # Results are already the output dictionaries
            return results
        else:
            # Dispatch all jobs (returns Job objects)
            jobs = self.dispatch_training_jobs(solutions, data_path, use_sync=False)
            
            # Wait for completion
            results = self.wait_for_jobs(jobs, timeout)
            
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

