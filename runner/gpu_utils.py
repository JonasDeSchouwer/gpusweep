import os
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from runner.job import Job
from typing import List, Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass

@dataclass
class GPUJobResult:
    success: bool
    error: str | None
    gpu_id: int
    out_file: str | None
    job: Job
    result: Any | None

def _run_job_on_gpu(job: Job, gpu_id: int) -> GPUJobResult:
    """Run a single job on a specific GPU by calling job.run().
    
    Note: CUDA_VISIBLE_DEVICES should already be set by _gpu_worker before this is called.
    """
    device_str = f"GPU {gpu_id}"
    print(f"Starting job on {device_str} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    
    out_file = job.get_out_file()
    
    try:
        result_value = None
        if out_file is None:
            # No output file, run directly
            result_value = job.run()
        else:
            # Redirect output to file
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                print(f"Running job on {device_str}", file=f, flush=True)
                with redirect_stdout(f), redirect_stderr(f):
                    result_value = job.run()
            print(f"Finished job on {device_str}", flush=True)
        
        return GPUJobResult(
            success=True,
            error=None,
            gpu_id=gpu_id,
            out_file=out_file,
            job=job,
            result=result_value,
        )
    except Exception as e:
        print(f"Error running job on {device_str}: {str(e)}", flush=True)
        return GPUJobResult(
            success=False,
            error=str(e),
            gpu_id=gpu_id,
            out_file=out_file,
            job=job,
            result=None,
        )


def _gpu_worker(job_gpu_pairs: List[tuple]) -> List[GPUJobResult]:
    """Worker function that runs multiple jobs sequentially on the same GPU.
    
    Note: All jobs in job_gpu_pairs should have the same gpu_id since they're
    grouped by GPU. We set CUDA_VISIBLE_DEVICES once at the start for all jobs.
    """
    if not job_gpu_pairs:
        return []
    
    # All jobs should be on the same GPU, so get gpu_id from first job
    _, gpu_id = job_gpu_pairs[0]
    
    # Set CUDA_VISIBLE_DEVICES for this process before any CUDA operations
    # This is critical when using 'spawn' method - must be set before torch.cuda is accessed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now run all jobs on this GPU
    results = []
    for job, _ in job_gpu_pairs:
        result = _run_job_on_gpu(job, gpu_id)
        results.append(result)
    return results


def get_available_gpu_ids() -> List[int]:
    """Get the available GPU device IDs.
    
    Returns:
        List of available GPU device IDs
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible.strip():
        return [int(d.strip()) for d in cuda_visible.split(",") if d.strip()]
    import torch
    return list(range(torch.cuda.device_count()))

def _run_single_job(job: Job, gpu_id: int) -> GPUJobResult:
    """Run a single job on a specific GPU in a subprocess.
    
    This function is designed to be run in a thread pool executor.
    """
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        result = pool.apply(_gpu_worker, ([(job, gpu_id)],))
        return result[0] if result else GPUJobResult(
            success=False,
            error="No result returned",
            gpu_id=gpu_id,
            out_file=job.get_out_file(),
            job=job,
            result=None,
        )

class GPUScheduler:
    """Scheduler for running jobs on GPUs with async support and internal queue management."""
    
    def __init__(self, max_gpus: Optional[int] = None, simultaneous_jobs_per_gpu: int = 1):
        """Initialize the GPU scheduler.
        
        Args:
            max_gpus: Maximum number of GPUs to use (defaults to all available)
            simultaneous_jobs_per_gpu: Number of simultaneous jobs to run on each GPU
        """
        available_gpu_ids = get_available_gpu_ids()
        if max_gpus is not None:
            available_gpu_ids = available_gpu_ids[:max_gpus]
        if not available_gpu_ids:
            raise ValueError("No available GPUs to use")
        
        self.gpu_ids = available_gpu_ids
        self.simultaneous_jobs_per_gpu = simultaneous_jobs_per_gpu
        self.semaphores = {gpu_id: asyncio.Semaphore(simultaneous_jobs_per_gpu) for gpu_id in self.gpu_ids}
        self.gpu_counter = 0  # For round-robin assignment
        self.executor = ThreadPoolExecutor()
        
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            pass
        
        print(f"GPUScheduler initialized with {len(self.gpu_ids)} GPUs (device IDs: {self.gpu_ids}), "
              f"{simultaneous_jobs_per_gpu} simultaneous jobs per GPU", flush=True)
    
    async def run_job(self, job: Job) -> GPUJobResult:
        """Run a job on an available GPU slot.
        
        Args:
            job: Job object to run
            
        Returns:
            GPUJobResult with job result
        """
        # Round-robin assign to GPU
        gpu_id = self.gpu_ids[self.gpu_counter % len(self.gpu_ids)]
        self.gpu_counter += 1
        
        # Wait for available slot on this GPU
        async with self.semaphores[gpu_id]:
            # Run job in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: _run_single_job(job, gpu_id)
            )
            return result
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
