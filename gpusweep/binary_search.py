import time
import asyncio
import warnings
import pandas as pd
from typing import Any, Callable
from gpusweep.gpu_utils import GPUScheduler
from gpusweep.configs.search_configs import BinarySearchConfig
from gpusweep.search_utils import Job, get_jobs

class BinarySearchJob(Job):
    def __init__(self, config: Any, base_dir: str, run_experiment_fn: Callable[[Any], Any]):
        self.config = config
        self.run_fn = run_experiment_fn
        self.base_dir = base_dir
    
    def run(self):
        return self.run_fn(self.config)

    def get_out_file(self) -> str:
        # Ensure base_dir is not empty to avoid creating files in root
        if not self.base_dir:
            raise ValueError("base_dir cannot be empty")
        return f"{self.base_dir}/experiment_output.log"

def get_jobs_for_mid(binary_config: BinarySearchConfig, mid: float):
    # Compose sweep props to include seeds and current mid value, plus any user-provided sweep props
    combined_sweep_props = dict(binary_config.sweep_props) if binary_config.sweep_props else {}
    combined_sweep_props[binary_config.prop] = [mid]
    return get_jobs(combined_sweep_props, binary_config, BinarySearchJob)

async def run_binary_search_job(job: BinarySearchJob, gpu_scheduler: GPUScheduler):
    return await gpu_scheduler.run_job(job)

async def run_binary_search(binary_config: BinarySearchConfig, gpu_scheduler: GPUScheduler, mid_selection_fn: Callable[[float, float], float] = lambda lo, hi: (lo + hi) / 2):
    start_time = time.time()
    lo, hi = binary_config.range
    precision = binary_config.precision

    # Log config to YAML
    # config_filename = f"{binary_config.base_dir}/binary_search_config.yaml"
    # log_config_to_yaml(config, config_filename)

    # Track results
    achieved_results = None
    failed_results = None
    error_results = None
    
    print(f"Binary search for {binary_config.prop} in [{lo}, {hi}], precision={precision}", flush=True)
    while (hi - lo) >= precision:
        mid = mid_selection_fn(lo, hi)
        print(f"  Testing {binary_config.prop}={mid} (range: [{lo}, {hi}])", flush=True)
        
        # Run training job
        jobs = get_jobs_for_mid(binary_config, mid)
        job_results = await asyncio.gather(*[run_binary_search_job(job, gpu_scheduler) for job in jobs])
        success, aggregated_result = binary_config.agg_results(job_results)
        
        if success is None:
            # Inconclusive: infrastructure error (e.g. OOM, CUDA error)
            warnings.warn(
                f"Inconclusive result at {binary_config.prop}={mid} due to infrastructure error. "
                f"Treating as failure for search continuation, but result may be unreliable."
            )
            error_results = (mid, aggregated_result)
            # Proceed same as failure
            if binary_config.success_direction_lower: lo = mid
            else: hi = mid
        elif success:
            # Success: this value works, try smaller
            print(f"Succeeded at {mid}")
            achieved_results = (mid, aggregated_result)
            if binary_config.success_direction_lower: hi = mid
            else: lo = mid
        else:
            print(f"Failed at {mid}")
            # Failure: try larger value
            failed_results = (mid, aggregated_result)
            if binary_config.success_direction_lower: lo = mid
            else: hi = mid
    
    print(f"Binary search complete for {binary_config.prop}", flush=True)
    print(f"Result: achieved_results={achieved_results}, failed_results={failed_results}, error_results={error_results}", flush=True)
    
    end_time = time.time()
    # Save binary search results to pickle file
    binary_search_results = {
        "search_range": [lo, hi],
        "precision": precision,
        "achieved_results": achieved_results,
        "failed_results": failed_results,
        "error_results": error_results,
        "total_time": end_time - start_time,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Create results filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"{binary_config.base_dir}/binary_search_results_{timestamp}.pkl"
    pd.to_pickle(binary_search_results, results_filename)
    print(f"Binary search results saved to: {results_filename}", flush=True)
    return achieved_results, failed_results, error_results


async def run_binary_searches_wrapper(configs, gpu_scheduler, mid_selection_fn: Callable[[float, float], float] = lambda lo, hi: (lo + hi) / 2):
    return await asyncio.gather(*[run_binary_search(config, gpu_scheduler, mid_selection_fn) for config in configs])

def run_binary_searches(configs: list[BinarySearchConfig], max_gpus: int | None = None, simultaneous_jobs_per_gpu: int | None = None, mid_selection_fn: Callable[[float, float], float] = lambda lo, hi: (lo + hi) / 2):
    gpu_scheduler = GPUScheduler(max_gpus=max_gpus, simultaneous_jobs_per_gpu=simultaneous_jobs_per_gpu)
    try:
        return asyncio.run(run_binary_searches_wrapper(configs, gpu_scheduler, mid_selection_fn))
    finally:
        gpu_scheduler.shutdown()