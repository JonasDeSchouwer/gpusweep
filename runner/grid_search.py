import time
import asyncio
import pandas as pd
from typing import Any, Callable
from runner.gpu_utils import GPUScheduler
from runner.configs.search_configs import GridSearchConfig
from runner.search_utils import Job, get_jobs

class GridSearchJob(Job):
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

async def run_grid_search_job(job: GridSearchJob, gpu_scheduler: GPUScheduler):
    return await gpu_scheduler.run_job(job)

async def run_grid_search(grid_config: GridSearchConfig, gpu_scheduler: GPUScheduler):
    start_time = time.time()
    jobs = get_jobs(grid_config.sweep_props, grid_config, GridSearchJob)
    job_results = await asyncio.gather(*[run_grid_search_job(job, gpu_scheduler) for job in jobs])
    aggregated_result = grid_config.agg_results(job_results)
    print("Run complete")
    end_time = time.time()

    grid_search_results = {
        "results": aggregated_result,
        "total_time": end_time - start_time,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }

    # Create results filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"{grid_config.base_dir}/grid_search_results_{timestamp}.pkl"
    pd.to_pickle(grid_search_results, results_filename)
    print(f"Grid search results saved to: {results_filename}", flush=True)
    return aggregated_result

async def run_grid_searches_wrapper(configs, gpu_scheduler):
    return await asyncio.gather(*[run_grid_search(config, gpu_scheduler) for config in configs])

def run_grid_searches(configs: list[GridSearchConfig], max_gpus: int | None = None, simultaneous_jobs_per_gpu: int | None = None):
    gpu_scheduler = GPUScheduler(max_gpus=max_gpus, simultaneous_jobs_per_gpu=simultaneous_jobs_per_gpu)
    return asyncio.run(run_grid_searches_wrapper(configs, gpu_scheduler))