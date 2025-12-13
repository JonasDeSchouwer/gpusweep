# Runner

A Python framework for efficient hyperparameter search with GPU resource management, supporting both binary search and grid search strategies.

## Features

- **Binary Search**: Efficiently find optimal hyperparameter values within a range
- **Grid Search**: Exhaustive search across all combinations of hyperparameter values
- **GPU Management**: Automatic GPU scheduling with support for parallel jobs per GPU
- **Configuration System**: Strict, validated configuration classes with nested config support
- **Async Execution**: Asynchronous job scheduling for optimal resource utilization
- **Result Aggregation**: Flexible result aggregation across multiple runs/seeds

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Define Your Experiment

Create an experiment configuration and run function:

```python
from configs import pydraclass, main

@pydraclass
class DummyExperimentConfig:
    name: str = "dummy_experiment"
    num_parameters: int = 10
    seed: int = 42

def run_experiment(config: DummyExperimentConfig):
    # Your experiment logic here
    result = perform_training(config)
    return result

@main(DummyExperimentConfig)
def main(config: DummyExperimentConfig):
    run_experiment(config)
```

### 2. Binary Search

Find the optimal value for a hyperparameter using binary search:

```python
from runner.binary_search import run_binary_searches
from runner.configs.search_configs import BinarySearchConfig
from examples.dummy_experiment import run_experiment, DummyExperimentConfig

@pydraclass
class ExperimentBinarySearchConfig(BinarySearchConfig):
    def get_experiment_config_and_base_dir(self, num_parameters: int, seed: int):
        config = copy.deepcopy(self.base_experiment_config)
        config.num_parameters = num_parameters
        config.seed = seed
        config.base_dir = f"{self.base_dir}/num_parameters_{num_parameters}_seed_{seed}"
        config.finalize()
        return config, config.base_dir

    def run_experiment_config(self, config):
        return run_experiment(config)

    def agg_results(self, results: list[GPUJobResult]) -> tuple[bool, Any]:
        # Aggregate results across seeds
        results = [r for r in results if r.success]
        if len(results) == 0:
            return False, None
        best_idx = np.argmax([r.result for r in results])
        result = results[best_idx]
        # Return (success, result) - success is True if result >= threshold
        return result.result >= 0.5, result

# Run binary search
configs = [ExperimentBinarySearchConfig(
    base_dir="./results/binary_search",
    prop="num_parameters",  # Property to search over
    range=(10, 100),         # Search range
    precision=1,             # Precision for stopping
    success_direction_lower=True,  # True if lower values are better
    sweep_props={"seed": [42, 43, 44, 45]},  # Other properties to sweep (these will get aggregated in agg_results!)
    base_experiment_config=DummyExperimentConfig(name="experiment_1"),
)]

run_binary_searches(configs, max_gpus=4, simultaneous_jobs_per_gpu=2)
```

### 3. Grid Search

Exhaustively search all combinations of hyperparameters:

```python
from runner.grid_search import run_grid_searches
from runner.configs.search_configs import GridSearchConfig
from examples.dummy_experiment import run_experiment, DummyExperimentConfig
import copy
import numpy as np

@pydraclass
class ExperimentGridSearchConfig(GridSearchConfig):
    def get_experiment_config_and_base_dir(self, num_parameters: int, seed: int):
        config = copy.deepcopy(self.base_experiment_config)
        config.num_parameters = num_parameters
        config.seed = seed
        config.base_dir = f"{self.base_dir}/num_parameters_{num_parameters}_seed_{seed}"
        config.finalize()
        return config, config.base_dir

    def run_experiment_config(self, config):
        return run_experiment(config)

    def agg_results(self, results: list[GPUJobResult]):
        # Aggregate results across all sweep_props points
        results = [r for r in results if r.success]
        if len(results) == 0:
            return None
        # Return best result
        best_idx = np.argmax([r.result for r in results])
        return results[best_idx]

# Run grid search
configs = [ExperimentGridSearchConfig(
    base_dir=f"./results/grid_search_{num_parameters}",
    sweep_props={
        "seed": [42, 43, 44, 45]
    },
    base_experiment_config=DummyExperimentConfig(name="experiment_1", num_parameters=num_parameters),
) for num_parameters in [10, 20, 30, 40]]

run_grid_searches(configs, max_gpus=4, simultaneous_jobs_per_gpu=2)
```

## Architecture

### Configuration System

The framework uses a strict configuration system based on dataclasses with the `@pydraclass` decorator:

- **Strict Validation**: Prevents typos by validating attribute names
- **Nested Configs**: Support for nested configuration objects
- **Finalization**: Automatic recursive finalization of configs
- **CLI Support**: Built-in CLI argument parsing

See `configs/README.md` for detailed documentation on the configuration system.

### GPU Scheduling

The `GPUScheduler` manages GPU resources:

- **Round-robin Assignment**: Jobs are distributed across available GPUs
- **Concurrent Execution**: Multiple jobs can run simultaneously on each GPU
- **Process Isolation**: Each job runs in a separate process with proper CUDA device isolation
- **Error Handling**: Failed jobs are tracked and reported

### Search Strategies

#### Binary Search

Binary search efficiently finds the optimal value for a single hyperparameter:

- Searches within a specified range `[lo, hi]`
- Stops when the range is smaller than `precision`
- Supports both directions (lower is better / higher is better)
- Can sweep over additional properties (e.g., seeds) at each test point

#### Grid Search

Grid search exhaustively tests all combinations:

- Generates the cross product of all `sweep_props` values
- Runs experiments for each combination
- Aggregates results across all runs

## API Reference

### Binary Search

#### `run_binary_searches(configs, max_gpus=None, simultaneous_jobs_per_gpu=None)`

Run multiple binary searches in parallel.

**Parameters:**
- `configs`: List of `BinarySearchConfig` instances
- `max_gpus`: Maximum number of GPUs to use (default: all available)
- `simultaneous_jobs_per_gpu`: Number of concurrent jobs per GPU (default: 1)

#### `BinarySearchConfig`

Configuration for binary search.

**Required Methods:**
- `get_experiment_config_and_base_dir(**prop_values)`: Create experiment config for given property values
- `run_experiment_config(config)`: Run the experiment and return result
- `agg_results(results)`: Aggregate results across runs, return `(success: bool, result: Any)`

**Properties:**
- `prop`: Name of the property to search over
- `range`: Tuple `(lo, hi)` defining search range
- `precision`: Minimum range size to stop search
- `success_direction_lower`: If `True`, lower values are considered better
- `sweep_props`: Dict of additional properties to sweep (e.g., `{"seed": [42, 43]}`)
- `base_experiment_config`: Base experiment configuration
- `base_dir`: Base directory for results

### Grid Search

#### `run_grid_searches(configs, max_gpus=None, simultaneous_jobs_per_gpu=None)`

Run multiple grid searches in parallel.

**Parameters:**
- `configs`: List of `GridSearchConfig` instances
- `max_gpus`: Maximum number of GPUs to use (default: all available)
- `simultaneous_jobs_per_gpu`: Number of concurrent jobs per GPU (default: 1)

#### `GridSearchConfig`

Configuration for grid search.

**Required Methods:**
- `get_experiment_config_and_base_dir(**prop_values)`: Create experiment config for given property values
- `run_experiment_config(config)`: Run the experiment and return result
- `agg_results(results)`: Aggregate results across runs, return aggregated result

**Properties:**
- `sweep_props`: Dict of properties to sweep (e.g., `{"lr": [0.001, 0.01], "seed": [42, 43]}`)
- `base_experiment_config`: Base experiment configuration
- `base_dir`: Base directory for results

### GPU Utilities

#### `GPUScheduler(max_gpus=None, simultaneous_jobs_per_gpu=1)`

Manages GPU resources for job execution.

**Methods:**
- `async run_job(job)`: Run a job on an available GPU slot

#### `GPUJobResult`

Result from a GPU job execution.

**Attributes:**
- `success`: Whether the job completed successfully
- `error`: Error message if job failed
- `gpu_id`: GPU device ID used
- `out_file`: Path to output log file
- `job`: Original job object
- `result`: Return value from `job.run()`

## Examples

See the `examples/` directory for complete working examples:

- `dummy_experiment.py`: Basic experiment setup
- `example_binary_search.py`: Binary search example
- `example_grid_search.py`: Grid search example

## Results

Results are automatically saved:

- **Binary Search**: Results saved to `{base_dir}/binary_search_results_{timestamp}.pkl`
- **Grid Search**: Results saved to `{base_dir}/grid_search_results_{timestamp}.pkl`
- **Job Outputs**: Each job's stdout/stderr saved to `{experiment_base_dir}/experiment_output.log`

## GPU Configuration

The scheduler automatically detects available GPUs:

- If `CUDA_VISIBLE_DEVICES` is set, only those GPUs are used
- Otherwise, all available GPUs are detected via `torch.cuda.device_count()`
- Each job runs in a separate process with proper CUDA device isolation
