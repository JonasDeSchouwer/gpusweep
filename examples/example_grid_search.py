from typing import Any
from configs import pydraclass, ConfigMeta
from datetime import datetime
from runner.gpu_utils import GPUJobResult
from runner.grid_search import run_grid_searches
from dummy_experiment import run_experiment, DummyExperimentConfig
from runner.configs.search_configs import GridSearchConfig
import copy
import numpy as np
import random

@pydraclass
class ExperimentGridSearchConfig(GridSearchConfig):
    def get_experiment_config_and_base_dir(self, seed: int) -> tuple[ConfigMeta, str]:
        # Extract num_parameters and seed from prop_values (the property names come from sweep_props)
        config = copy.deepcopy(self.base_experiment_config)
        config.seed = seed
        config.base_dir = f"{self.base_dir}/_seed_{seed}"
        config.finalize()
        return config, config.base_dir

    def run_experiment_config(self, config: ConfigMeta) -> Any:
        # this should just run the experiment and return the result
        return run_experiment(config)

    def agg_results(self, results: list[GPUJobResult]) -> Any:
        # result.success is True if the job completed without crashing
        results = [result for result in results if result.success]
        if len(results) == 0:
            return None
        # when aggregating, we pick the "best" result across all grid points
        # result.result contains the return value from run_experiment
        best_idx = np.argmax([result.result for result in results])
        result = results[best_idx]
        return result

def main():
    # we set up the base_dir and configs
    base_dir = f"./dev/example_grid_search_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    configs = []
    for name in ["experiment_1", "experiment_2", "experiment_3"]:
        for num_parameters in [10, 20, 30, 40]:
            # every config is one point in the grid
            configs.append(ExperimentGridSearchConfig(
                base_dir=f"{base_dir}/binary_search_{name}_num_parameters_{num_parameters}_{random.randint(0, 1000)}",
                sweep_props={
                    "seed": [42, 43, 44, 45]
                },
                base_experiment_config=DummyExperimentConfig(name=name),
            ))
    # finally we just run the grid searches
    run_grid_searches(configs, max_gpus=4, simultaneous_jobs_per_gpu=2)

if __name__ == "__main__":
    main()
