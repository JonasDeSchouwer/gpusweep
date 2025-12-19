from __future__ import annotations

from typing import Any, Protocol

from pydrafig import pydraclass
from datetime import datetime
from gpusweep.gpu_utils import GPUJobResult
from gpusweep.binary_search import run_binary_searches
from dummy_experiment import run_experiment, DummyExperimentConfig
from gpusweep.configs.search_configs import BinarySearchConfig
import copy
import numpy as np
import random


class ConfigLike(Protocol):
    def finalize(self) -> None: ...

@pydraclass
class ExperimentBinarySearchConfig(BinarySearchConfig):
    def get_experiment_config_and_base_dir(self, num_parameters: int, seed: int) -> tuple[ConfigLike, str]:
        # Extract distill_samples from prop_values (the property name comes from self.prop)
        config = copy.deepcopy(self.base_experiment_config)
        config.num_parameters = num_parameters
        config.seed = seed
        config.base_dir = f"{self.base_dir}/num_parameters_{num_parameters}_seed_{seed}"
        config.finalize()
        return config, config.base_dir

    def run_experiment_config(self, config: ConfigLike) -> Any:
        # this should just run the experiment and return the result
        return run_experiment(config)

    def agg_results(self, results: list[GPUJobResult]) -> tuple[bool, Any]:
        # result.success is True if the job completed without crashing
        results = [result for result in results if result.success]
        if len(results) == 0:
            return False, None
        # when aggregating, we pick the "best" result across all seeds, you can do whatever you want here
        # result.result contains the return value from run_experiment
        best_idx = np.argmax([result.result for result in results])
        result = results[best_idx]
        return result.result >= 0.5, result

def main():
    # we set up the base_dir and configs
    base_dir = f"./dev/example_binary_search_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    configs = []
    for name in ["experiment_1", "experiment_2", "experiment_3"]:
        configs.append(ExperimentBinarySearchConfig(
            base_dir=f"{base_dir}/binary_search_{name}_{random.randint(0, 1000)}",
            prop="num_parameters",
            range=(10, 100),
            precision=1,
            success_direction_lower=True,
            sweep_props={
                "seed": [42, 43, 44, 45]
            },
            base_experiment_config=DummyExperimentConfig(name=name),
        ))
    # finally we just run the binary searches
    run_binary_searches(configs, max_gpus=4, simultaneous_jobs_per_gpu=2)

if __name__ == "__main__":
    main()
