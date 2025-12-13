from abc import ABC, abstractmethod
import itertools
from typing import Any

class Job(ABC):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_out_file(self) -> str:
        pass

def get_jobs(props_dict: dict[str, list[Any]] | None, config: Any, job_class: type[Job]):
    jobs = []
    # Generate cross product of other_sweep_props lists
    cross_product_sweep_props = []
    if props_dict is not None:
        # Generate cross product of all property combinations
        prop_names = list(props_dict.keys())
        prop_value_lists = [props_dict[name] for name in prop_names]
        for combination in itertools.product(*prop_value_lists) if prop_value_lists else [()]:
            prop_dict = dict(zip(prop_names, combination))
            cross_product_sweep_props.append(prop_dict)
    else:
        # If no other_sweep_props, just add empty dict
        cross_product_sweep_props = [{}]
    
    for props in cross_product_sweep_props:
        exp_config, experiment_base_dir = config._get_experiment_config_and_base_dir(**props)
        jobs.append(job_class(exp_config, experiment_base_dir, config.run_experiment_config))
    return jobs
