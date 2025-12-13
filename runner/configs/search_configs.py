from configs import pydraclass, ConfigMeta
from typing import Any
from runner.gpu_utils import GPUJobResult

@pydraclass
class _BaseSearchConfig:
    base_dir: str | None = None
    sweep_props: dict[str, list[Any]] | None = None
    base_experiment_config: ConfigMeta | None = None

    def _get_experiment_config_and_base_dir(self, **prop_values) -> tuple[ConfigMeta, str]:
        config, base_dir = self.get_experiment_config_and_base_dir(**prop_values)
        config.finalize() # in case user forgot
        return config, base_dir

    def get_experiment_config_and_base_dir(self, **prop_values) -> tuple[ConfigMeta, str]:
        pass

    def run_experiment_config(self, config: ConfigMeta) -> Any:
        pass

    def agg_results(self, results: list[GPUJobResult]) -> Any:
        pass

@pydraclass
class BinarySearchConfig(_BaseSearchConfig):
    prop: str | None = None # property to search over
    range: tuple[float, float] | None = None
    precision: float | None = None
    success_direction_lower: bool = True

    def agg_results(self, results: list[GPUJobResult]) -> tuple[bool, Any]:
        # should return (success, result)
        pass

@pydraclass
class GridSearchConfig(_BaseSearchConfig):
    pass