from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from gpusweep.gpu_utils import GPUJobResult, GPUScheduler, get_available_gpu_ids
from gpusweep.binary_search import run_binary_search, run_binary_searches
from gpusweep.grid_search import run_grid_search, run_grid_searches

try:
    __version__ = version("gpusweep")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "GPUJobResult",
    "GPUScheduler",
    "get_available_gpu_ids",
    "run_binary_search",
    "run_binary_searches",
    "run_grid_search",
    "run_grid_searches",
]

