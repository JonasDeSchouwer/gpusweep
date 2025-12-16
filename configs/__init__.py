"""
Configuration system for mlps.

Provides @pydraclass decorator for creating strict, auto-finalizing config classes
with enhanced CLI parsing support.
"""

from configs.base_config import pydraclass, REQUIRED, ConfigMeta
from configs.cli import main, run, apply_overrides

__all__ = [
    'pydraclass',
    'REQUIRED',
    'main',
    'run',
    'apply_overrides',
    'ConfigMeta',
]
