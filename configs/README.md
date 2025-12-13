# Configuration System

A dataclass-based configuration system with strict validation, automatic finalization, and enhanced CLI parsing.

## Features

- **@pydraclass decorator**: Creates config classes with automatic `__init__` generation
- **Strict attribute validation**: Catches typos with helpful error messages
- **Recursive finalization**: Automatically finalizes all nested configs (including those in lists/dicts/tuples)
- **Enhanced CLI parsing**: Safe expression evaluation with `ast.literal_eval`
- **Serialization**: Export to dict/YAML/pickle/dill
- **Type hints**: Full IDE support with autocomplete

## Quick Start

### Basic Config

```python
from mlps.models.mlp.configs.base_config import pydraclass
from dataclasses import field

@pydraclass
class TrainConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

config = TrainConfig()
config.learning_rate = 0.01  # ✅ Valid
config.learning_rat = 0.01   # ❌ Raises InvalidConfigurationError with suggestion
```

### Nested Configs

```python
@pydraclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001

@pydraclass
class ModelConfig:
    hidden_size: int = 128
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

config = ModelConfig()
config.optimizer.lr = 0.01
```

**Important**: Use `field(default_factory=ConfigClass)` for nested configs to avoid shared instances!

### Finalization

Configs support a `finalize()` hook for custom validation:

```python
@pydraclass
class Config:
    batch_size: int = 32
    max_batch_size: int = 128

    def finalize(self):
        if self.batch_size > self.max_batch_size:
            raise ValueError("batch_size exceeds max_batch_size")

config = Config()
config.batch_size = 256
config._finalize()  # Raises ValueError
```

The `_finalize()` method automatically:
1. Recursively finalizes all nested configs (even in lists/dicts/tuples)
2. Calls your custom `finalize()` hook
3. Marks the config as finalized

### CLI Usage

```python
from mlps.models.mlp.configs.cli import main

@pydraclass
class TrainConfig:
    learning_rate: float = 0.001
    batch_size: int = 32

@main(TrainConfig)
def train(config: TrainConfig):
    print(f"Training with lr={config.learning_rate}, batch_size={config.batch_size}")

if __name__ == "__main__":
    train()  # Automatically parses CLI args
```

Run with:
```bash
# Use defaults
python train.py

# Override single values
python train.py learning_rate=0.01 batch_size=64

# Use complex Python literals (lists, dicts, tuples, etc.)
python train.py 'layers=[64,128,256]' 'params={"dropout":0.1}'

# Show config without running
python train.py --show learning_rate=0.01

# Nested configs
python train.py optimizer.lr=0.01 optimizer.weight_decay=1e-4
```

### CLI Expression Evaluation

The CLI parser uses `ast.literal_eval()` for safe parsing:

```bash
# These all work safely:
python train.py 'values=[1,2,3]'              # list
python train.py 'params={"a":1,"b":2}'        # dict
python train.py 'point=(1.0,2.0)'             # tuple
python train.py 'enabled={True,False}'        # set
python train.py 'ratio=1e-3'                  # float
python train.py 'name="model"'                # string
```

For more complex expressions (e.g., `2**8`, `math.sqrt(2)`), use `--eval` mode:

```bash
# ONLY for trusted input!
python train.py --eval 'hidden_size=2**8' 'threshold=math.sqrt(2)'
```

## Migration from StrictConfig

### Before (StrictConfig)

```python
class MyConfig(StrictConfig):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.optimizer = OptimizerConfig()  # ❌ Shared instance bug!
```

### After (@pydraclass)

```python
@pydraclass
class MyConfig:
    learning_rate: float = 0.001
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # ✅ Lazy instantiation
```

## Key Differences from pydra.Config

1. **Strict validation**: Invalid attribute names raise errors (prevents typos)
2. **Recursive finalization**: Auto-discovers and finalizes configs in nested structures
3. **Better CLI parsing**: `ast.literal_eval()` instead of custom parser (safer, more powerful)
4. **Dataclass-based**: Standard Python feature with better IDE support

## API Reference

### @pydraclass

Decorator that creates a strict, auto-finalizing config class.

```python
@pydraclass
class MyConfig:
    param: type = default_value
```

### ConfigMeta Methods

All `@pydraclass` decorated classes have these methods:

- `_finalize()`: Recursively finalize all nested configs, then call `finalize()`
- `finalize()`: User-defined hook for custom validation (override this)
- `to_dict()`: Convert config to dictionary
- `save_yaml(path)`: Save config to YAML file
- `save_pickle(path)`: Save config to pickle file
- `save_dill(path)`: Save config to dill file

### CLI Functions

- `main(ConfigClass)`: Decorator for main functions that take a config argument
- `run(fn)`: Run a function with config parsed from CLI (infers config type from annotation)
- `apply_overrides(config, args)`: Manually apply CLI overrides to a config

## Examples

See:
- `test_new_config.py` - Full test suite demonstrating all features
- `test_cli.py` - CLI parsing examples
- `task_config.py` - Migrated core configs (ActivationConfig, MLPConfig, SharedConstructionConfig, MLPTask)

## Files

- `base_config.py` - Core `@pydraclass` decorator and `ConfigMeta` class
- `cli.py` - Enhanced CLI parsing with `ast.literal_eval`
- `task_config.py` - Migrated core configs using `@pydraclass`
- `strict_config.py` - Old system (deprecated, will be removed eventually)
