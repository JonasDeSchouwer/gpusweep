"""
Unified CLI parsing for config overrides using Python statement execution.

This module provides simple CLI parsing by:
- Treating each CLI argument as a Python statement
- Auto-prepending 'config.' to access config attributes
- Using exec() for full Python expression support
- Supporting assignments, method calls, and nested access

Examples:
    python script.py "x = 10" "method(arg=5)" "nested.attr = 2**8"
    → Executes: config.x = 10; config.method(arg=5); config.nested.attr = 256
"""

import sys
from typing import Any, Type, Callable, TypeVar

import yaml

from moes.configs.base_config import _is_config_instance


def apply_overrides(
    config: Any,
    args: list[str],
    finalize: bool = True,
) -> bool:
    """
    Apply command-line overrides to a config object.

    Uses unified Python statement execution - each arg is prepended with 'config.'
    and executed via exec().

    Args:
        config: Config object to modify
        args: List of command-line arguments (Python statements)
        finalize: Whether to call finalize() after applying overrides

    Returns:
        True if --show flag was present, False otherwise

    Examples:
        apply_overrides(config, ["x = 10", "method(arg=5)"])
        → Executes: config.x = 10; config.method(arg=5)
    """
    import math

    show = False

    # Create namespace with config and common modules
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import torch
    except ImportError:
        torch = None

    namespace = {
        "config": config,
        "math": math,
        "np": np,
        "numpy": np,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "len": len,
        "range": range,
        "sum": sum,
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
        "pow": pow,
        "torch": torch,
    }

    # Build combined statement for single exec call (much faster)
    statements = []
    for arg in args:
        # Handle --show flag
        if arg == "--show":
            show = True
            continue

        # Prepend 'config.' to each statement
        statements.append(f"config.{arg}")

    if statements:
        # Update namespace with current config attributes for expressions
        if hasattr(config, '__dict__'):
            namespace.update(config.__dict__)

        # Combine all statements with newlines and execute once
        combined = "\n".join(statements)
        try:
            exec(combined, namespace)
        except Exception as e:
            raise type(e)(f"{str(e)}\n\nCombined statements: {combined}").with_traceback(e.__traceback__)

    if finalize and _is_config_instance(config):
        config.finalize()

    return show


T = TypeVar("T")
U = TypeVar("U")


def _apply_overrides_and_call(
    fn: Callable[[T], U],
    config_t: Type[T],
    args: list[str] | None = None
) -> U | None:
    """
    Internal helper: create config, apply overrides, and call function.

    Args:
        fn: Function to call with configured config
        config_t: Config class type
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Result of fn(config) or None if --show was used
    """
    config = config_t()

    if args is None:
        args = sys.argv[1:]

    show = apply_overrides(config, args, finalize=True)

    if show:
        print(yaml.dump(config.to_dict(yaml_compatible=True), sort_keys=False))
        return None

    return fn(config)


def main(base: Type[T]) -> Callable[[Callable[[T], U]], Callable[[list[str] | None], U | None]]:
    """
    Decorator for main functions that take a config argument.

    Usage:
        @pydraclass
        class MyConfig:
            learning_rate: float = 0.001

        @main(MyConfig)
        def train(config: MyConfig):
            print(f"Training with lr={config.learning_rate}")

        if __name__ == "__main__":
            train()  # Automatically parses CLI args

    Args:
        base: Config class type

    Returns:
        Decorator function
    """
    def decorator(fn: Callable[[T], U]) -> Callable[[list[str] | None], U | None]:
        def wrapped_fn(args: list[str] | None = None) -> U | None:
            return _apply_overrides_and_call(fn, base, args)
        return wrapped_fn
    return decorator


def run(fn: Callable[[T], U], args: list[str] | None = None) -> U | None:
    """
    Run a function with config parsed from CLI arguments.

    Infers the config type from the function's type annotation.

    Usage:
        def train(config: MyConfig):
            print(f"Training with lr={config.learning_rate}")

        if __name__ == "__main__":
            pydra.run(train)

    Args:
        fn: Function that takes a single config argument
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Result of fn(config)

    Raises:
        ValueError: If function signature is invalid
    """
    import inspect

    signature = inspect.signature(fn)
    params = signature.parameters

    if len(params) != 1:
        raise ValueError(f"Function '{fn.__name__}' must take exactly one argument")

    first_arg_type = list(params.values())[0].annotation

    if first_arg_type is inspect.Parameter.empty:
        raise ValueError(
            f"Function '{fn.__name__}' argument must have a type annotation"
        )

    return _apply_overrides_and_call(fn, first_arg_type, args)
