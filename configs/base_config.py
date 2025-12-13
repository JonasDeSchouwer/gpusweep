"""
Dataclass-based configuration system with strict validation and auto-finalization.

This module provides a @pydraclass decorator that creates configuration classes with:
- Automatic __init__ generation from type annotations
- Strict attribute validation (prevents typos)
- Recursive finalization of nested configs
- CLI parsing support with expression evaluation
- Serialization to dict/yaml
"""

import dataclasses
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, Set, TypeVar, Type
from pathlib import Path

class InvalidConfigurationError(ValueError):
    """Invalid configuration parameter error."""
    pass


def _is_config_instance(obj: Any) -> bool:
    """Check if an object is a Config instance."""
    return hasattr(obj, '_is_pydraclass') and obj._is_pydraclass


def _repr_recursive(obj: Any, indent_level: int = 0) -> str:
    """
    Recursive helper for pretty printing configs with indentation and color.

    Args:
        obj: Object to format (config instance or other value)
        indent_level: Current indentation level

    Returns:
        Formatted string with appropriate indentation and colors
    """
    # ANSI color codes - mild but helpful colors
    COLORS = {
        'class_name': '\033[38;5;111m',  # Mild blue
        'field_name': '\033[38;5;150m',  # Mild green
        'equals': '\033[38;5;247m',      # Gray
        'punctuation': '\033[38;5;247m', # Gray
        'reset': '\033[0m'
    }

    # Generate a consistent color for each class based on its name hash
    def get_class_color(class_name: str) -> str:
        """Get a consistent mild color for a class name."""
        # Use mild colors from ANSI 256-color palette (avoiding too bright or dark)
        mild_colors = [
            111,  # Mild blue
            150,  # Mild green
            180,  # Mild cyan
            186,  # Mild yellow
            182,  # Mild lime
            146,  # Mild purple
            174,  # Mild pink
            216,  # Mild peach
            152,  # Mild teal
            223,  # Mild gold
        ]
        color_idx = hash(class_name) % len(mild_colors)
        return f'\033[38;5;{mild_colors[color_idx]}m'

    indent = '\t' * indent_level
    next_indent = '\t' * (indent_level + 1)

    if _is_config_instance(obj):
        # Get the color for this specific class
        class_color = get_class_color(obj.__class__.__name__)
        lines = [f"{class_color}{obj.__class__.__name__}{COLORS['reset']}{COLORS['punctuation']}({COLORS['reset']}"]

        # Get all dataclass fields
        config_fields = fields(obj)

        for i, f in enumerate(config_fields):
            value = getattr(obj, f.name)

            # Format the value based on its type
            if _is_config_instance(value):
                # Recursively format nested configs
                value_repr = _repr_recursive(value, indent_level + 1)
            elif isinstance(value, list):
                # Format lists with potential nested configs
                if not value:
                    value_repr = f"{COLORS['punctuation']}[]{COLORS['reset']}"
                elif any(_is_config_instance(item) for item in value):
                    # Multi-line list with nested configs
                    list_items = []
                    for item in value:
                        if _is_config_instance(item):
                            list_items.append(_repr_recursive(item, indent_level + 2))
                        else:
                            list_items.append(repr(item))
                    list_content = f"{COLORS['punctuation']},{COLORS['reset']}\n{next_indent}\t".join(list_items)
                    value_repr = f"{COLORS['punctuation']}[{COLORS['reset']}\n{next_indent}\t{list_content}\n{next_indent}{COLORS['punctuation']}]{COLORS['reset']}"
                else:
                    # Simple list
                    value_repr = repr(value)
            elif isinstance(value, dict):
                # Format dicts with potential nested configs
                if not value:
                    value_repr = f"{COLORS['punctuation']}{{}}{COLORS['reset']}"
                elif any(_is_config_instance(v) for v in value.values()):
                    # Multi-line dict with nested configs
                    dict_items = []
                    for k, v in value.items():
                        if _is_config_instance(v):
                            v_repr = _repr_recursive(v, indent_level + 2)
                            dict_items.append(f"{repr(k)}{COLORS['punctuation']}:{COLORS['reset']} {v_repr}")
                        else:
                            dict_items.append(f"{repr(k)}{COLORS['punctuation']}:{COLORS['reset']} {repr(v)}")
                    dict_content = f"{COLORS['punctuation']},{COLORS['reset']}\n{next_indent}\t".join(dict_items)
                    value_repr = f"{COLORS['punctuation']}{{{COLORS['reset']}\n{next_indent}\t{dict_content}\n{next_indent}{COLORS['punctuation']}}}{COLORS['reset']}"
                else:
                    # Simple dict
                    value_repr = repr(value)
            else:
                # Use default repr for other types
                value_repr = repr(value)

            # Add the field line with field name in the current class's color
            is_last = (i == len(config_fields) - 1)
            comma = "" if is_last else f"{COLORS['punctuation']},{COLORS['reset']}"
            lines.append(f"{next_indent}{class_color}{f.name}{COLORS['reset']}{COLORS['equals']}={COLORS['reset']}{value_repr}{comma}")

        lines.append(f"{indent}{COLORS['punctuation']}){COLORS['reset']}")
        return "\n".join(lines)
    else:
        # Not a config instance, just return repr
        return repr(obj)


def _recursive_finalize(obj: Any,
                       visited: Set[int] | None = None,
                       stack: Set[int] | None = None) -> None:
    """
    Recursively finalize configs with circular reference detection.

    Uses depth-first bottom-up traversal: finalizes leaf configs first, then parents.
    Detects circular references by tracking the current call stack.

    Args:
        obj: Object to traverse and finalize
        visited: Set of object ids we've completed processing (optimization)
        stack: Set of object ids currently in the call stack (cycle detection)

    Raises:
        ValueError: If a circular reference is detected
    """
    if visited is None:
        visited = set()
    if stack is None:
        stack = set()

    obj_id = id(obj)

    # Already finished processing this object? Skip
    if obj_id in visited:
        return

    # Already in call stack? Circular reference!
    if obj_id in stack:
        if _is_config_instance(obj):
            raise ValueError(
                f"Circular reference detected: {obj.__class__.__name__} "
                f"references itself in a cycle"
            )
        else:
            raise ValueError(
                f"Circular reference detected in {type(obj).__name__} structure"
            )

    # Add to stack
    stack.add(obj_id)

    # Process based on type
    if _is_config_instance(obj):
        if obj._finalized:
            # Already finalized, just mark as visited
            stack.remove(obj_id)
            visited.add(obj_id)
            return

        # 1. Recursively finalize all fields (bottom-up)
        for f in fields(obj):
            value = getattr(obj, f.name)
            _recursive_finalize(value, visited, stack)

        # 2. Call custom_finalize on this config
        if hasattr(obj, 'custom_finalize'):
            obj.custom_finalize()

        # 3. Mark as finalized
        object.__setattr__(obj, '_finalized', True)

    elif isinstance(obj, dict):
        for value in obj.values():
            _recursive_finalize(value, visited, stack)

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _recursive_finalize(item, visited, stack)

    # Remove from stack and add to visited
    stack.remove(obj_id)
    visited.add(obj_id)


class ConfigMeta:
    """Mixin class that provides Config functionality to dataclasses."""

    # Note: Don't set these as class attributes, they should be instance attributes
    # _is_pydraclass: bool = True
    # _valid_attributes: Set[str] | None = None
    # _finalized: bool = False

    def __post_init__(self):
        """Called after dataclass __init__. Sets up validation."""
        # Store valid attributes for strict checking
        object.__setattr__(self, '_is_pydraclass', True)
        object.__setattr__(self, '_valid_attributes', self._get_valid_attributes())
        object.__setattr__(self, '_finalized', False)

    def _get_valid_attributes(self) -> Set[str]:
        """
        Get the set of valid attribute names for this config class.

        Returns:
            Set of valid attribute names from dataclass fields and special attrs
        """
        valid_attrs = set()

        # Add all dataclass field names
        for f in fields(self):
            valid_attrs.add(f.name)

        # Add special internal attributes
        valid_attrs.update(['_valid_attributes', '_finalized', '_is_pydraclass'])

        # Add any methods/properties from class hierarchy
        for cls in self.__class__.__mro__:
            for attr_name in dir(cls):
                if not attr_name.startswith('__'):
                    valid_attrs.add(attr_name)

        return valid_attrs

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override __setattr__ to validate parameter names.

        Args:
            name: The attribute name to set
            value: The value to assign

        Raises:
            InvalidConfigurationError: If the attribute name is not valid
        """
        # During __init__ and __post_init__, _valid_attributes might not exist yet
        valid_attrs = object.__getattribute__(self, '__dict__').get('_valid_attributes')
        if valid_attrs is None:
            object.__setattr__(self, name, value)
            return

        # Allow setting valid attributes
        if name in valid_attrs:
            object.__setattr__(self, name, value)
            return

        # Provide helpful error message with suggestions
        suggestions = self._get_similar_attributes(name)
        error_msg = f"Invalid parameter '{name}' for {self.__class__.__name__}"

        if suggestions:
            suggestions_str = "', '".join(suggestions)
            error_msg += f". Did you mean: '{suggestions_str}'?"
        else:
            # Show only user-defined fields, not internal attributes
            user_fields = [f.name for f in fields(self)]
            error_msg += f". Available parameters: {', '.join(sorted(user_fields))}"

        raise InvalidConfigurationError(error_msg)

    def _get_similar_attributes(self, name: str, max_distance: int = 2) -> list[str]:
        """
        Find attributes with names similar to the given name using edit distance.

        Args:
            name: The attribute name to find similar names for
            max_distance: Maximum edit distance to consider

        Returns:
            List of similar attribute names, sorted by similarity
        """
        def edit_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        similar = []
        # Only check user-defined fields for suggestions
        user_fields = {f.name for f in fields(self)}

        for attr in user_fields:
            distance = edit_distance(name, attr)
            if distance <= max_distance:
                similar.append((distance, attr))

        # Sort by distance, then by name
        similar.sort(key=lambda x: (x[0], x[1]))
        return [attr for _, attr in similar[:3]]  # Return top 3 matches

    def __repr__(self) -> str:
        """
        Pretty print representation of the config with nested indentation and colors.

        Returns:
            Formatted string representation with nested indentation and colors
        """
        return _repr_recursive(self, indent_level=0)

    def finalize(self) -> None:
        """
        Finalize this config and all nested configs using bottom-up traversal.

        This method delegates to _recursive_finalize() which:
        1. Recursively finalizes all nested configs (depth-first)
        2. Calls custom_finalize() on each config (leaves first, then parents)
        3. Detects circular references and raises ValueError if found

        Users should define custom_finalize() to add custom validation or computed fields,
        NOT override this method.

        Example:
            @pydraclass
            class NestedConfig:
                x: int = 1
                computed: int = 0

                def custom_finalize(self):
                    self.computed = self.x * 2

            @pydraclass
            class ParentConfig:
                nested: NestedConfig = field(default_factory=NestedConfig)
                total: int = 0

                def custom_finalize(self):
                    # nested is already finalized here
                    self.total = self.nested.computed + 10

        Raises:
            ValueError: If a circular reference is detected in the config structure
        """
        _recursive_finalize(self)

    def to_dict(self, yaml_compatible: bool = False) -> dict[str, Any]:
        """
        Convert config to a dictionary representation.

        Recursively converts nested configs and handles common types.

        Returns:
            Dictionary representation of the config
        """
        data = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if _is_config_instance(value):
                data[f.name] = value.to_dict(yaml_compatible=yaml_compatible)
            elif isinstance(value, (list, tuple)):
                data[f.name] = [
                    x.to_dict(yaml_compatible=yaml_compatible) if _is_config_instance(x) else x
                    for x in value
                ]
            elif isinstance(value, dict):
                data[f.name] = {
                    k: v.to_dict(yaml_compatible=yaml_compatible) if _is_config_instance(v) else v
                    for k, v in value.items()
                }
            else:
                if yaml_compatible and not isinstance(value, (int, float, str, bool)):
                    data[f.name] = str(value)
                else:
                    data[f.name] = value

        return data

    def save_yaml(self, path: Path | str) -> None:
        """
        Save config to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        import yaml
        path = Path(path)
        data = self.to_dict(yaml_compatible=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    def save_pickle(self, path: Path | str) -> None:
        """
        Save config to a pickle file.

        Args:
            path: Path to save the pickle file
        """
        import pickle
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save_dill(self, path: Path | str) -> None:
        """
        Save config to a dill file (supports more types than pickle).

        Args:
            path: Path to save the dill file
        """
        import dill
        path = Path(path)
        with open(path, 'wb') as f:
            dill.dump(self, f)


T = TypeVar('T')


def pydraclass(cls: Type[T]) -> Type[T]:
    """
    Decorator that creates a strict, auto-finalizing config class.

    This decorator:
    - Applies @dataclass to the class
    - Adds ConfigMeta functionality (strict validation, finalization, serialization)
    - Automatically generates __init__ from type annotations
    - Validates attribute names on assignment
    - Recursively finalizes nested configs

    Usage:
        @pydraclass
        class MyConfig:
            learning_rate: float = 0.001
            batch_size: int = 32
            optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

            def custom_finalize(self):
                # Custom finalization logic
                self.computed_field = self.learning_rate * 2

        config = MyConfig()
        config.learning_rate = 0.01  # ✅ Valid
        config.learning_rat = 0.01   # ❌ Raises InvalidConfigurationError
        config.finalize()             # Calls custom_finalize() + recursive finalization

    Args:
        cls: The class to decorate

    Returns:
        Decorated dataclass with Config functionality

    Raises:
        TypeError: If the class defines a finalize() method instead of custom_finalize()
    """
    # Check if user incorrectly defined finalize() instead of custom_finalize()
    if 'finalize' in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} defines a 'finalize()' method, but @pydraclass configs "
            f"should use 'custom_finalize()' instead.\n"
            f"Please rename 'def finalize(self):' to 'def custom_finalize(self):' in {cls.__name__}.\n"
            f"The finalize() method is reserved by the config system and handles "
            f"recursive finalization automatically."
        )

    # Store the original __post_init__ if it exists
    user_post_init = getattr(cls, '__post_init__', None)

    # Define our combined __post_init__
    def combined_post_init(self):
        # Always call ConfigMeta's initialization first
        ConfigMeta.__post_init__(self)
        # Then call user's __post_init__ if it exists
        if user_post_init is not None:
            user_post_init(self)

    # Add our combined __post_init__ to the class
    cls.__post_init__ = combined_post_init

    # Apply dataclass decorator (this processes field() objects)
    # Set repr=False to use our custom __repr__ from ConfigMeta instead
    cls = dataclass(cls, repr=False)

    # Now add ConfigMeta methods via inheritance
    # We need to create a new class that has both dataclass features and ConfigMeta features
    class ConfigClass(cls, ConfigMeta):
        pass

    # Preserve the original class name and module
    ConfigClass.__name__ = cls.__name__
    ConfigClass.__qualname__ = cls.__qualname__
    ConfigClass.__module__ = cls.__module__

    return ConfigClass


# Sentinel value for required parameters (compatibility with pydra)
class _Required:
    """Sentinel value indicating a required parameter."""
    def __repr__(self):
        return "REQUIRED"

REQUIRED = _Required()
