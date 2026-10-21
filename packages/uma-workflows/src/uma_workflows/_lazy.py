from __future__ import annotations

from importlib import import_module
from typing import Any


def resolve_export(
    name: str,
    exports: dict[str, str],
    namespace: dict[str, Any],
) -> Any:
    module_name = exports.get(name)
    if module_name is None:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    namespace[name] = value
    return value
