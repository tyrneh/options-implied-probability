"""Pipeline orchestration namespace with lazy submodule loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    "estimator",
    "market_inputs",
    "vol_estimation",
    "prob_estimation",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"oipd.pipelines.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'oipd.pipelines' has no attribute '{name}'")  # pragma: no cover


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(globals()) | set(__all__))
