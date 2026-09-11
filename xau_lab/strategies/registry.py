from __future__ import annotations

from xau_lab.strategies.base import StrategyDefinition

_REGISTRY: dict[str, StrategyDefinition] = {}


def register_strategy(definition: StrategyDefinition) -> StrategyDefinition:
    name = definition.name
    if name in _REGISTRY:
        raise ValueError(f"duplicate strategy name: {name}")
    _REGISTRY[name] = definition
    return definition


def get_strategy(name: str) -> StrategyDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {name}") from exc


def list_strategies() -> tuple[StrategyDefinition, ...]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))
