from .base import StrategyDefinition
_REGISTRY={}
def register_strategy(d: StrategyDefinition):
    if d.name in _REGISTRY: raise ValueError(f'duplicate strategy {d.name}')
    _REGISTRY[d.name]=d; return d
def get_strategy(name: str)->StrategyDefinition: return _REGISTRY[name]
def list_strategies(): return sorted(_REGISTRY)
def clear_registry(): _REGISTRY.clear()
