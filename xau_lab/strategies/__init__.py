from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import get_strategy, list_strategies, register_strategy

# Keep research-only strategy families registered whenever the package is imported.
# The original v1 sampler still filters strictly to its frozen approved families.
from xau_lab.strategies import edge_b as _edge_b  # noqa: F401,E402

__all__ = [
    "StrategyContext",
    "StrategyDefinition",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
