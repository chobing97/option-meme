from src.backtest.strategy.base import Action, ActionResult, BaseStrategy
from src.backtest.strategy.put_buy import PutBuyConfig, PutBuyStrategy
from src.backtest.strategy.filtered_put import FilteredPutConfig, FilteredPutStrategy
from src.backtest.strategy.call_buy import CallBuyConfig, CallBuyStrategy

# Backward compatibility aliases
Strategy = PutBuyStrategy
StrategyConfig = PutBuyConfig

STRATEGY_REGISTRY = {
    "put_buy": (PutBuyStrategy, PutBuyConfig),
    "filtered_put": (FilteredPutStrategy, FilteredPutConfig),
    "call_buy": (CallBuyStrategy, CallBuyConfig),
}


def create_strategy(name: str, **kwargs) -> BaseStrategy:
    """Create a strategy instance by name with keyword arguments for its config."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_REGISTRY.keys())}")
    cls, config_cls = STRATEGY_REGISTRY[name]
    config = config_cls(**kwargs)
    return cls(config)


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())
