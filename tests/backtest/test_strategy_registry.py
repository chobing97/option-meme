"""Tests for strategy registry — 6 tests."""

import pytest

from src.backtest.strategy import (
    create_strategy,
    list_strategies,
    STRATEGY_REGISTRY,
    PutBuyStrategy,
    FilteredPutStrategy,
    CallBuyStrategy,
    BaseStrategy,
)


class TestStrategyRegistry:

    # 1. create_strategy("put_buy") returns PutBuyStrategy
    def test_create_put_buy(self):
        s = create_strategy("put_buy", threshold=0.3)
        assert isinstance(s, PutBuyStrategy)
        assert s.config.threshold == 0.3
        assert s.name() == "put_buy"

    # 2. create_strategy("filtered_put") returns FilteredPutStrategy
    def test_create_filtered_put(self):
        s = create_strategy("filtered_put", threshold=0.5, min_holding_minutes=15)
        assert isinstance(s, FilteredPutStrategy)
        assert s.config.threshold == 0.5
        assert s.config.min_holding_minutes == 15

    # 3. create_strategy("call_buy") returns CallBuyStrategy
    def test_create_call_buy(self):
        s = create_strategy("call_buy", threshold=0.4)
        assert isinstance(s, CallBuyStrategy)
        assert s.config.option_type == "call"

    # 4. Unknown strategy raises ValueError
    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("nonexistent")

    # 5. list_strategies returns sorted list
    def test_list_strategies(self):
        names = list_strategies()
        assert names == sorted(names)
        assert "put_buy" in names
        assert "filtered_put" in names
        assert "call_buy" in names

    # 6. All registry entries produce BaseStrategy instances
    def test_registry_all_basestrategy(self):
        for name in STRATEGY_REGISTRY:
            s = create_strategy(name)
            assert isinstance(s, BaseStrategy)
            assert s.name() == name
            assert isinstance(s.config_dict(), dict)
