"""Tests for BaseStrategy ABC — 5 tests."""

import pytest

from src.backtest.strategy.base import BaseStrategy, Action, ActionResult


class TestBaseStrategyABC:

    # 1. Cannot instantiate ABC directly
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    # 2. on_bar is abstract
    def test_on_bar_is_abstract(self):
        assert "on_bar" in BaseStrategy.__abstractmethods__

    # 3. name is abstract
    def test_name_is_abstract(self):
        assert "name" in BaseStrategy.__abstractmethods__

    # 4. config_dict is abstract
    def test_config_dict_is_abstract(self):
        assert "config_dict" in BaseStrategy.__abstractmethods__

    # 5. reset and on_day_start are default no-ops
    def test_reset_and_on_day_start_default_noop(self):
        """Concrete subclass with minimal implementation can call reset/on_day_start without error."""

        class MinimalStrategy(BaseStrategy):
            def on_bar(self, bar, position, session_minutes=390):
                return ActionResult(Action.HOLD)

            def name(self):
                return "minimal"

            def config_dict(self):
                return {}

        s = MinimalStrategy()
        # Should not raise
        s.reset()
        s.on_day_start("2026-03-20")
