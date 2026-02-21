"""Tests for src.model.evaluate."""

import numpy as np
import pandas as pd

from src.model.evaluate import (
    _max_drawdown,
    compute_pr_metrics,
    compute_time_error,
    full_evaluation,
    simple_backtest,
)
from tests.model.conftest import _make_eval_df


class TestComputePrMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        result = compute_pr_metrics(y_true, y_pred)
        assert result["pr_auc"] == 1.0

    def test_random_predictions_low_auc(self):
        rng = np.random.RandomState(42)
        y_true = np.zeros(200)
        y_true[:10] = 1
        y_pred = rng.rand(200)
        result = compute_pr_metrics(y_true, y_pred)
        assert result["pr_auc"] < 1.0

    def test_result_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_pr_metrics(y_true, y_pred)
        assert "pr_auc" in result
        assert "total_samples" in result
        assert "threshold_metrics" in result

    def test_threshold_metrics_structure(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_pr_metrics(y_true, y_pred)
        for tm in result["threshold_metrics"]:
            assert "threshold" in tm
            assert "precision" in tm
            assert "recall" in tm
            assert "f1" in tm


class TestComputeTimeError:
    def test_perfect_alignment(self):
        df = _make_eval_df(20, peak_idx=[5, 15])
        proba = np.zeros(20)
        proba[5] = 0.9
        proba[15] = 0.9
        result = compute_time_error(df, proba, target_label=1, threshold=0.5)
        assert result["mean_error_bars"] == 0
        assert result["on_time_pct"] == 1.0

    def test_early_predictions(self):
        df = _make_eval_df(20, peak_idx=[10])
        proba = np.zeros(20)
        proba[5] = 0.9  # predicted 5 bars earlier
        result = compute_time_error(df, proba, target_label=1, threshold=0.5)
        assert result["mean_error_bars"] < 0

    def test_no_actual_positives(self):
        df = _make_eval_df(20)  # no peaks
        proba = np.zeros(20)
        proba[5] = 0.9
        result = compute_time_error(df, proba, target_label=1, threshold=0.5)
        assert "error" in result

    def test_no_predicted_positives(self):
        df = _make_eval_df(20, peak_idx=[5])
        proba = np.zeros(20)  # all below threshold
        result = compute_time_error(df, proba, target_label=1, threshold=0.5)
        assert "error" in result


class TestSimpleBacktest:
    def test_no_signals_no_trades(self):
        df = _make_eval_df(50)
        peak_proba = np.zeros(50)
        trough_proba = np.zeros(50)
        result = simple_backtest(df, peak_proba, trough_proba)
        assert result["n_trades"] == 0

    def test_buy_low_sell_high(self):
        n = 50
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-02 09:00", periods=n, freq="min"),
                "close": np.linspace(100, 150, n),  # steadily rising
            }
        )
        trough_proba = np.zeros(n)
        peak_proba = np.zeros(n)
        trough_proba[5] = 0.9  # buy early
        peak_proba[40] = 0.9  # sell late
        result = simple_backtest(df, peak_proba, trough_proba, slippage_pct=0.0)
        assert result["total_return"] > 0
        assert result["n_trades"] == 1

    def test_result_keys_with_trades(self):
        n = 50
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-02 09:00", periods=n, freq="min"),
                "close": np.linspace(100, 150, n),
            }
        )
        trough_proba = np.zeros(n)
        peak_proba = np.zeros(n)
        trough_proba[5] = 0.9
        peak_proba[40] = 0.9
        result = simple_backtest(df, peak_proba, trough_proba)
        for key in ["win_rate", "max_drawdown", "sharpe_approx", "n_trades", "total_return"]:
            assert key in result


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = [1.0, 1.1, 1.2, 1.3]
        assert _max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = [1.0, 0.5, 1.0]
        assert _max_drawdown(equity) == 0.5


class TestFullEvaluation:
    def test_result_structure(self):
        df = _make_eval_df(100, peak_idx=[10, 50, 80], trough_idx=[20, 60, 90])
        rng = np.random.RandomState(42)
        peak_proba = rng.rand(100)
        trough_proba = rng.rand(100)
        result = full_evaluation(df, peak_proba, trough_proba)
        assert "peak" in result
        assert "trough" in result
        assert "backtest" in result
        assert "pr_metrics" in result["peak"]
        assert "time_error" in result["peak"]
