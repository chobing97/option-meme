"""Evaluation metrics: PR-AUC, time error analysis, simple backtesting."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)


def compute_pr_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds_to_report: Optional[list[float]] = None,
) -> dict:
    """Compute Precision-Recall metrics.

    Args:
        y_true: Binary ground truth labels.
        y_pred_proba: Predicted probabilities.
        thresholds_to_report: Specific thresholds for precision/recall reporting.

    Returns:
        Dict with PR-AUC and threshold-specific metrics.
    """
    if thresholds_to_report is None:
        thresholds_to_report = [0.3, 0.4, 0.5, 0.6, 0.7]

    pr_auc = average_precision_score(y_true, y_pred_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    results = {
        "pr_auc": float(pr_auc),
        "total_samples": len(y_true),
        "total_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }

    # Metrics at specific thresholds
    threshold_metrics = []
    for t in thresholds_to_report:
        y_pred = (y_pred_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        threshold_metrics.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "predicted_positives": int(y_pred.sum()),
        })

    results["threshold_metrics"] = threshold_metrics

    return results


def compute_time_error(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    target_label: int,
    threshold: float = 0.5,
    tolerance_minutes: int = 3,
) -> dict:
    """Analyze temporal accuracy of predictions vs actual turning points.

    For each actual turning point, find the nearest predicted turning point
    and measure the time difference in minutes.

    Args:
        df: DataFrame with 'label', 'datetime', 'date' columns.
        y_pred_proba: Predicted probabilities.
        target_label: 1 for peak, 2 for trough.
        threshold: Probability threshold for positive prediction.
        tolerance_minutes: Predictions within this window are "on time".

    Returns:
        Dict with time error statistics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    actual_mask = (df["label"] == target_label).values
    pred_mask = y_pred == 1

    # Indices of actual and predicted positives
    actual_indices = np.where(actual_mask)[0]
    pred_indices = np.where(pred_mask)[0]

    if len(actual_indices) == 0:
        return {"error": "no actual positives"}

    if len(pred_indices) == 0:
        return {"error": "no predicted positives", "total_actual": len(actual_indices)}

    # For each actual positive, find nearest predicted positive
    time_errors = []
    for ai in actual_indices:
        if len(pred_indices) == 0:
            continue
        distances = np.abs(pred_indices - ai)
        nearest_pred = pred_indices[distances.argmin()]
        error_bars = int(nearest_pred - ai)  # positive = predicted later
        time_errors.append(error_bars)

    time_errors = np.array(time_errors)

    on_time = np.abs(time_errors) <= tolerance_minutes
    early = time_errors < -tolerance_minutes
    late = time_errors > tolerance_minutes

    results = {
        "total_actual": len(actual_indices),
        "total_predicted": len(pred_indices),
        "matched": len(time_errors),
        "mean_error_bars": float(np.mean(time_errors)),
        "median_error_bars": float(np.median(time_errors)),
        "std_error_bars": float(np.std(time_errors)),
        "mae_bars": float(np.mean(np.abs(time_errors))),
        "on_time_pct": float(on_time.mean()),
        "early_pct": float(early.mean()),
        "late_pct": float(late.mean()),
        "tolerance_minutes": tolerance_minutes,
        "error_distribution": {
            "within_1": float((np.abs(time_errors) <= 1).mean()),
            "within_2": float((np.abs(time_errors) <= 2).mean()),
            "within_3": float((np.abs(time_errors) <= 3).mean()),
            "within_5": float((np.abs(time_errors) <= 5).mean()),
        },
    }

    return results


def simple_backtest(
    df: pd.DataFrame,
    peak_proba: np.ndarray,
    trough_proba: np.ndarray,
    peak_threshold: float = 0.5,
    trough_threshold: float = 0.5,
    slippage_pct: float = 0.001,
) -> dict:
    """Simple backtest: sell at predicted peaks, buy at predicted troughs.

    Strategy:
    - Buy when trough is predicted (price expected to rise)
    - Sell when peak is predicted (price expected to fall)
    - Flat otherwise

    Args:
        df: DataFrame with 'close', 'datetime' columns.
        peak_proba: Peak prediction probabilities.
        trough_proba: Trough prediction probabilities.
        peak_threshold: Threshold for peak signals.
        trough_threshold: Threshold for trough signals.
        slippage_pct: Slippage per trade as fraction.

    Returns:
        Backtest results dict.
    """
    close = df["close"].values
    n = min(len(close), len(peak_proba), len(trough_proba))

    if n == 0:
        return {"error": "no data"}

    # Align arrays
    close = close[:n]
    peak_pred = (peak_proba[:n] >= peak_threshold).astype(int)
    trough_pred = (trough_proba[:n] >= trough_threshold).astype(int)

    # Track positions and PnL
    position = 0  # 0=flat, 1=long
    entry_price = 0.0
    trades = []
    equity = [1.0]

    for i in range(1, n):
        if position == 0 and trough_pred[i] == 1:
            # Buy signal
            position = 1
            entry_price = close[i] * (1 + slippage_pct)
        elif position == 1 and peak_pred[i] == 1:
            # Sell signal
            exit_price = close[i] * (1 - slippage_pct)
            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_idx": i - 1,
                "exit_idx": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
            })
            position = 0
            equity.append(equity[-1] * (1 + pnl_pct))
            continue

        equity.append(equity[-1])

    # Buy and hold benchmark
    buy_hold_return = (close[-1] - close[0]) / close[0]

    # Trade statistics
    if trades:
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        results = {
            "n_trades": len(trades),
            "total_return": float(equity[-1] - 1),
            "buy_hold_return": float(buy_hold_return),
            "win_rate": float(len(wins) / len(trades)),
            "avg_win": float(np.mean(wins)) if wins else 0,
            "avg_loss": float(np.mean(losses)) if losses else 0,
            "profit_factor": float(
                abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
            ),
            "max_drawdown": float(_max_drawdown(equity)),
            "sharpe_approx": float(
                np.mean(pnls) / np.std(pnls) * np.sqrt(252)
                if np.std(pnls) > 0 else 0
            ),
        }
    else:
        results = {
            "n_trades": 0,
            "total_return": 0.0,
            "buy_hold_return": float(buy_hold_return),
            "win_rate": 0.0,
        }

    return results


def full_evaluation(
    df: pd.DataFrame,
    peak_proba: np.ndarray,
    trough_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Run complete evaluation: PR metrics + time error + backtest.

    Args:
        df: DataFrame with 'label', 'close', 'datetime' columns.
        peak_proba: Peak prediction probabilities.
        trough_proba: Trough prediction probabilities.
        threshold: Decision threshold.

    Returns:
        Comprehensive evaluation dict.
    """
    y_true_peak = (df["label"] == 1).values[:len(peak_proba)]
    y_true_trough = (df["label"] == 2).values[:len(trough_proba)]

    results = {
        "peak": {
            "pr_metrics": compute_pr_metrics(y_true_peak, peak_proba),
            "time_error": compute_time_error(df.iloc[:len(peak_proba)], peak_proba, 1, threshold),
        },
        "trough": {
            "pr_metrics": compute_pr_metrics(y_true_trough, trough_proba),
            "time_error": compute_time_error(df.iloc[:len(trough_proba)], trough_proba, 2, threshold),
        },
        "backtest": simple_backtest(
            df, peak_proba, trough_proba, threshold, threshold,
        ),
    }

    logger.info(
        f"Evaluation: peak PR-AUC={results['peak']['pr_metrics']['pr_auc']:.4f}, "
        f"trough PR-AUC={results['trough']['pr_metrics']['pr_auc']:.4f}, "
        f"backtest return={results['backtest'].get('total_return', 0):.4f}"
    )

    return results


def _max_drawdown(equity: list[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd
