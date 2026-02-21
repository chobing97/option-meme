"""LightGBM training with Optuna hyperparameter optimization."""

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
from loguru import logger
from sklearn.metrics import average_precision_score, classification_report

from config.settings import LGB_PARAMS, RANDOM_SEED
from src.model.dataset import SplitResult, prepare_xy


def train_lgb(
    split: SplitResult,
    target_label: int,
    feature_cols: Optional[list[str]] = None,
    params: Optional[dict] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM binary classifier for peak or trough detection.

    Args:
        split: Time-based train/val/test split.
        target_label: 1 for peak, 2 for trough.
        feature_cols: Feature columns. Auto-detected if None.
        params: LightGBM params. Uses defaults if None.
        num_boost_round: Max boosting rounds.
        early_stopping_rounds: Patience for early stopping.

    Returns:
        Tuple of (trained Booster, evaluation metrics dict).
    """
    if params is None:
        params = LGB_PARAMS.copy()

    X_train, y_train = prepare_xy(split.train, target_label, feature_cols)
    X_val, y_val = prepare_xy(split.val, target_label, feature_cols)
    X_test, y_test = prepare_xy(split.test, target_label, feature_cols)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    logger.info(
        f"Training LightGBM for label={target_label}: "
        f"train={len(y_train)} (pos={y_train.sum()}), "
        f"val={len(y_val)} (pos={y_val.sum()})"
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Evaluate on test set
    y_pred_proba = model.predict(X_test)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Feature importance
    importance = dict(
        zip(
            feature_cols or [f"f_{i}" for i in range(X_train.shape[1])],
            model.feature_importance(importance_type="gain"),
        )
    )
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    metrics = {
        "target_label": target_label,
        "pr_auc_test": pr_auc,
        "best_iteration": model.best_iteration,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "top_features": top_features,
    }

    logger.info(f"LightGBM test PR-AUC: {pr_auc:.4f}")

    return model, metrics


def optimize_lgb(
    split: SplitResult,
    target_label: int,
    feature_cols: Optional[list[str]] = None,
    n_trials: int = 50,
    timeout: Optional[int] = 3600,
) -> tuple[lgb.Booster, dict, dict]:
    """Optimize LightGBM hyperparameters with Optuna.

    Args:
        split: Time-based data split.
        target_label: 1 for peak, 2 for trough.
        feature_cols: Feature columns.
        n_trials: Number of Optuna trials.
        timeout: Max optimization time in seconds.

    Returns:
        Tuple of (best model, best metrics, best params).
    """
    X_train, y_train = prepare_xy(split.train, target_label, feature_cols)
    X_val, y_val = prepare_xy(split.val, target_label, feature_cols)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "is_unbalance": True,
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
        ]

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=1000,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_val)
        return average_precision_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize", seed=RANDOM_SEED)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Optuna best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params on full train+val
    best_params = {**LGB_PARAMS, **study.best_params}
    best_model, best_metrics = train_lgb(
        split, target_label, feature_cols, params=best_params,
    )

    return best_model, best_metrics, study.best_params


def save_model(model: lgb.Booster, path: Path) -> None:
    """Save LightGBM model to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> lgb.Booster:
    """Load LightGBM model from file."""
    model = lgb.Booster(model_file=str(path))
    logger.info(f"Model loaded from {path}")
    return model
