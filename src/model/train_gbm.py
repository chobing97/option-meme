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
            "feature_pre_filter": False,
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

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Optuna best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Retrain with best params on full train+val
    best_params = {**LGB_PARAMS, **study.best_params}
    best_model, best_metrics = train_lgb(
        split, target_label, feature_cols, params=best_params,
    )

    return best_model, best_metrics, study.best_params


def train_lgb_incremental(
    chunks: list,
    target_label: int,
    feature_cols: list[str],
    split_dates: dict,
    params: Optional[dict] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM incrementally across memory-safe chunks.

    Each chunk is loaded, split by date into train/val, and used to continue
    training via ``init_model``.  The validation set from each chunk is used
    for early stopping in that round.  Final evaluation is done by loading
    and predicting on each chunk's test portion.

    Args:
        chunks: List of chunks from ``build_incremental_chunks``.
                Each chunk is a list of partition info dicts with 'path' key.
        target_label: 1 for peak, 2 for trough.
        feature_cols: Feature column names.
        split_dates: Dict with 'val_start' and 'test_start' date strings
                     for time-based splitting.
        params: LightGBM params.
        num_boost_round: Total boosting rounds (distributed across chunks).
        early_stopping_rounds: Patience per chunk.

    Returns:
        Tuple of (trained Booster, evaluation metrics dict).
    """
    import pandas as pd
    from src.features.feature_pipeline import load_chunk

    if params is None:
        params = LGB_PARAMS.copy()

    val_start = pd.Timestamp(split_dates["val_start"])
    test_start = pd.Timestamp(split_dates["test_start"])
    rounds_per_chunk = max(10, num_boost_round // len(chunks))

    model = None
    total_train = 0
    total_val = 0

    logger.info(
        f"Incremental training: {len(chunks)} chunks, "
        f"{rounds_per_chunk} rounds/chunk, label={target_label}"
    )

    for i, chunk in enumerate(chunks):
        df = load_chunk(chunk)
        df["datetime"] = pd.to_datetime(df["datetime"])

        train_df = df[df["datetime"] < val_start]
        val_df = df[(df["datetime"] >= val_start) & (df["datetime"] < test_start)]

        if train_df.empty:
            logger.warning(f"  Chunk {i}: no train data, skipping")
            del df, train_df, val_df
            continue

        X_train, y_train = prepare_xy(train_df, target_label, feature_cols)
        total_train += len(y_train)

        train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

        valid_sets = [train_ds]
        valid_names = ["train"]

        if not val_df.empty:
            X_val, y_val = prepare_xy(val_df, target_label, feature_cols)
            total_val += len(y_val)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, free_raw_data=True)
            valid_sets.append(val_ds)
            valid_names.append("val")

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=rounds_per_chunk,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
            init_model=model,
        )

        chunk_rows = len(train_df) + len(val_df)
        logger.info(
            f"  Chunk {i+1}/{len(chunks)}: train={len(y_train)}, "
            f"val={len(val_df)}, trees={model.num_trees()}"
        )

        del df, train_df, val_df, X_train, y_train, train_ds
        if "X_val" in dir():
            del X_val, y_val, val_ds

    if model is None:
        raise ValueError("No training data found in any chunk")

    # Final evaluation: iterate chunks and collect test predictions
    all_y_test = []
    all_y_pred = []
    for chunk in chunks:
        df = load_chunk(chunk)
        df["datetime"] = pd.to_datetime(df["datetime"])
        test_df = df[df["datetime"] >= test_start]
        if test_df.empty:
            del df, test_df
            continue

        X_test, y_test = prepare_xy(test_df, target_label, feature_cols)
        y_pred = model.predict(X_test)
        all_y_test.append(y_test)
        all_y_pred.append(y_pred)
        del df, test_df, X_test

    if all_y_test:
        y_test_all = np.concatenate(all_y_test)
        y_pred_all = np.concatenate(all_y_pred)
        pr_auc = average_precision_score(y_test_all, y_pred_all)
    else:
        pr_auc = 0.0
        y_test_all = np.array([])

    # Feature importance
    importance = dict(
        zip(feature_cols, model.feature_importance(importance_type="gain"))
    )
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    metrics = {
        "target_label": target_label,
        "pr_auc_test": pr_auc,
        "best_iteration": model.num_trees(),
        "train_size": total_train,
        "val_size": total_val,
        "test_size": len(y_test_all),
        "positive_rate_test": float(y_test_all.mean()) if len(y_test_all) > 0 else 0.0,
        "n_chunks": len(chunks),
        "top_features": top_features,
    }

    logger.info(f"Incremental LightGBM test PR-AUC: {pr_auc:.4f} ({len(chunks)} chunks)")

    return model, metrics


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
