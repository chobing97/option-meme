"""Time-based dataset splitting and PyTorch dataset utilities."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

from config.settings import (
    LOOKBACK_WINDOW,
    RANDOM_SEED,
    TEST_MONTHS,
    TRAIN_YEARS,
    VAL_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
)
from src.features.feature_pipeline import get_all_feature_columns


@dataclass
class SplitResult:
    """Result of time-based train/val/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]


def time_based_split(
    df: pd.DataFrame,
    train_years: int = TRAIN_YEARS,
    val_months: int = VAL_MONTHS,
    test_months: int = TEST_MONTHS,
) -> SplitResult:
    """Split data by time (no random shuffle) to prevent future leakage.

    Args:
        df: Labeled + featured DataFrame with 'datetime' column.
        train_years: Years of data for training.
        val_months: Months for validation.
        test_months: Months for test.

    Returns:
        SplitResult with train/val/test DataFrames and date ranges.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    max_date = df["datetime"].max()
    test_start = max_date - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train = df[df["datetime"] < val_start]
    val = df[(df["datetime"] >= val_start) & (df["datetime"] < test_start)]
    test = df[df["datetime"] >= test_start]

    result = SplitResult(
        train=train,
        val=val,
        test=test,
        train_dates=(str(train["datetime"].min()), str(train["datetime"].max())),
        val_dates=(str(val["datetime"].min()), str(val["datetime"].max())),
        test_dates=(str(test["datetime"].min()), str(test["datetime"].max())),
    )

    logger.info(
        f"Split: train={len(train)} ({result.train_dates[0][:10]}~{result.train_dates[1][:10]}), "
        f"val={len(val)} ({result.val_dates[0][:10]}~{result.val_dates[1][:10]}), "
        f"test={len(test)} ({result.test_dates[0][:10]}~{result.test_dates[1][:10]})"
    )

    return result


def walk_forward_splits(
    df: pd.DataFrame,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward validation splits.

    Sliding window: train on N months, test on next M months, slide forward.

    Args:
        df: Full DataFrame sorted by datetime.
        train_months: Training window size in months.
        test_months: Test window size in months.

    Returns:
        List of (train_df, test_df) tuples.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    min_date = df["datetime"].min()
    max_date = df["datetime"].max()

    splits = []
    current_train_start = min_date

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > max_date + pd.Timedelta(days=1):
            break

        train = df[(df["datetime"] >= current_train_start) & (df["datetime"] < train_end)]
        test = df[(df["datetime"] >= train_end) & (df["datetime"] < test_end)]

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

        current_train_start += pd.DateOffset(months=test_months)

    logger.info(f"Walk-forward: {len(splits)} splits generated")
    return splits


def prepare_xy(
    df: pd.DataFrame,
    target_label: int,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and binary target y from DataFrame.

    Args:
        df: Featured DataFrame with 'label' column.
        target_label: Which label to predict (1=peak, 2=trough).
        feature_cols: Specific feature columns to use. If None, auto-detect.

    Returns:
        Tuple of (X, y) numpy arrays.
    """
    if feature_cols is None:
        feature_cols = get_all_feature_columns(df)

    X = df[feature_cols].values.astype(np.float32)
    y = (df["label"] == target_label).astype(int).values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(
        f"Prepared X: {X.shape}, y: {y.shape}, "
        f"positive rate: {y.mean():.4f}"
    )

    return X, y


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for LSTM/Transformer with lookback window."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_label: int,
        lookback: int = LOOKBACK_WINDOW,
        feature_cols: Optional[list[str]] = None,
    ):
        """
        Args:
            df: Featured DataFrame sorted by datetime, with 'label' column.
            target_label: Which label to predict (1=peak, 2=trough).
            lookback: Number of past bars as input sequence.
            feature_cols: Feature columns to use.
        """
        if feature_cols is None:
            feature_cols = get_all_feature_columns(df)

        self.feature_cols = feature_cols
        self.lookback = lookback
        self.target_label = target_label

        # Build sequences per day to avoid cross-day leakage
        self.sequences = []
        self.targets = []

        if "date" in df.columns:
            for _, day_df in df.groupby("date"):
                self._build_sequences(day_df)
        else:
            self._build_sequences(df)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Replace NaN/inf
        self.sequences = np.nan_to_num(self.sequences, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_sequences(self, df: pd.DataFrame) -> None:
        """Build lookback sequences from a contiguous block of data."""
        features = df[self.feature_cols].values.astype(np.float32)
        labels = (df["label"] == self.target_label).astype(int).values

        for i in range(self.lookback, len(features)):
            seq = features[i - self.lookback : i]  # [lookback, n_features]
            target = labels[i]
            self.sequences.append(seq)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

    @property
    def positive_rate(self) -> float:
        return float(self.targets.mean()) if len(self.targets) > 0 else 0.0

    @property
    def n_features(self) -> int:
        return self.sequences.shape[-1] if len(self.sequences) > 0 else 0
