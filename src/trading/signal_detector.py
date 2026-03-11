"""Signal detector: bar accumulation, feature pipeline, and model inference."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DATA_DIR, LOOKBACK_WINDOW
from src.trading.broker.base import Signal, SignalType


class BarAccumulator:
    """Manages historical bars + today's accumulating bars for feature computation."""

    def __init__(self, history_df: pd.DataFrame):
        """Initialize with multi-day history DataFrame.

        Args:
            history_df: Early-session DataFrame with OHLCV, 'date', 'minutes_from_open'.
        """
        self._history = history_df.copy()
        self._today_bars: list[pd.Series] = []

    def add_bar(self, bar: pd.Series) -> None:
        self._today_bars.append(bar)

    def get_feature_df(self) -> pd.DataFrame:
        """Return history + today's bars combined as a single DataFrame."""
        if not self._today_bars:
            return self._history.copy()

        today_df = pd.DataFrame(self._today_bars)
        today_df = today_df.reset_index(drop=True)

        combined = pd.concat([self._history, today_df], ignore_index=True)
        return combined

    @property
    def bar_count(self) -> int:
        return len(self._today_bars)


class SignalDetector:
    """Wraps feature pipeline + model inference for real-time signal detection."""

    def __init__(
        self,
        market: str,
        model_type: str = "gbm",
        threshold: float = 0.5,
        label_config: str = "L2",
        model_config: str = "M3",
    ):
        self.market = market
        self.model_type = model_type
        self.threshold = threshold
        self.label_config = label_config
        self.model_config = model_config
        self._peak_model = None
        self._trough_model = None
        self._feature_cols: Optional[list[str]] = None

    def _ensure_models(self) -> None:
        """Lazy-load models on first call."""
        if self._peak_model is not None:
            return

        models_dir = DATA_DIR / "models" / self.label_config / self.model_config

        if self.model_type == "gbm":
            from src.model.train_gbm import load_model

            peak_path = models_dir / f"lgb_{self.market}_peak.txt"
            trough_path = models_dir / f"lgb_{self.market}_trough.txt"

            if not peak_path.exists() or not trough_path.exists():
                raise FileNotFoundError(
                    f"Model files not found at {models_dir}. "
                    f"Train first: ./optionmeme model --market {self.market} --model gbm "
                    f"--label-config {self.label_config} --model-config {self.model_config}"
                )

            self._peak_model = load_model(peak_path)
            self._trough_model = load_model(trough_path)
        else:
            raise ValueError(f"Unsupported model_type for trading: {self.model_type}")

    def detect(self, accumulator: BarAccumulator) -> Signal:
        """Build features from accumulated bars and run model inference.

        Returns Signal for the latest bar.
        """
        from src.features.feature_pipeline import (
            build_features,
            build_lookback_features,
            clean_features,
            get_all_feature_columns,
        )

        self._ensure_models()

        # Build feature DataFrame from history + today
        df = accumulator.get_feature_df()

        if "label" not in df.columns:
            df["label"] = 0

        df = build_features(df)
        df = build_lookback_features(df)
        df = clean_features(df)

        feature_cols = get_all_feature_columns(df)
        if not feature_cols:
            return self._no_signal(accumulator)

        self._feature_cols = feature_cols

        if df.empty:
            return self._no_signal(accumulator)

        # Take only the last row (current bar)
        last_row = df.iloc[[-1]]
        X = last_row[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        peak_prob = float(self._peak_model.predict(X)[0])
        trough_prob = float(self._trough_model.predict(X)[0])

        # Determine signal
        signal_type = SignalType.NONE
        if peak_prob >= self.threshold and peak_prob > trough_prob:
            signal_type = SignalType.PEAK
        elif trough_prob >= self.threshold and trough_prob > peak_prob:
            signal_type = SignalType.TROUGH
        elif peak_prob >= self.threshold:
            signal_type = SignalType.PEAK
        elif trough_prob >= self.threshold:
            signal_type = SignalType.TROUGH

        # Get timestamp and close from the last bar in the accumulator
        latest_bar = accumulator._today_bars[-1] if accumulator._today_bars else None
        ts = latest_bar["datetime"] if latest_bar is not None else datetime.now()
        close = float(latest_bar["close"]) if latest_bar is not None else 0.0

        return Signal(
            signal_type=signal_type,
            timestamp=ts,
            close_price=close,
            peak_prob=peak_prob,
            trough_prob=trough_prob,
        )

    @staticmethod
    def _no_signal(accumulator: BarAccumulator) -> Signal:
        latest = accumulator._today_bars[-1] if accumulator._today_bars else None
        return Signal(
            signal_type=SignalType.NONE,
            timestamp=latest["datetime"] if latest is not None else datetime.now(),
            close_price=float(latest["close"]) if latest is not None else 0.0,
        )
