"""Inference pipeline: load latest data → features → model prediction."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import DATA_DIR, LOOKBACK_WINDOW, PREDICTIONS_DIR, PROCESSED_DIR


def predict_all(
    market: str,
    model_type: str = "gbm",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Batch predict all symbols/dates from featured parquet.

    Loads the pre-computed featured data, runs model prediction on all rows
    at once, and saves results in labeled parquet format with probabilities.

    Args:
        market: 'kr' or 'us'
        model_type: 'gbm' (only gbm supported for batch)
        threshold: Probability threshold for peak/trough classification

    Returns:
        DataFrame with predictions in labeled format (includes peak_prob, trough_prob).
    """
    from src.features.feature_pipeline import get_all_feature_columns
    from src.model.train_gbm import load_model

    # 1. Load featured data
    featured_path = PROCESSED_DIR / "featured" / f"{market}_featured.parquet"
    if not featured_path.exists():
        raise FileNotFoundError(
            f"Featured data not found: {featured_path}\n"
            f"Run features first: ./run.sh features --market {market}"
        )

    df = pd.read_parquet(featured_path)
    logger.info(f"Loaded {len(df)} rows from {featured_path}")

    # 2. Get feature columns (same ordering as training)
    feature_cols = get_all_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns found in featured data")
    logger.info(f"Using {len(feature_cols)} feature columns")

    # 3. Prepare feature matrix
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. Load models and predict
    models_dir = DATA_DIR / "models"
    peak_path = models_dir / f"lgb_{market}_peak.txt"
    trough_path = models_dir / f"lgb_{market}_trough.txt"

    if not peak_path.exists() or not trough_path.exists():
        raise FileNotFoundError(
            f"LightGBM models not found at {models_dir}\n"
            f"Train first: ./run.sh model --market {market} --model gbm"
        )

    peak_model = load_model(peak_path)
    trough_model = load_model(trough_path)

    peak_proba = peak_model.predict(X)
    trough_proba = trough_model.predict(X)
    logger.info("Prediction complete")

    # 5. Vectorized label assignment
    peak_proba = np.asarray(peak_proba)
    trough_proba = np.asarray(trough_proba)

    labels = np.zeros(len(df), dtype=np.int8)
    labels[(peak_proba >= threshold) & (peak_proba > trough_proba)] = 1
    labels[(trough_proba >= threshold) & (trough_proba > peak_proba)] = 2

    # 6. Build output DataFrame in labeled format
    output_cols = ["datetime", "open", "high", "low", "close", "volume",
                   "date", "minutes_from_open", "symbol", "market"]
    # Only keep columns that exist in df
    output_cols = [c for c in output_cols if c in df.columns]

    result = df[output_cols].copy()
    result["label"] = labels
    result["peak_prob"] = np.round(peak_proba, 4).astype(np.float32)
    result["trough_prob"] = np.round(trough_proba, 4).astype(np.float32)

    # 7. Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / f"{market}_predicted.parquet"
    result.to_parquet(output_path, index=False, compression="snappy")

    n_peaks = int((labels == 1).sum())
    n_troughs = int((labels == 2).sum())
    logger.info(
        f"Saved {len(result)} rows to {output_path} "
        f"(peaks={n_peaks}, troughs={n_troughs}, neither={len(result) - n_peaks - n_troughs})"
    )

    return result


def predict_symbol(
    market: str,
    symbol: str,
    model_type: str = "gbm",
    threshold: float = 0.5,
    date: Optional[str] = None,
) -> dict:
    """Run prediction pipeline for a single symbol.

    Args:
        market: 'kr' or 'us'
        symbol: Ticker symbol (e.g. '005930', 'AAPL')
        model_type: 'gbm', 'lstm', or 'all'
        threshold: Probability threshold for signal detection.
        date: Target date (YYYY-MM-DD). Latest trading day if None.

    Returns:
        Dict with prediction results, signals, and metadata.
    """
    from src.collector.storage import load_bars
    from src.features.feature_pipeline import (
        build_features,
        build_lookback_features,
        clean_features,
        get_all_feature_columns,
    )
    from src.labeler.session_extractor import extract_early_session

    # 1. Load raw OHLCV
    logger.info(f"Loading bars for {market}/{symbol}...")
    raw_df = load_bars(market, symbol)
    if raw_df.empty:
        raise FileNotFoundError(
            f"No data found for {market}/{symbol}. Run collector first: "
            f"./run.sh collector --market {market} --symbol {symbol}"
        )

    # 2. Extract early session (first 60 min)
    early_df = extract_early_session(raw_df, market)
    if early_df.empty:
        raise ValueError(f"No early session bars for {market}/{symbol}")

    # 3. Filter by date
    early_df["date_str"] = early_df["date"].astype(str)
    available_dates = sorted(early_df["date_str"].unique())

    if date is not None:
        if date not in available_dates:
            raise ValueError(
                f"Date {date} not available. Latest dates: {available_dates[-5:]}"
            )
        target_date = date
    else:
        target_date = available_dates[-1]

    logger.info(f"Target date: {target_date}")

    # We need multiple days for lookback features, so keep recent history
    date_idx = available_dates.index(target_date)
    # Keep enough prior days for feature lookback (at least lookback bars before target)
    start_idx = max(0, date_idx - 5)
    needed_dates = available_dates[start_idx : date_idx + 1]
    multi_day_df = early_df[early_df["date_str"].isin(needed_dates)].copy()

    # 4. Build features
    # Add a dummy 'label' column (required by some feature functions, 0 = no label)
    if "label" not in multi_day_df.columns:
        multi_day_df["label"] = 0

    multi_day_df = build_features(multi_day_df)
    multi_day_df = build_lookback_features(multi_day_df)
    multi_day_df = clean_features(multi_day_df)

    feature_cols = get_all_feature_columns(multi_day_df)
    if not feature_cols:
        raise ValueError("No feature columns generated")

    # Filter to target date only for predictions
    target_df = multi_day_df[multi_day_df["date_str"] == target_date].copy()
    if target_df.empty:
        raise ValueError(
            f"No bars remaining for {target_date} after feature engineering "
            f"(lookback may have dropped them)"
        )

    logger.info(f"Predicting on {len(target_df)} bars with {len(feature_cols)} features")

    # 5. Load models and predict
    models = _load_models(market, model_type, feature_cols, target_df)

    # 6. Run predictions per model type
    results = {
        "market": market,
        "symbol": symbol,
        "date": target_date,
        "threshold": threshold,
        "n_bars": len(target_df),
        "predictions": [],
    }

    for mtype, model_predictions in models.items():
        peak_proba = model_predictions["peak"]
        trough_proba = model_predictions["trough"]

        # Build per-bar results
        bars = []
        for i, (_, row) in enumerate(target_df.iterrows()):
            # Get time string from datetime
            dt = row["datetime"]
            if hasattr(dt, "strftime"):
                time_str = dt.strftime("%H:%M")
            else:
                time_str = str(dt)

            pp = float(peak_proba[i]) if i < len(peak_proba) else 0.0
            tp = float(trough_proba[i]) if i < len(trough_proba) else 0.0

            signal = "none"
            if pp >= threshold and pp > tp:
                signal = "peak"
            elif tp >= threshold and tp > pp:
                signal = "trough"
            elif pp >= threshold:
                signal = "peak"
            elif tp >= threshold:
                signal = "trough"

            bars.append({
                "time": time_str,
                "close": float(row["close"]),
                "peak_prob": round(pp, 4),
                "trough_prob": round(tp, 4),
                "signal": signal,
            })

        n_peaks = sum(1 for b in bars if b["signal"] == "peak")
        n_troughs = sum(1 for b in bars if b["signal"] == "trough")

        results["predictions"].append({
            "model": mtype,
            "bars": bars,
            "n_peaks": n_peaks,
            "n_troughs": n_troughs,
        })

    # 7. Format output
    _format_and_print(results)

    return results


def _load_models(
    market: str,
    model_type: str,
    feature_cols: list[str],
    target_df: pd.DataFrame,
) -> dict:
    """Load models and run predictions.

    Returns:
        Dict mapping model type string to dict with 'peak' and 'trough' probability arrays.
    """
    models_dir = DATA_DIR / "models"
    results = {}

    if model_type in ("gbm", "all"):
        results["LightGBM"] = _predict_gbm(models_dir, market, feature_cols, target_df)

    if model_type in ("lstm", "all"):
        results["LSTM"] = _predict_lstm(models_dir, market, feature_cols, target_df)

    if not results:
        raise ValueError(f"No models loaded for model_type='{model_type}'")

    return results


def _predict_gbm(
    models_dir: Path,
    market: str,
    feature_cols: list[str],
    target_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Load LightGBM models and predict."""
    from src.model.train_gbm import load_model

    peak_path = models_dir / f"lgb_{market}_peak.txt"
    trough_path = models_dir / f"lgb_{market}_trough.txt"

    if not peak_path.exists():
        raise FileNotFoundError(
            f"LightGBM peak model not found: {peak_path}\n"
            f"Train first: ./run.sh model --market {market} --model gbm"
        )
    if not trough_path.exists():
        raise FileNotFoundError(
            f"LightGBM trough model not found: {trough_path}\n"
            f"Train first: ./run.sh model --market {market} --model gbm"
        )

    peak_model = load_model(peak_path)
    trough_model = load_model(trough_path)

    X = target_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    peak_proba = peak_model.predict(X)
    trough_proba = trough_model.predict(X)

    return {"peak": peak_proba, "trough": trough_proba}


def _predict_lstm(
    models_dir: Path,
    market: str,
    feature_cols: list[str],
    target_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Load LSTM models and predict."""
    import torch

    from src.model.dataset import TimeSeriesDataset
    from src.model.train_lstm import load_model, predict

    peak_path = models_dir / f"lstm_{market}_peak.pt"
    trough_path = models_dir / f"lstm_{market}_trough.pt"

    if not peak_path.exists():
        raise FileNotFoundError(
            f"LSTM peak model not found: {peak_path}\n"
            f"Train first: ./run.sh model --market {market} --model lstm"
        )
    if not trough_path.exists():
        raise FileNotFoundError(
            f"LSTM trough model not found: {trough_path}\n"
            f"Train first: ./run.sh model --market {market} --model lstm"
        )

    n_features = len(feature_cols)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Ensure 'label' column exists for TimeSeriesDataset
    df_for_ds = target_df.copy()
    if "label" not in df_for_ds.columns:
        df_for_ds["label"] = 0

    peak_model = load_model(peak_path, n_features=n_features)
    trough_model = load_model(trough_path, n_features=n_features)

    peak_ds = TimeSeriesDataset(df_for_ds, target_label=1, feature_cols=feature_cols)
    trough_ds = TimeSeriesDataset(df_for_ds, target_label=2, feature_cols=feature_cols)

    peak_proba = predict(peak_model, peak_ds, device=device)
    trough_proba = predict(trough_model, trough_ds, device=device)

    # LSTM drops the first `lookback` bars per day, so pad with zeros to align
    n_target = len(target_df)
    if len(peak_proba) < n_target:
        pad_len = n_target - len(peak_proba)
        peak_proba = np.concatenate([np.zeros(pad_len), peak_proba])
    if len(trough_proba) < n_target:
        pad_len = n_target - len(trough_proba)
        trough_proba = np.concatenate([np.zeros(pad_len), trough_proba])

    return {"peak": peak_proba, "trough": trough_proba}


def _format_and_print(results: dict) -> None:
    """Print prediction results as a formatted table and save JSON."""
    market = results["market"]
    symbol = results["symbol"]
    date = results["date"]
    threshold = results["threshold"]

    for pred in results["predictions"]:
        model_name = pred["model"]
        bars = pred["bars"]

        print(f"\n=== Predict: {market}/{symbol} ({date}) ===")
        print(f"Model: {model_name} | Threshold: {threshold}")
        print()
        print(f"  {'Time':<10}{'Close':>10}{'Peak%':>9}{'Trough%':>9}  Signal")
        print(f"  {'─' * 10}{'─' * 10}{'─' * 9}{'─' * 9}  {'─' * 12}")

        signal_bars = [b for b in bars if b["signal"] != "none"]

        if signal_bars:
            for b in signal_bars:
                sig_str = ""
                if b["signal"] == "peak":
                    sig_str = "▲ PEAK"
                elif b["signal"] == "trough":
                    sig_str = "▼ TROUGH"

                print(
                    f"  {b['time']:<10}"
                    f"{b['close']:>10.0f}"
                    f"{b['peak_prob']:>9.2f}"
                    f"{b['trough_prob']:>9.2f}"
                    f"  {sig_str}"
                )
        else:
            print("  (no signals detected at this threshold)")

        print()
        print(
            f"Summary: {pred['n_peaks']} peaks, {pred['n_troughs']} troughs "
            f"detected in {len(bars)} bars"
        )

    # Save JSON
    predictions_dir = DATA_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    json_path = predictions_dir / f"{market}_{symbol}_{date}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to: {json_path}")
