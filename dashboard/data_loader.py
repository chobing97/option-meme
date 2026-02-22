"""Cached data loading layer for the Streamlit dashboard."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, LABELED_DIR, LABELED_MANUAL_DIR, PREDICTIONS_DIR, PROCESSED_DIR, RAW_DIR

import logging

logger = logging.getLogger(__name__)


# ── Stock info ───────────────────────────────────────────


@st.cache_data(show_spinner=False)
def get_stock_name_map(market: str) -> dict[str, str]:
    """Return {symbol: name} mapping from stock_info.db."""
    from src.collector.stock_info_db import StockInfoDB

    with StockInfoDB() as db:
        rows = db.get_all(market)
    return {r["symbol"]: r["name"] for r in rows if r.get("name")}


# ── Path helpers ──────────────────────────────────────────


FEATURED_DIR = PROCESSED_DIR / "featured"
MODELS_DIR = DATA_DIR / "models"


def _labeled_path(market: str, label_config: str) -> Path:
    return LABELED_DIR / label_config / f"{market}_labeled.parquet"


def _featured_path(market: str, label_config: str, model_config: str) -> Path:
    return FEATURED_DIR / label_config / model_config / f"{market}_featured.parquet"


def _model_path(market: str, label_config: str, model_config: str, mtype: str, target: str) -> Path:
    ext = "txt" if mtype == "lgb" else "pt"
    return MODELS_DIR / label_config / model_config / f"{mtype}_{market}_{target}.{ext}"


def _splits_path(market: str, label_config: str, model_config: str, split: str) -> Path:
    return MODELS_DIR / label_config / model_config / "splits" / f"{market}_{split}.parquet"


def _predicted_path(market: str, label_config: str, model_config: str) -> Path:
    return PREDICTIONS_DIR / label_config / model_config / f"{market}_predicted.parquet"


def _raw_symbols(market: str) -> list[str]:
    """List available symbols in raw data directory."""
    market_dir = RAW_DIR / market
    if not market_dir.exists():
        return []
    return sorted(d.name for d in market_dir.iterdir() if d.is_dir())


# ── Raw OHLCV ─────────────────────────────────────────────


@st.cache_data(show_spinner="Loading raw bars...")
def load_raw_bars(market: str, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    from src.collector.storage import load_bars
    return load_bars(market, symbol, start_date, end_date)


@st.cache_data(show_spinner="Scanning symbols...")
def get_raw_symbols(market: str) -> list[str]:
    return _raw_symbols(market)


@st.cache_data(show_spinner="Counting raw data...")
def get_raw_summary(market: str) -> dict:
    """Summarise raw data for a market: symbol count, total bars, date range."""
    symbols = _raw_symbols(market)
    if not symbols:
        return {"exists": False}
    total_bars = 0
    min_dt, max_dt = None, None
    for sym in symbols:
        sym_dir = RAW_DIR / market / sym
        for pf in sym_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(pf, columns=["datetime"])
                total_bars += len(df)
                dt_min, dt_max = df["datetime"].min(), df["datetime"].max()
                if min_dt is None or dt_min < min_dt:
                    min_dt = dt_min
                if max_dt is None or dt_max > max_dt:
                    max_dt = dt_max
            except Exception:
                continue
    return {
        "exists": True,
        "n_symbols": len(symbols),
        "total_bars": total_bars,
        "date_range": (str(min_dt)[:10], str(max_dt)[:10]) if min_dt else None,
    }


# ── Labeled data ──────────────────────────────────────────


def _load_manual_overrides(market: str, label_config: str) -> pd.DataFrame:
    """Load manual label overrides (empty DataFrame if none exist)."""
    path = LABELED_MANUAL_DIR / label_config / f"{market}_manual.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "datetime", "label"])
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading labeled data...")
def load_labeled(market: str, label_config: str) -> pd.DataFrame:
    """Load auto-generated labels, then overlay manual overrides."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        return df

    manual = _load_manual_overrides(market, label_config)
    if manual.empty:
        return df

    # Align datetime dtype for merge
    manual["datetime"] = pd.to_datetime(manual["datetime"])
    key = ["symbol", "datetime"]
    merged = df.merge(manual[key + ["label"]], on=key, how="left", suffixes=("", "_manual"))
    has_override = merged["label_manual"].notna()
    merged.loc[has_override, "label"] = merged.loc[has_override, "label_manual"].astype(int)
    merged.drop(columns=["label_manual"], inplace=True)
    return merged


def save_label_edit(market: str, symbol: str, datetime_str: str, new_label: int, label_config: str) -> None:
    """Upsert a manual label override and clear cache."""
    manual = _load_manual_overrides(market, label_config)
    dt = pd.to_datetime(datetime_str)

    mask = (manual["symbol"] == symbol) & (manual["datetime"] == dt)
    if new_label == 0:
        # label=0 means "remove override" — delete the row if it exists
        manual = manual[~mask]
    elif mask.any():
        manual.loc[mask, "label"] = new_label
    else:
        manual = pd.concat(
            [manual, pd.DataFrame([{"symbol": symbol, "datetime": dt, "label": new_label}])],
            ignore_index=True,
        )

    out_dir = LABELED_MANUAL_DIR / label_config
    out_dir.mkdir(parents=True, exist_ok=True)
    manual.to_parquet(out_dir / f"{market}_manual.parquet", index=False)
    st.cache_data.clear()
    logger.info("Manual override saved: %s %s %s -> %d", label_config, symbol, datetime_str, new_label)


@st.cache_data(show_spinner="Summarising labels...")
def get_labeled_summary(market: str, label_config: str) -> dict:
    path = _labeled_path(market, label_config)
    if not path.exists():
        return {"exists": False}
    df = pd.read_parquet(path, columns=["label", "symbol", "datetime"])
    return {
        "exists": True,
        "total_bars": len(df),
        "n_symbols": df["symbol"].nunique(),
        "label_counts": df["label"].value_counts().to_dict(),
        "date_range": (str(df["datetime"].min())[:10], str(df["datetime"].max())[:10]),
    }


# ── Prediction data ───────────────────────────────────────


@st.cache_data(show_spinner="Loading predictions...")
def load_predicted(market: str, label_config: str, model_config: str) -> pd.DataFrame:
    """Load predicted labels from PREDICTIONS_DIR."""
    path = _predicted_path(market, label_config, model_config)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Split info ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_split_dates(market: str, label_config: str, model_config: str) -> dict[str, set[str]]:
    """Return {split_name: set_of_date_strings} for train/val/test."""
    result: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = _splits_path(market, label_config, model_config, split)
        if path.exists():
            dates = pd.read_parquet(path, columns=["date"])["date"].astype(str).unique()
            result[split] = set(dates)
    return result


# ── Featured data ─────────────────────────────────────────


@st.cache_data(show_spinner="Loading feature columns...")
def get_feature_column_list(market: str, label_config: str, model_config: str) -> list[str]:
    """Read only the column names from featured parquet (fast)."""
    path = _featured_path(market, label_config, model_config)
    if not path.exists():
        return []
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    from src.features.feature_pipeline import FEATURE_PREFIXES
    return sorted(
        col for col in schema.names
        if any(col.startswith(p) for p in FEATURE_PREFIXES)
    )


@st.cache_data(show_spinner="Loading features...")
def load_featured(market: str, label_config: str, model_config: str, columns: list[str] | None = None, max_rows: int = 50_000) -> pd.DataFrame:
    """Load featured parquet with optional column filtering and row sampling."""
    path = _featured_path(market, label_config, model_config)
    if not path.exists():
        return pd.DataFrame()
    read_cols = None
    if columns:
        meta_cols = ["datetime", "symbol", "date", "label", "minutes_from_open"]
        read_cols = list(set(meta_cols + columns))
    df = pd.read_parquet(path, columns=read_cols)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).sort_values("datetime").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Summarising features...")
def get_featured_summary(market: str, label_config: str, model_config: str) -> dict:
    path = _featured_path(market, label_config, model_config)
    if not path.exists():
        return {"exists": False}
    import pyarrow.parquet as pq
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    from src.features.feature_pipeline import FEATURE_PREFIXES
    feat_cols = [c for c in schema.names if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    return {
        "exists": True,
        "total_rows": meta.num_rows,
        "n_features": len(feat_cols),
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 1),
    }


# ── Model evaluation ─────────────────────────────────────


@st.cache_data(show_spinner="Evaluating model...", ttl=3600)
def run_model_evaluation(market: str, label_config: str, model_config: str) -> dict | None:
    """Load LightGBM models and run full_evaluation on the test split."""
    featured_path = _featured_path(market, label_config, model_config)
    peak_model_path = _model_path(market, label_config, model_config, "lgb", "peak")
    trough_model_path = _model_path(market, label_config, model_config, "lgb", "trough")

    if not all(p.exists() for p in [featured_path, peak_model_path, trough_model_path]):
        return None

    import numpy as np
    from src.features.feature_pipeline import get_all_feature_columns
    from src.model.dataset import prepare_xy, time_based_split
    from src.model.evaluate import full_evaluation
    from src.model.train_gbm import load_model

    df = pd.read_parquet(featured_path)
    feature_cols = get_all_feature_columns(df)
    split = time_based_split(df)

    peak_model = load_model(peak_model_path)
    trough_model = load_model(trough_model_path)

    X_test, _ = prepare_xy(split.test, target_label=1, feature_cols=feature_cols)
    peak_proba = peak_model.predict(X_test)
    trough_proba = trough_model.predict(X_test)

    eval_result = full_evaluation(split.test, peak_proba, trough_proba)
    eval_result["split_info"] = {
        "train_dates": split.train_dates,
        "val_dates": split.val_dates,
        "test_dates": split.test_dates,
        "train_size": len(split.train),
        "val_size": len(split.val),
        "test_size": len(split.test),
    }
    return eval_result


@st.cache_data(show_spinner="Computing PR curve data...", ttl=3600)
def get_pr_curve_data(market: str, label_config: str, model_config: str, target_label: int) -> dict | None:
    """Get precision-recall curve arrays for plotting."""
    featured_path = _featured_path(market, label_config, model_config)
    label_name = "peak" if target_label == 1 else "trough"
    model_path = _model_path(market, label_config, model_config, "lgb", label_name)

    if not featured_path.exists() or not model_path.exists():
        return None

    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from src.features.feature_pipeline import get_all_feature_columns
    from src.model.dataset import prepare_xy, time_based_split
    from src.model.train_gbm import load_model

    df = pd.read_parquet(featured_path)
    feature_cols = get_all_feature_columns(df)
    split = time_based_split(df)

    model = load_model(model_path)
    X_test, y_test = prepare_xy(split.test, target_label=target_label, feature_cols=feature_cols)
    y_proba = model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


@st.cache_data(show_spinner="Loading feature importance...", ttl=3600)
def get_feature_importance(market: str, label_config: str, model_config: str, target_label: int, top_n: int = 20) -> pd.DataFrame:
    """Extract feature importance from LightGBM model."""
    label_name = "peak" if target_label == 1 else "trough"
    model_path = _model_path(market, label_config, model_config, "lgb", label_name)
    featured_path = _featured_path(market, label_config, model_config)

    if not model_path.exists() or not featured_path.exists():
        return pd.DataFrame()

    import lightgbm as lgb
    from src.features.feature_pipeline import get_all_feature_columns

    model = lgb.Booster(model_file=str(model_path))

    import pyarrow.parquet as pq
    schema = pq.read_schema(featured_path)
    df_tmp = pd.DataFrame(columns=schema.names)
    feature_cols = get_all_feature_columns(df_tmp)

    importance = model.feature_importance(importance_type="gain")
    if len(feature_cols) != len(importance):
        feature_cols = [f"f_{i}" for i in range(len(importance))]

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return imp_df


# ── Model file checks ────────────────────────────────────


def get_model_status(market: str, label_config: str, model_config: str) -> dict:
    """Check existence of model files for a market."""
    models = {}
    for mtype in ["lgb", "lstm"]:
        for target in ["peak", "trough"]:
            path = _model_path(market, label_config, model_config, mtype, target)
            models[f"{mtype}_{target}"] = path.exists()
    return models
