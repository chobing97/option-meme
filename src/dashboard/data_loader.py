"""Cached data loading layer for the Streamlit dashboard."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, LABELED_DIR, LABELED_MANUAL_DIR, PREDICTIONS_DIR, PROCESSED_DIR, RAW_OPTIONS_DIR, RAW_STOCK_DIR

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


def _predicted_path(market: str, label_config: str, model_config: str, model_type: str = "gbm") -> Path:
    return PREDICTIONS_DIR / model_type / label_config / model_config / f"{market}_predicted.parquet"


def _raw_symbols(market: str) -> list[str]:
    """List available symbols in raw data directory."""
    market_dir = RAW_STOCK_DIR / market
    if not market_dir.exists():
        return []
    return sorted(d.name for d in market_dir.iterdir() if d.is_dir())


# ── Raw OHLCV ─────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def get_raw_date_range(market: str, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Get min/max datetime for a symbol by reading only the first and last parquet files."""
    from src.collector.storage import RAW_STOCK_DIR as _RAW

    symbol_dir = _RAW / market / symbol
    if not symbol_dir.exists() and market == "kr":
        stripped = symbol.lstrip("0")
        if stripped != symbol:
            symbol_dir = _RAW / market / stripped
    if not symbol_dir.exists():
        return None

    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    first = pd.read_parquet(parquet_files[0], columns=["datetime"])
    last = pd.read_parquet(parquet_files[-1], columns=["datetime"]) if len(parquet_files) > 1 else first
    return first["datetime"].min(), last["datetime"].max()


@st.cache_data(show_spinner="Loading trading dates...")
def get_raw_trading_dates(market: str, symbol: str) -> list:
    """Get unique trading dates for a symbol by reading only the datetime column."""
    from src.collector.storage import RAW_STOCK_DIR as _RAW

    symbol_dir = _RAW / market / symbol
    if not symbol_dir.exists() and market == "kr":
        stripped = symbol.lstrip("0")
        if stripped != symbol:
            symbol_dir = _RAW / market / stripped
    if not symbol_dir.exists():
        return []

    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    dates = set()
    for pf in parquet_files:
        dt_col = pd.read_parquet(pf, columns=["datetime"])
        dates.update(dt_col["datetime"].dt.date.unique())
    return sorted(dates)


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
        sym_dir = RAW_STOCK_DIR / market / sym
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


# ── Options OHLCV ────────────────────────────────────────


@st.cache_data(show_spinner=False)
def has_options_data(market: str, symbol: str) -> bool:
    """Check if options OHLCV data exists for a symbol."""
    sym_dir = RAW_OPTIONS_DIR / market / symbol
    return sym_dir.exists() and any(sym_dir.glob("*.parquet"))


@st.cache_data(show_spinner="Loading options data...")
def load_options_ohlcv(market: str, symbol: str, date_str: str) -> pd.DataFrame:
    """Load options OHLCV for a symbol on a specific date.

    Returns DataFrame with ATM contract's minute bars for that day.
    Selects the contract whose strike is closest to the stock close price.
    """
    sym_dir = RAW_OPTIONS_DIR / market / symbol
    if not sym_dir.exists():
        return pd.DataFrame()

    contracts_path = sym_dir / "contracts.parquet"
    if not contracts_path.exists():
        return pd.DataFrame()

    contracts = pd.read_parquet(contracts_path)
    contracts["period_start"] = pd.to_datetime(contracts["period_start"])
    contracts["expiry"] = pd.to_datetime(contracts["expiry"])

    target_date = pd.Timestamp(date_str)

    # Find active contract: period_start <= date < expiry
    active = contracts[
        (contracts["period_start"] <= target_date)
        & (contracts["expiry"] > target_date)
    ]
    if active.empty:
        return pd.DataFrame()

    # Pick ATM put (closest strike to stock_close)
    puts = active[active["cp"] == "P"]
    if puts.empty:
        puts = active  # fallback to any type
    atm = puts.iloc[(puts["strike"] - puts["stock_close"]).abs().argsort()[:1]]
    contract_symbol = atm["symbol"].iloc[0].strip()

    # Load only the relevant year's OHLCV file with row-level filter
    year_file = sym_dir / f"{target_date.year}.parquet"
    if not year_file.exists():
        return pd.DataFrame()

    day_start = pd.Timestamp(target_date.date())
    day_end = day_start + pd.Timedelta(days=1)
    ohlcv = pd.read_parquet(
        year_file,
        filters=[
            ("symbol", "==", atm["symbol"].iloc[0]),  # exact match (padded)
            ("datetime", ">=", day_start),
            ("datetime", "<", day_end),
        ],
    )
    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    result = ohlcv.sort_values("datetime").reset_index(drop=True)

    if not result.empty:
        strike = float(atm["strike"].iloc[0])
        expiry = atm["expiry"].iloc[0].strftime("%Y-%m-%d")
        result.attrs["contract_info"] = f"Put K={strike:.0f} Exp={expiry}"

    return result


@st.cache_data(show_spinner=False)
def load_options_ohlcv_by_strike(
    market: str, symbol: str, date_str: str, strike: float, cp: str = "P",
) -> pd.DataFrame:
    """Load options OHLCV for a specific strike on a given date.

    Finds the contract matching the strike + cp (put/call) that is active
    on the target date and returns its minute bars.
    """
    sym_dir = RAW_OPTIONS_DIR / market / symbol
    if not sym_dir.exists():
        return pd.DataFrame()

    contracts_path = sym_dir / "contracts.parquet"
    if not contracts_path.exists():
        return pd.DataFrame()

    contracts = pd.read_parquet(contracts_path)
    contracts["period_start"] = pd.to_datetime(contracts["period_start"])
    contracts["expiry"] = pd.to_datetime(contracts["expiry"])

    target_date = pd.Timestamp(date_str)

    # Find active contract matching strike and cp
    match = contracts[
        (contracts["strike"] == strike)
        & (contracts["cp"] == cp)
        & (contracts["period_start"] <= target_date)
        & (contracts["expiry"] > target_date)
    ]
    if match.empty:
        return pd.DataFrame()

    contract_row = match.iloc[0]

    year_file = sym_dir / f"{target_date.year}.parquet"
    if not year_file.exists():
        return pd.DataFrame()

    day_start = pd.Timestamp(target_date.date())
    day_end = day_start + pd.Timedelta(days=1)
    ohlcv = pd.read_parquet(
        year_file,
        filters=[
            ("symbol", "==", contract_row["symbol"]),
            ("datetime", ">=", day_start),
            ("datetime", "<", day_end),
        ],
    )
    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
    result = ohlcv.sort_values("datetime").reset_index(drop=True)

    if not result.empty:
        expiry = contract_row["expiry"].strftime("%Y-%m-%d")
        result.attrs["contract_info"] = f"Put K={strike:.0f} Exp={expiry}"

    return result


# ── Labeled data ──────────────────────────────────────────


def _load_manual_overrides(market: str, label_config: str) -> pd.DataFrame:
    """Load manual label overrides (empty DataFrame if none exist)."""
    path = LABELED_MANUAL_DIR / label_config / f"{market}_manual.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "datetime", "label"])
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_labeled_symbols(market: str, label_config: str) -> list[str]:
    """Return sorted symbol list from labeled parquet (lightweight — symbol column only)."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["symbol"])
    return sorted(df["symbol"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_labeled_date_range(market: str, label_config: str, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (min, max) datetime for a symbol in labeled data."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["symbol", "datetime"], filters=[("symbol", "==", symbol)])
    if df.empty:
        return None
    return df["datetime"].min(), df["datetime"].max()


@st.cache_data(show_spinner=False)
def get_labeled_trading_dates(market: str, label_config: str, symbol: str) -> list:
    """Return sorted list of trading dates for a symbol."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["symbol", "date"], filters=[("symbol", "==", symbol)])
    if df.empty:
        return []
    return sorted(df["date"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_labeled_symbol_stats(market: str, label_config: str, symbol: str) -> dict:
    """Return label counts for a symbol (lightweight — label column only)."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["symbol", "label"], filters=[("symbol", "==", symbol)])
    return df["label"].value_counts().to_dict()


@st.cache_data(show_spinner="Loading labeled data...")
def load_labeled(market: str, label_config: str, symbol: str | None = None, date_str: str | None = None) -> pd.DataFrame:
    """Load labeled data with optional symbol/date filtering via pyarrow pushdown."""
    path = _labeled_path(market, label_config)
    if not path.exists():
        return pd.DataFrame()

    filters = []
    if symbol is not None:
        filters.append(("symbol", "==", symbol))
    if date_str is not None:
        import datetime
        filters.append(("date", "==", datetime.date.fromisoformat(date_str)))

    df = pd.read_parquet(path, filters=filters if filters else None)
    if df.empty:
        return df

    manual = _load_manual_overrides(market, label_config)
    if manual.empty:
        return df

    # Filter manual overrides to match
    manual["datetime"] = pd.to_datetime(manual["datetime"])
    if symbol is not None:
        manual = manual[manual["symbol"] == symbol]
    if manual.empty:
        return df

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
def load_predicted(market: str, label_config: str, model_config: str, model_type: str = "gbm") -> pd.DataFrame:
    """Load predicted labels from PREDICTIONS_DIR."""
    path = _predicted_path(market, label_config, model_config, model_type)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_available_model_types(market: str, label_config: str, model_config: str) -> list[str]:
    """Scan PREDICTIONS_DIR for available model_type subdirectories that have data."""
    available = []
    for mt in ("gbm", "lstm", "ensemble"):
        path = _predicted_path(market, label_config, model_config, mt)
        if path.exists():
            available.append(mt)
    return available


@st.cache_data(show_spinner=False)
def find_configs_for_model_type(market: str, model_type: str) -> list[tuple[str, str]]:
    """Find all (label_config, model_config) pairs that have prediction data for a model_type."""
    from config.variants import LABEL_CONFIGS, MODEL_CONFIGS
    results = []
    for lc in sorted(LABEL_CONFIGS.keys()):
        for mc in sorted(MODEL_CONFIGS.keys()):
            if _predicted_path(market, lc, mc, model_type).exists():
                results.append((lc, mc))
    return results


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


# ── Backtest data ────────────────────────────────────────


BACKTEST_DIR = DATA_DIR / "trading" / "backtests"


@st.cache_data(show_spinner=False)
def get_backtest_files() -> list[str]:
    """List available backtest parquet files (stem names)."""
    if not BACKTEST_DIR.exists():
        return []
    return sorted(
        (p.stem for p in BACKTEST_DIR.glob("*.parquet")),
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def get_backtest_symbols(name: str) -> list[str]:
    """Return sorted symbol list from a backtest file."""
    path = BACKTEST_DIR / f"{name}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["symbol"])
    return sorted(df["symbol"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_backtest_trading_dates(name: str, symbol: str) -> list:
    """Return sorted list of trading dates for a symbol in a backtest."""
    path = BACKTEST_DIR / f"{name}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["timestamp", "symbol"])
    df = df[df["symbol"] == symbol]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return sorted(df["timestamp"].dt.date.unique().tolist())


@st.cache_data(show_spinner="Loading backtest data...")
def load_backtest(name: str, symbol: str | None = None, date_str: str | None = None) -> pd.DataFrame:
    """Load a backtest parquet file with optional symbol/date filtering."""
    path = BACKTEST_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if symbol is not None:
        df = df[df["symbol"] == symbol]
    if date_str is not None:
        import datetime as _dt
        target = _dt.date.fromisoformat(date_str)
        df = df[df["timestamp"].dt.date == target]
    return df


# ── Feature data ─────────────────────────────────────────


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
