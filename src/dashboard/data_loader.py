"""Cached data loading layer for the Streamlit dashboard."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, LABELED_DIR, LABELED_MANUAL_DIR, PREDICTIONS_DIR, PROCESSED_DIR, RAW_GENERATED_DIR, RAW_OPTIONS_DIR, RAW_STOCK_DIR

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


def _labeled_path(market: str, label_config: str, timeframe: str = "1m") -> Path:
    """Return the labeled directory for partitioned layout, or legacy file path."""
    return LABELED_DIR / timeframe / label_config / market


def _labeled_legacy_path(market: str, label_config: str, timeframe: str = "1m") -> Path:
    """Return legacy single-file labeled path (backward compat)."""
    return LABELED_DIR / timeframe / label_config / f"{market}_labeled.parquet"


def _labeled_dir_exists(market: str, label_config: str, timeframe: str = "1m") -> bool:
    """Check if partitioned labeled data exists for this market."""
    d = _labeled_path(market, label_config, timeframe)
    return d.exists() and d.is_dir() and any(d.rglob("*.parquet"))


def _featured_path(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> Path:
    """Return the featured directory for partitioned layout."""
    return FEATURED_DIR / timeframe / label_config / model_config / market


def _featured_legacy_path(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> Path:
    """Return legacy single-file featured path (backward compat)."""
    return FEATURED_DIR / timeframe / label_config / model_config / f"{market}_featured.parquet"


def _featured_dir_exists(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> bool:
    """Check if partitioned featured data exists."""
    d = _featured_path(market, label_config, model_config, timeframe)
    return d.exists() and d.is_dir() and any(d.rglob("*.parquet"))


def _model_path(market: str, label_config: str, model_config: str, mtype: str, target: str, timeframe: str = "1m") -> Path:
    ext = "txt" if mtype == "lgb" else "pt"
    return MODELS_DIR / timeframe / label_config / model_config / f"{mtype}_{market}_{target}.{ext}"


def _splits_path(market: str, label_config: str, model_config: str, split: str, timeframe: str = "1m") -> Path:
    return MODELS_DIR / timeframe / label_config / model_config / "splits" / f"{market}_{split}.parquet"


def _predicted_path(market: str, label_config: str, model_config: str, model_type: str = "gbm", timeframe: str = "1m") -> Path:
    """Legacy single-file path (backward compat)."""
    return PREDICTIONS_DIR / model_type / timeframe / label_config / model_config / f"{market}_predicted.parquet"


def _predicted_dir(market: str, label_config: str, model_config: str, model_type: str = "gbm", timeframe: str = "1m") -> Path:
    """Partitioned prediction directory: .../market/ containing symbol/year.parquet."""
    return PREDICTIONS_DIR / model_type / timeframe / label_config / model_config / market


def _raw_stock_dir(timeframe: str = "1m") -> Path:
    """Return the raw stock directory for the given timeframe."""
    if timeframe == "5m":
        return RAW_GENERATED_DIR / "stock" / "5m"
    return RAW_STOCK_DIR


def _raw_symbols(market: str, timeframe: str = "1m") -> list[str]:
    """List available symbols in raw data directory."""
    market_dir = _raw_stock_dir(timeframe) / market
    if not market_dir.exists():
        return []
    return sorted(d.name for d in market_dir.iterdir() if d.is_dir())


# ── Raw OHLCV ─────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def get_raw_date_range(market: str, symbol: str, timeframe: str = "1m") -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Get min/max datetime for a symbol by reading only the first and last parquet files."""
    raw_dir = _raw_stock_dir(timeframe)

    symbol_dir = raw_dir / market / symbol
    if not symbol_dir.exists() and market == "kr":
        stripped = symbol.lstrip("0")
        if stripped != symbol:
            symbol_dir = raw_dir / market / stripped
    if not symbol_dir.exists():
        return None

    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    first = pd.read_parquet(parquet_files[0], columns=["datetime"])
    last = pd.read_parquet(parquet_files[-1], columns=["datetime"]) if len(parquet_files) > 1 else first
    return first["datetime"].min(), last["datetime"].max()


@st.cache_data(show_spinner="Loading trading dates...")
def get_raw_trading_dates(market: str, symbol: str, timeframe: str = "1m") -> list:
    """Get unique trading dates for a symbol by reading only the datetime column."""
    raw_dir = _raw_stock_dir(timeframe)

    symbol_dir = raw_dir / market / symbol
    if not symbol_dir.exists() and market == "kr":
        stripped = symbol.lstrip("0")
        if stripped != symbol:
            symbol_dir = raw_dir / market / stripped
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
def load_raw_bars(market: str, symbol: str, start_date: str | None = None, end_date: str | None = None, timeframe: str = "1m") -> pd.DataFrame:
    if timeframe == "5m":
        # For 5m, load from raw-generated directory
        raw_dir = _raw_stock_dir("5m")
        symbol_dir = raw_dir / market / symbol
        if not symbol_dir.exists():
            return pd.DataFrame()
        dfs = []
        for pf in sorted(symbol_dir.glob("*.parquet")):
            df = pd.read_parquet(pf)
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        result["datetime"] = pd.to_datetime(result["datetime"])
        if start_date:
            result = result[result["datetime"] >= pd.Timestamp(start_date)]
        if end_date:
            result = result[result["datetime"] < pd.Timestamp(end_date)]
        return result.sort_values("datetime").reset_index(drop=True)
    from src.collector.storage import load_bars
    return load_bars(market, symbol, start_date, end_date)


@st.cache_data(show_spinner="Scanning symbols...")
def get_raw_symbols(market: str, timeframe: str = "1m") -> list[str]:
    return _raw_symbols(market, timeframe)


@st.cache_data(show_spinner="Counting raw data...")
def get_raw_summary(market: str, timeframe: str = "1m") -> dict:
    """Summarise raw data for a market: symbol count, total bars, date range."""
    symbols = _raw_symbols(market, timeframe)
    if not symbols:
        return {"exists": False}
    raw_dir = _raw_stock_dir(timeframe)
    total_bars = 0
    min_dt, max_dt = None, None
    for sym in symbols:
        sym_dir = raw_dir / market / sym
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


def _load_manual_overrides(market: str, label_config: str, timeframe: str = "1m") -> pd.DataFrame:
    """Load manual label overrides (empty DataFrame if none exist)."""
    # Timeframe-aware path first
    path = LABELED_MANUAL_DIR / timeframe / label_config / f"{market}_manual.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Legacy fallback (no timeframe)
    legacy = LABELED_MANUAL_DIR / label_config / f"{market}_manual.parquet"
    if legacy.exists():
        return pd.read_parquet(legacy)
    return pd.DataFrame(columns=["symbol", "datetime", "label"])


@st.cache_data(show_spinner=False)
def get_labeled_symbols(market: str, label_config: str, timeframe: str = "1m") -> list[str]:
    """Return sorted symbol list from labeled data."""
    if _labeled_dir_exists(market, label_config, timeframe):
        d = _labeled_path(market, label_config, timeframe)
        return sorted(p.name for p in d.iterdir() if p.is_dir())
    # Legacy fallback
    legacy = _labeled_legacy_path(market, label_config, timeframe)
    if not legacy.exists():
        return []
    df = pd.read_parquet(legacy, columns=["symbol"])
    return sorted(df["symbol"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_labeled_date_range(market: str, label_config: str, symbol: str, timeframe: str = "1m") -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (min, max) datetime for a symbol in labeled data."""
    if _labeled_dir_exists(market, label_config, timeframe):
        sym_dir = _labeled_path(market, label_config, timeframe) / symbol
        if not sym_dir.exists():
            return None
        files = sorted(sym_dir.glob("*.parquet"))
        if not files:
            return None
        first = pd.read_parquet(files[0], columns=["datetime"])
        last = pd.read_parquet(files[-1], columns=["datetime"]) if len(files) > 1 else first
        return first["datetime"].min(), last["datetime"].max()
    # Legacy fallback
    legacy = _labeled_legacy_path(market, label_config, timeframe)
    if not legacy.exists():
        return None
    df = pd.read_parquet(legacy, columns=["symbol", "datetime"], filters=[("symbol", "==", symbol)])
    if df.empty:
        return None
    return df["datetime"].min(), df["datetime"].max()


@st.cache_data(show_spinner=False)
def get_labeled_trading_dates(market: str, label_config: str, symbol: str, timeframe: str = "1m") -> list:
    """Return sorted list of trading dates for a symbol."""
    if _labeled_dir_exists(market, label_config, timeframe):
        sym_dir = _labeled_path(market, label_config, timeframe) / symbol
        if not sym_dir.exists():
            return []
        dates = set()
        for pf in sym_dir.glob("*.parquet"):
            df = pd.read_parquet(pf, columns=["date"])
            for d in df["date"].unique():
                dates.add(pd.Timestamp(d).date() if not isinstance(d, date) else d)
        return sorted(dates)
    # Legacy fallback
    legacy = _labeled_legacy_path(market, label_config, timeframe)
    if not legacy.exists():
        return []
    df = pd.read_parquet(legacy, columns=["symbol", "date"], filters=[("symbol", "==", symbol)])
    if df.empty:
        return []
    return sorted(pd.Timestamp(d).date() if not isinstance(d, date) else d for d in df["date"].unique())


@st.cache_data(show_spinner=False)
def get_labeled_symbol_stats(market: str, label_config: str, symbol: str, timeframe: str = "1m") -> dict:
    """Return label counts for a symbol (lightweight — label column only)."""
    if _labeled_dir_exists(market, label_config, timeframe):
        sym_dir = _labeled_path(market, label_config, timeframe) / symbol
        if not sym_dir.exists():
            return {}
        dfs = [pd.read_parquet(pf, columns=["label"]) for pf in sym_dir.glob("*.parquet")]
        if not dfs:
            return {}
        df = pd.concat(dfs, ignore_index=True)
        return df["label"].value_counts().to_dict()
    # Legacy fallback
    legacy = _labeled_legacy_path(market, label_config, timeframe)
    if not legacy.exists():
        return {}
    df = pd.read_parquet(legacy, columns=["symbol", "label"], filters=[("symbol", "==", symbol)])
    return df["label"].value_counts().to_dict()


@st.cache_data(show_spinner="Loading labeled data...")
def load_labeled(market: str, label_config: str, symbol: str | None = None, date_str: str | None = None, timeframe: str = "1m") -> pd.DataFrame:
    """Load labeled data with optional symbol/date filtering.

    Supports both partitioned (symbol/year) and legacy single-file layouts.
    """
    if _labeled_dir_exists(market, label_config, timeframe):
        base_dir = _labeled_path(market, label_config, timeframe)
        files: list[Path] = []
        if symbol is not None:
            sym_dir = base_dir / symbol
            if not sym_dir.exists():
                return pd.DataFrame()
            files = sorted(sym_dir.glob("*.parquet"))
        else:
            for sym_dir in sorted(base_dir.iterdir()):
                if sym_dir.is_dir():
                    files.extend(sorted(sym_dir.glob("*.parquet")))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        legacy = _labeled_legacy_path(market, label_config, timeframe)
        if not legacy.exists():
            return pd.DataFrame()
        filters = []
        if symbol is not None:
            filters.append(("symbol", "==", symbol))
        if date_str is not None:
            import datetime as _dt
            filters.append(("date", "==", _dt.date.fromisoformat(date_str)))
        df = pd.read_parquet(legacy, filters=filters if filters else None)

    if df.empty:
        return df

    # Apply date filter for partitioned layout
    if date_str is not None and _labeled_dir_exists(market, label_config, timeframe):
        import datetime as _dt
        target = _dt.date.fromisoformat(date_str)
        df["date"] = pd.to_datetime(df["date"]).dt.date if not isinstance(df["date"].iloc[0], _dt.date) else df["date"]
        df = df[df["date"] == target]

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


def save_label_edit(market: str, symbol: str, datetime_str: str, new_label: int, label_config: str, timeframe: str = "1m") -> None:
    """Upsert a manual label override and clear cache."""
    manual = _load_manual_overrides(market, label_config, timeframe)
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

    out_dir = LABELED_MANUAL_DIR / timeframe / label_config
    out_dir.mkdir(parents=True, exist_ok=True)
    manual.to_parquet(out_dir / f"{market}_manual.parquet", index=False)
    st.cache_data.clear()
    logger.info("Manual override saved: %s/%s %s %s -> %d", timeframe, label_config, symbol, datetime_str, new_label)


@st.cache_data(show_spinner="Summarising labels...")
def get_labeled_summary(market: str, label_config: str, timeframe: str = "1m") -> dict:
    if _labeled_dir_exists(market, label_config, timeframe):
        base_dir = _labeled_path(market, label_config, timeframe)
        files = sorted(base_dir.rglob("*.parquet"))
        if not files:
            return {"exists": False}
        dfs = [pd.read_parquet(f, columns=["label", "symbol", "datetime"]) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        legacy = _labeled_legacy_path(market, label_config, timeframe)
        if not legacy.exists():
            return {"exists": False}
        df = pd.read_parquet(legacy, columns=["label", "symbol", "datetime"])
    return {
        "exists": True,
        "total_bars": len(df),
        "n_symbols": df["symbol"].nunique(),
        "label_counts": df["label"].value_counts().to_dict(),
        "date_range": (str(df["datetime"].min())[:10], str(df["datetime"].max())[:10]),
    }


# ── Prediction data ───────────────────────────────────────


@st.cache_data(show_spinner="Loading predictions...")
def load_predicted(
    market: str, label_config: str, model_config: str,
    model_type: str = "gbm", timeframe: str = "1m",
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load predicted labels from PREDICTIONS_DIR.

    Checks partitioned directory first, falls back to legacy single file.
    """
    # Partitioned layout: .../market/symbol/year.parquet
    part_dir = _predicted_dir(market, label_config, model_config, model_type, timeframe)
    if part_dir.exists() and part_dir.is_dir():
        if symbol:
            sym_dir = part_dir / symbol
            if not sym_dir.exists():
                return pd.DataFrame()
            files = sorted(sym_dir.glob("*.parquet"))
        else:
            files = sorted(part_dir.rglob("*.parquet"))
        if files:
            dfs = [pd.read_parquet(f) for f in files]
            return pd.concat(dfs, ignore_index=True)

    # Legacy single-file fallback
    path = _predicted_path(market, label_config, model_config, model_type, timeframe)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_available_model_types(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> list[str]:
    """Scan PREDICTIONS_DIR for available model_type subdirectories that have data."""
    available = []
    for mt in ("gbm", "lstm", "ensemble"):
        part_dir = _predicted_dir(market, label_config, model_config, mt, timeframe)
        legacy_path = _predicted_path(market, label_config, model_config, mt, timeframe)
        if (part_dir.exists() and any(part_dir.rglob("*.parquet"))) or legacy_path.exists():
            available.append(mt)
    return available


@st.cache_data(show_spinner=False)
def find_configs_for_model_type(market: str, model_type: str, timeframe: str = "1m") -> list[tuple[str, str]]:
    """Find all (label_config, model_config) pairs that have prediction data for a model_type."""
    from config.variants import get_label_configs, get_model_configs
    label_configs = get_label_configs(timeframe)
    model_configs = get_model_configs(timeframe)
    results = []
    for lc in sorted(label_configs.keys()):
        for mc in sorted(model_configs.keys()):
            part_dir = _predicted_dir(market, lc, mc, model_type, timeframe)
            legacy_path = _predicted_path(market, lc, mc, model_type, timeframe)
            if (part_dir.exists() and any(part_dir.rglob("*.parquet"))) or legacy_path.exists():
                results.append((lc, mc))
    return results


# ── Split info ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_split_dates(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> dict[str, set[str]]:
    """Return {split_name: set_of_date_strings} for train/val/test."""
    result: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = _splits_path(market, label_config, model_config, split, timeframe)
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
def get_backtest_file_symbols(name: str) -> list[str]:
    """Return sorted symbol list from a backtest file (old backtest system)."""
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
def get_feature_column_list(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> list[str]:
    """Read only the column names from featured parquet (fast)."""
    import pyarrow.parquet as pq
    from src.features.feature_pipeline import FEATURE_PREFIXES

    if _featured_dir_exists(market, label_config, model_config, timeframe):
        d = _featured_path(market, label_config, model_config, timeframe)
        # Read schema from the first parquet file found
        first_file = next(d.rglob("*.parquet"), None)
        if first_file is None:
            return []
        schema = pq.read_schema(first_file)
    else:
        legacy = _featured_legacy_path(market, label_config, model_config, timeframe)
        if not legacy.exists():
            return []
        schema = pq.read_schema(legacy)

    return sorted(
        col for col in schema.names
        if any(col.startswith(p) for p in FEATURE_PREFIXES)
    )


@st.cache_data(show_spinner="Loading features...")
def load_featured(market: str, label_config: str, model_config: str, columns: list[str] | None = None, max_rows: int = 50_000, timeframe: str = "1m") -> pd.DataFrame:
    """Load featured parquet with optional column filtering and row sampling."""
    read_cols = None
    if columns:
        meta_cols = ["datetime", "symbol", "date", "label", "minutes_from_open"]
        read_cols = list(set(meta_cols + columns))

    if _featured_dir_exists(market, label_config, model_config, timeframe):
        d = _featured_path(market, label_config, model_config, timeframe)
        files = sorted(d.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f, columns=read_cols) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        legacy = _featured_legacy_path(market, label_config, model_config, timeframe)
        if not legacy.exists():
            return pd.DataFrame()
        df = pd.read_parquet(legacy, columns=read_cols)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).sort_values("datetime").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Summarising features...")
def get_featured_summary(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> dict:
    import pyarrow.parquet as pq
    from src.features.feature_pipeline import FEATURE_PREFIXES

    if _featured_dir_exists(market, label_config, model_config, timeframe):
        d = _featured_path(market, label_config, model_config, timeframe)
        files = sorted(d.rglob("*.parquet"))
        if not files:
            return {"exists": False}
        total_rows = 0
        total_size = 0
        schema = pq.read_schema(files[0])
        for f in files:
            meta = pq.read_metadata(f)
            total_rows += meta.num_rows
            total_size += f.stat().st_size
        feat_cols = [c for c in schema.names if any(c.startswith(p) for p in FEATURE_PREFIXES)]
        return {
            "exists": True,
            "total_rows": total_rows,
            "n_features": len(feat_cols),
            "file_size_mb": round(total_size / 1024 / 1024, 1),
        }

    legacy = _featured_legacy_path(market, label_config, model_config, timeframe)
    if not legacy.exists():
        return {"exists": False}
    meta = pq.read_metadata(legacy)
    schema = pq.read_schema(legacy)
    feat_cols = [c for c in schema.names if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    return {
        "exists": True,
        "total_rows": meta.num_rows,
        "n_features": len(feat_cols),
        "file_size_mb": round(legacy.stat().st_size / 1024 / 1024, 1),
    }


# ── Model evaluation ─────────────────────────────────────


@st.cache_data(show_spinner="Evaluating model...", ttl=3600)
def run_model_evaluation(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> dict | None:
    """Load LightGBM models and run full_evaluation on the test split."""
    peak_model_path = _model_path(market, label_config, model_config, "lgb", "peak", timeframe)
    trough_model_path = _model_path(market, label_config, model_config, "lgb", "trough", timeframe)

    has_featured = _featured_dir_exists(market, label_config, model_config, timeframe) or _featured_legacy_path(market, label_config, model_config, timeframe).exists()
    if not has_featured or not peak_model_path.exists() or not trough_model_path.exists():
        return None

    import numpy as np
    from src.features.feature_pipeline import get_all_feature_columns, load_all_featured
    from src.model.dataset import prepare_xy, time_based_split
    from src.model.evaluate import full_evaluation
    from src.model.train_gbm import load_model

    df = load_all_featured(market, label_config, model_config, timeframe)
    if df.empty:
        return None
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
def get_pr_curve_data(market: str, label_config: str, model_config: str, target_label: int, timeframe: str = "1m") -> dict | None:
    """Get precision-recall curve arrays for plotting."""
    label_name = "peak" if target_label == 1 else "trough"
    model_path = _model_path(market, label_config, model_config, "lgb", label_name, timeframe)

    has_featured = _featured_dir_exists(market, label_config, model_config, timeframe) or _featured_legacy_path(market, label_config, model_config, timeframe).exists()
    if not has_featured or not model_path.exists():
        return None

    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from src.features.feature_pipeline import get_all_feature_columns, load_all_featured
    from src.model.dataset import prepare_xy, time_based_split
    from src.model.train_gbm import load_model

    df = load_all_featured(market, label_config, model_config, timeframe)
    if df.empty:
        return None
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
def get_feature_importance(market: str, label_config: str, model_config: str, target_label: int, top_n: int = 20, timeframe: str = "1m") -> pd.DataFrame:
    """Extract feature importance from LightGBM model."""
    label_name = "peak" if target_label == 1 else "trough"
    model_path = _model_path(market, label_config, model_config, "lgb", label_name, timeframe)

    has_featured = _featured_dir_exists(market, label_config, model_config, timeframe) or _featured_legacy_path(market, label_config, model_config, timeframe).exists()
    if not model_path.exists() or not has_featured:
        return pd.DataFrame()

    import lightgbm as lgb
    import pyarrow.parquet as pq
    from src.features.feature_pipeline import FEATURE_PREFIXES, get_all_feature_columns

    model = lgb.Booster(model_file=str(model_path))

    # Get schema from any available parquet file
    if _featured_dir_exists(market, label_config, model_config, timeframe):
        d = _featured_path(market, label_config, model_config, timeframe)
        first_file = next(d.rglob("*.parquet"), None)
        if first_file is None:
            return pd.DataFrame()
        schema = pq.read_schema(first_file)
    else:
        schema = pq.read_schema(_featured_legacy_path(market, label_config, model_config, timeframe))

    df_tmp = pd.DataFrame(columns=schema.names)
    feature_cols = get_all_feature_columns(df_tmp)

    importance = model.feature_importance(importance_type="gain")
    if len(feature_cols) != len(importance):
        feature_cols = [f"f_{i}" for i in range(len(importance))]

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return imp_df


# ── Model file checks ────────────────────────────────────


def get_model_status(market: str, label_config: str, model_config: str, timeframe: str = "1m") -> dict:
    """Check existence of model files for a market."""
    models = {}
    for mtype in ["lgb", "lstm"]:
        for target in ["peak", "trough"]:
            path = _model_path(market, label_config, model_config, mtype, target, timeframe)
            models[f"{mtype}_{target}"] = path.exists()
    return models


# ── New backtest system helpers ──────────────────────────


@st.cache_data(show_spinner=False)
def get_backtest_symbols(market: str = "us") -> list[str]:
    """List symbols that have options data for backtesting."""
    options_dir = RAW_OPTIONS_DIR / market
    if not options_dir.exists():
        return []
    return sorted(d.name for d in options_dir.iterdir() if d.is_dir())


@st.cache_data(show_spinner="Loading prediction data...")
def load_prediction_for_backtest(
    market: str, symbol: str, timeframe: str, label_config: str,
    model_config: str, model_type: str = "gbm",
) -> pd.DataFrame:
    """Load prediction data for a single symbol for backtesting."""
    pred_dir = PREDICTIONS_DIR / model_type / timeframe / label_config / model_config / market / symbol
    if not pred_dir.exists():
        return pd.DataFrame()
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    result = pd.concat(dfs, ignore_index=True)
    result["symbol"] = symbol
    return result


def run_dashboard_backtest(
    market: str, symbol: str, pred_df: pd.DataFrame,
    threshold: float, tp_pct: float, sl_pct: float,
    session_minutes: int = 390,
) -> dict:
    """Run backtest and return analyzer results for dashboard display.

    Returns dict with keys: "result" (SimulationResult), "metrics" (dict),
    "trades_df" (DataFrame), "equity_df" (DataFrame)
    """
    from src.backtest.analyzer import Analyzer
    from src.backtest.engine import BacktestEngine
    from src.backtest.executor.backtest import BacktestExecutor
    from src.backtest.strategy import Strategy, StrategyConfig

    config = StrategyConfig(threshold=threshold, tp_pct=tp_pct, sl_pct=sl_pct)
    executor = BacktestExecutor(symbols=[symbol], market=market)
    executor.load_data()
    engine = BacktestEngine(Strategy(config), executor)
    result = engine.run(pred_df, market, session_minutes)

    analyzer = Analyzer()
    metrics = analyzer.compute_metrics(result)
    dfs = analyzer.to_dataframes(result)

    return {
        "result": result,
        "metrics": metrics,
        "trades_df": dfs["trades"],
        "equity_df": dfs["equity"],
    }
