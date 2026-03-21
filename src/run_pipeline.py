"""Pipeline orchestrator: collector → labeler → features → model → predict → trade.

Usage:
    python run_pipeline.py all                    # Full pipeline
    python run_pipeline.py collector              # Data collection only
    python run_pipeline.py labeler                # Labeling only
    python run_pipeline.py features               # Feature engineering only
    python run_pipeline.py model                  # Model training only
    python run_pipeline.py predict                # Inference on latest data
    python run_pipeline.py batch_predict          # Batch prediction (all symbols)
    python run_pipeline.py trade                  # Mock trading simulation

Options:
    --market kr|us|all                            # Market selection (default: all)
    --model gbm|lstm|all                          # Model type (default: all)
    --full                                        # Full re-collection (instead of incremental)
    --threshold 0.5                               # Prediction threshold (predict/trade)
    --date 2026-02-20                             # Target date (predict/trade)
    --optimize                                    # Optuna HP search for GBM (model only)
    --timeframe 1m|5m                             # Timeframe (default: 1m)
    --label-config L1|L2|L3|all                   # Label variant (default: all)
    --model-config M1|M2|M3|M4|all                # Model variant (default: all)
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

from config.settings import DATA_DIR, LABELED_DIR, PREDICTIONS_DIR, PROCESSED_DIR, SUPPORTED_TIMEFRAMES
from config.variants import LABEL_CONFIGS, MODEL_CONFIGS, get_label_configs, get_model_configs

# MLflow tracking URI (local file store)
MLFLOW_TRACKING_URI = str(DATA_DIR / "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Variant helpers ──────────────────────────────────────


def _resolve_label_configs(label_config_arg: str, timeframe: str = "1m") -> list[str]:
    """Resolve --label-config argument to list of label config keys."""
    configs = get_label_configs(timeframe)
    if label_config_arg == "all":
        return list(configs.keys())
    if label_config_arg not in configs:
        raise ValueError(
            f"Label config {label_config_arg!r} not valid for timeframe {timeframe!r}. "
            f"Valid: {list(configs.keys())}"
        )
    return [label_config_arg]


def _resolve_model_configs(model_config_arg: str, timeframe: str = "1m") -> list[str]:
    """Resolve --model-config argument to list of model config keys."""
    configs = get_model_configs(timeframe)
    if model_config_arg == "all":
        return list(configs.keys())
    if model_config_arg not in configs:
        raise ValueError(
            f"Model config {model_config_arg!r} not valid for timeframe {timeframe!r}. "
            f"Valid: {list(configs.keys())}"
        )
    return [model_config_arg]


# ── Market helpers ───────────────────────────────────────


def _resolve_markets(market_arg: str) -> list[str]:
    """Resolve --market argument to list of market keys."""
    if market_arg == "all":
        return ["kr", "us"]
    return [market_arg]


def _symbol_list_keys(market: str) -> list[str]:
    """Return load_symbol_list() keys for a given market.

    'kr' → ['kr']
    'us' → ['us_stocks', 'us_etf_index']
    """
    if market == "kr":
        return ["kr"]
    return ["us_stocks", "us_etf_index"]


# ── Collector ────────────────────────────────────────────


def _ensure_databento_data() -> None:
    """Databento 원본 데이터가 없으면 rclone 으로 Google Drive 에서 복원 시도."""
    import subprocess

    stock_dbn = DATA_DIR / "raw" / "stock" / "databento" / "us"
    options_dbn = DATA_DIR / "raw" / "options" / "databento" / "us"

    has_stock = stock_dbn.exists() and any(stock_dbn.iterdir())
    has_options = options_dbn.exists() and any(options_dbn.iterdir())

    if has_stock and has_options:
        return

    restore_script = Path(__file__).parent / "collector" / "databento" / "restore_databento.sh"
    if not restore_script.exists():
        return

    logger.info("Databento 원본 데이터 없음 — Google Drive 에서 복원 시도")
    result = subprocess.run(["bash", str(restore_script)], capture_output=False)
    if result.returncode != 0:
        logger.warning("Databento 복원 실패 (rclone 미설치 또는 remote 미설정). 계속 진행합니다.")


def run_collector(markets: list[str], full: bool = False, symbols: list[str] | None = None, timeframe: str = "1m") -> None:
    """Run data collection for specified markets.

    Default (incremental): fetch from last collected date to today.
    --full: fetch entire available history (~60 days).
    --symbol: collect specific symbol(s) only.
    """
    from src.collector.bar_fetcher import BarFetcher, fetch_yfinance, load_symbol_list
    from src.collector.storage import get_symbol_date_range

    _ensure_databento_data()
    fetcher = BarFetcher()

    for market in markets:
        for key in _symbol_list_keys(market):
            symbols_df = load_symbol_list(key)

            if symbols:
                symbols_df = symbols_df[symbols_df["ticker"].astype(str).isin(symbols)]
                if symbols_df.empty:
                    continue
                symbols_df = symbols_df.reset_index(drop=True)

            total = len(symbols_df)
            logger.info(f"=== Collecting {key}: {total} symbols (full={full}) ===")

            for idx, row in symbols_df.iterrows():
                ticker = str(row["ticker"])
                exchange = row.get("exchange", "")
                tv_symbol = row.get("tv_symbol", "")
                if tv_symbol and ":" in tv_symbol:
                    exchange = tv_symbol

                logger.info(f"[{idx + 1}/{total}] {ticker}")

                if full:
                    # Full collection: clear existing + fresh collect
                    fetcher.collect_single(ticker, exchange, market, full=True)
                else:
                    # Incremental: check last collected date
                    date_range = get_symbol_date_range(market, ticker)
                    if date_range is not None:
                        last_date = date_range[1]
                        # Fetch from last date onward
                        start = (
                            datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                        ).strftime("%Y-%m-%d")
                        today = datetime.now().strftime("%Y-%m-%d")

                        if start >= today:
                            logger.info(f"  Already up-to-date ({last_date})")
                            continue

                        logger.info(f"  Incremental: {start} → {today}")
                        yf_df = fetch_yfinance(ticker, market, start_date=start)
                        if yf_df is not None and not yf_df.empty:
                            from src.collector.storage import save_bars

                            save_bars(yf_df, market=market, symbol=ticker)
                            logger.info(f"  Saved {len(yf_df)} incremental bars")

                        # tvDatafeed overlay (always latest ~13 days)
                        try:
                            tv_df = fetcher.tv_client.get_hist(
                                tv_symbol.split(":")[-1] if ":" in tv_symbol else ticker,
                                exchange.split(":")[0] if ":" in exchange else exchange,
                            )
                            if tv_df is not None and not tv_df.empty:
                                from config.settings import KR_TIMEZONE, US_TIMEZONE
                                from src.collector.storage import save_bars

                                # tvDatafeed returns KST timestamps; convert to ET for US
                                if market == "us":
                                    tv_df.index = (
                                        tv_df.index
                                        .tz_localize(KR_TIMEZONE)
                                        .tz_convert(US_TIMEZONE)
                                        .tz_localize(None)
                                    )
                                save_bars(tv_df, market=market, symbol=ticker)
                                logger.info(f"  tvDatafeed overlay: {len(tv_df)} bars")
                        except Exception as e:
                            logger.warning(f"  tvDatafeed failed: {e}")
                    else:
                        # No existing data → full collection for this symbol
                        logger.info(f"  No existing data, full collection")
                        fetcher.collect_single(ticker, exchange, market)

    summary = fetcher.tracker.summary()
    logger.info(f"Collection summary: {summary}")

    # Resample 1m → 5m when timeframe is 5m
    if timeframe == "5m":
        from src.collector.resampler import resample_all

        for market in markets:
            logger.info(f"=== Resampling 1m → 5m for {market} ===")
            resample_all(market)


# ── Labeler ──────────────────────────────────────────────


def run_labeler(markets: list[str], label_config: str = "L1", timeframe: str = "1m") -> None:
    """Run labeling for all symbols in specified markets with given label config.

    Saves labeled data in partitioned layout: symbol/year parquet files.
    """
    from config.settings import RAW_STOCK_DIR
    from src.labeler.label_generator import (
        apply_manual_overrides,
        label_statistics,
        label_symbol,
        save_labeled_partitioned,
    )

    label_configs_dict = get_label_configs(timeframe)
    lcfg = label_configs_dict[label_config]

    for market in markets:
        logger.info(f"=== Labeling {market} [{label_config}] (timeframe={timeframe}) ===")

        market_dir = RAW_STOCK_DIR / market
        if not market_dir.exists():
            logger.error(f"No raw data directory for market {market}")
            continue
        symbols = [d.name for d in market_dir.iterdir() if d.is_dir()]
        if not symbols:
            logger.warning(f"No symbols found for {market}")
            continue

        from tqdm import tqdm

        all_stats_dfs: list[pd.DataFrame] = []
        for symbol in tqdm(symbols, desc=f"Labeling {market}"):
            labeled = label_symbol(
                market, symbol,
                prominence_pct=lcfg["prominence_pct"],
                distance=lcfg["distance"],
                width=lcfg["width"],
                shift=lcfg["shift"],
            )
            if labeled is None:
                continue

            labeled = apply_manual_overrides(labeled, market, label_config=label_config, timeframe=timeframe)
            save_labeled_partitioned(labeled, market, label_config, timeframe)
            all_stats_dfs.append(labeled)

        if not all_stats_dfs:
            logger.warning(f"No data to label for {market}")
            continue

        combined_stats = pd.concat(all_stats_dfs, ignore_index=True)
        stats = label_statistics(combined_stats)
        logger.info(f"Label statistics for {market} [{label_config}]:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")


# ── Features ─────────────────────────────────────────────


def run_features(markets: list[str], label_config: str = "L1", model_config: str = "M1", timeframe: str = "1m") -> None:
    """Build features from labeled data with given variant configs.

    Processes per symbol/year to keep peak memory low, saving each partition
    separately to disk.
    """
    from config.settings import KR_SESSION_MINUTES, US_SESSION_MINUTES
    from src.features.feature_pipeline import (
        build_features,
        build_lookback_features,
        clean_features,
        get_feature_columns,
        save_featured_partitioned,
    )
    from src.labeler.label_generator import (
        list_labeled_symbols,
        list_labeled_years,
        load_labeled,
    )

    model_configs_dict = get_model_configs(timeframe)
    mcfg = model_configs_dict[model_config]

    for market in markets:
        logger.info(f"=== Building features for {market} [{label_config}/{model_config}] (timeframe={timeframe}) ===")

        symbols = list_labeled_symbols(market, label_config, timeframe)
        if not symbols:
            logger.warning(f"No labeled symbols for {market} [{label_config}], skipping")
            continue

        session_minutes = KR_SESSION_MINUTES if market == "kr" else US_SESSION_MINUTES
        total_rows = 0

        from tqdm import tqdm

        for symbol in tqdm(symbols, desc=f"Features {market}"):
            years = list_labeled_years(market, label_config, timeframe, symbol)
            for year in years:
                df = load_labeled(market, label_config, timeframe, symbol=symbol, year=year)
                if df.empty:
                    continue

                df = build_features(df, session_minutes=session_minutes)
                df = build_lookback_features(
                    df,
                    lookback=mcfg["gbm_lookback"],
                    fill_method=mcfg["fill_method"],
                )
                df = clean_features(df)

                save_featured_partitioned(
                    df, market, label_config, model_config, timeframe, symbol, year,
                )
                total_rows += len(df)

        if total_rows > 0:
            feature_cols = get_feature_columns(df)
            logger.info(
                f"Featured {market} [{label_config}/{model_config}]: "
                f"{total_rows} rows, {len(feature_cols)} base features"
            )
        else:
            logger.warning(f"No featured data produced for {market}")


# ── Model ────────────────────────────────────────────────


def run_model(
    markets: list[str],
    model_type: str = "all",
    label_config: str = "L1",
    model_config: str = "M1",
    optimize: bool = False,
    timeframe: str = "1m",
) -> None:
    """Train and evaluate models for a specific variant."""
    from src.features.feature_pipeline import get_all_feature_columns, get_base_feature_columns
    from src.model.dataset import SplitResult, prepare_xy, time_based_split
    from src.model.evaluate import full_evaluation

    model_configs_dict = get_model_configs(timeframe)
    label_configs_dict = get_label_configs(timeframe)
    mcfg = model_configs_dict[model_config]
    lcfg = label_configs_dict[label_config]

    models_dir = DATA_DIR / "models" / timeframe / label_config / model_config
    models_dir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        logger.info(f"=== Training models for {market} [{label_config}/{model_config}] (timeframe={timeframe}) ===")

        from src.features.feature_pipeline import (
            build_incremental_chunks,
            get_featured_partition_info,
            load_all_featured,
            load_chunk,
        )

        # Decide: incremental vs standard loading
        partition_info = get_featured_partition_info(market, label_config, model_config, timeframe)
        if not partition_info:
            logger.warning(f"No featured data for {market} [{label_config}/{model_config}], skipping")
            continue

        chunks = build_incremental_chunks(market, label_config, model_config, timeframe)
        use_incremental = len(chunks) > 1

        if use_incremental:
            logger.info(f"Using incremental training ({len(chunks)} chunks)")

            # Compute split dates from partition metadata (load minimal data)
            # Read datetime range from first and last partition
            import pyarrow.parquet as pq
            all_paths = [p["path"] for p in partition_info]
            first_df = pd.read_parquet(all_paths[0], columns=["datetime"])
            last_df = pd.read_parquet(all_paths[-1], columns=["datetime"])
            min_date = pd.to_datetime(first_df["datetime"]).min()
            max_date = pd.to_datetime(last_df["datetime"]).max()
            del first_df, last_df

            from config.settings import TEST_MONTHS, VAL_MONTHS
            test_start = max_date - pd.DateOffset(months=TEST_MONTHS)
            val_start = test_start - pd.DateOffset(months=VAL_MONTHS)

            # Fallback: 데이터가 부족하면 비율 기반 (날짜 기준)
            total_days = (max_date - min_date).days
            if total_days < 60:
                total_days_span = total_days
                val_start = min_date + pd.Timedelta(days=int(total_days_span * 0.6))
                test_start = min_date + pd.Timedelta(days=int(total_days_span * 0.8))
                logger.warning(
                    f"Insufficient data ({total_days} days). "
                    f"Falling back to ratio-based split dates."
                )

            split_dates = {
                "val_start": str(val_start),
                "test_start": str(test_start),
            }
            logger.info(f"Split dates: val_start={val_start.date()}, test_start={test_start.date()}")

            # Get feature columns from first partition (small read)
            sample_df = pd.read_parquet(all_paths[0])
            feature_cols = get_all_feature_columns(sample_df)
            lstm_feature_cols = get_base_feature_columns(sample_df)
            del sample_df
            logger.info(f"Feature columns: {len(feature_cols)} (GBM), {len(lstm_feature_cols)} (LSTM)")

            mlflow.set_experiment(f"option-meme/{market}")
            with mlflow.start_run(run_name=f"{label_config}_{model_config}"):
                mlflow.log_params({
                    "market": market,
                    "label_config": label_config,
                    "model_config": model_config,
                    "model_type": model_type,
                    "incremental": True,
                    "n_chunks": len(chunks),
                    "prominence_pct": lcfg["prominence_pct"],
                    "label_distance": lcfg["distance"],
                    "label_width": lcfg["width"],
                    "gbm_lookback": mcfg["gbm_lookback"],
                    "lstm_lookback": mcfg["lstm_lookback"],
                    "fill_method": mcfg["fill_method"],
                })

                for target_label, label_name in [(1, "peak"), (2, "trough")]:
                    logger.info(f"--- Target: {label_name} (label={target_label}) ---")

                    if model_type in ("gbm", "all"):
                        _train_lgb_model_incremental(
                            chunks, target_label, label_name, feature_cols,
                            split_dates, market, models_dir, optimize=optimize,
                        )

                # Full evaluation (incremental: load test data per chunk)
                if model_type in ("gbm", "all"):
                    _run_full_evaluation_incremental(
                        chunks, feature_cols, market, models_dir,
                        test_start=str(test_start),
                    )
        else:
            # Standard path: data fits in memory
            df = load_all_featured(market, label_config, model_config, timeframe)
            if df.empty:
                logger.warning(f"No featured data for {market} [{label_config}/{model_config}], skipping")
                continue
            logger.info(f"Loaded {len(df)} featured bars")

            feature_cols = get_all_feature_columns(df)
            lstm_feature_cols = get_base_feature_columns(df)
            logger.info(f"Feature columns: {len(feature_cols)} (GBM), {len(lstm_feature_cols)} (LSTM)")

            split = time_based_split(df)
            logger.info(
                f"Split: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}"
            )

            # Save train/val/test splits for reproducibility
            splits_dir = models_dir / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            for split_name, split_df in [("train", split.train), ("val", split.val), ("test", split.test)]:
                path = splits_dir / f"{market}_{split_name}.parquet"
                split_df.to_parquet(path, index=False, compression="snappy")
            logger.info(f"Saved train/val/test splits to {splits_dir}")

            mlflow.set_experiment(f"option-meme/{market}")
            with mlflow.start_run(run_name=f"{label_config}_{model_config}"):
                mlflow.log_params({
                    "market": market,
                    "label_config": label_config,
                    "model_config": model_config,
                    "model_type": model_type,
                    "incremental": False,
                    "prominence_pct": lcfg["prominence_pct"],
                    "label_distance": lcfg["distance"],
                    "label_width": lcfg["width"],
                    "gbm_lookback": mcfg["gbm_lookback"],
                    "lstm_lookback": mcfg["lstm_lookback"],
                    "fill_method": mcfg["fill_method"],
                })

                for target_label, label_name in [(1, "peak"), (2, "trough")]:
                    logger.info(f"--- Target: {label_name} (label={target_label}) ---")

                    # LightGBM
                    if model_type in ("gbm", "all"):
                        _train_lgb_model(
                            split, target_label, label_name, feature_cols,
                            market, models_dir, optimize=optimize,
                        )

                    # LSTM
                    if model_type in ("lstm", "all"):
                        _train_lstm_model(
                            split, target_label, label_name, lstm_feature_cols,
                            market, models_dir,
                            lstm_lookback=mcfg["lstm_lookback"],
                            fill_method=mcfg["fill_method"],
                        )

                # Full evaluation with LightGBM predictions (if available)
                if model_type in ("gbm", "all"):
                    _run_full_evaluation(split, feature_cols, market, models_dir)


def _train_lgb_model_incremental(
    chunks, target_label, label_name, feature_cols, split_dates,
    market, models_dir, optimize: bool = False,
):
    """Train and save a LightGBM model using incremental learning."""
    from src.model.train_gbm import save_model as save_lgb
    from src.model.train_gbm import train_lgb_incremental

    # Use saved Optuna params if available
    params_path = models_dir / f"lgb_{market}_{label_name}_params.json"
    if params_path.exists():
        from config.settings import LGB_PARAMS
        saved_params = json.loads(params_path.read_text())
        lgb_params = {**LGB_PARAMS, **saved_params}
        logger.info(f"Training LightGBM for {label_name} (incremental, saved Optuna params)...")
    else:
        lgb_params = None
        logger.info(f"Training LightGBM for {label_name} (incremental)...")

    model, metrics = train_lgb_incremental(
        chunks, target_label, feature_cols, split_dates, params=lgb_params,
    )

    model_path = models_dir / f"lgb_{market}_{label_name}.txt"
    save_lgb(model, model_path)

    logger.info(f"LightGBM {label_name} metrics:")
    for k, v in metrics.items():
        if k != "top_features":
            logger.info(f"  {k}: {v}")
    if "top_features" in metrics:
        logger.info("  Top 10 features:")
        for name, imp in metrics["top_features"][:10]:
            logger.info(f"    {name}: {imp:.1f}")

    with mlflow.start_run(run_name=f"gbm_{label_name}", nested=True):
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if k != "top_features" and isinstance(v, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_artifact(str(model_path))
        if params_path.exists():
            mlflow.log_artifact(str(params_path))


def _run_full_evaluation_incremental(
    chunks, feature_cols, market, models_dir, test_start: str,
):
    """Run full evaluation using incremental chunks (test set only)."""
    import numpy as np

    from src.features.feature_pipeline import load_chunk
    from src.model.evaluate import full_evaluation
    from src.model.train_gbm import load_model as load_lgb

    peak_model_path = models_dir / f"lgb_{market}_peak.txt"
    trough_model_path = models_dir / f"lgb_{market}_trough.txt"

    if not peak_model_path.exists() or not trough_model_path.exists():
        logger.warning("Skipping full evaluation: LightGBM models not found")
        return

    peak_model = load_lgb(peak_model_path)
    trough_model = load_lgb(trough_model_path)

    test_start_ts = pd.Timestamp(test_start)

    # Collect test data from all chunks
    test_parts = []
    for chunk in chunks:
        df = load_chunk(chunk)
        df["datetime"] = pd.to_datetime(df["datetime"])
        test_df = df[df["datetime"] >= test_start_ts]
        if not test_df.empty:
            test_parts.append(test_df)
        del df

    if not test_parts:
        logger.warning("No test data found for evaluation")
        return

    test_all = pd.concat(test_parts, ignore_index=True)
    del test_parts

    from src.model.dataset import prepare_xy

    X_test, _ = prepare_xy(test_all, target_label=1, feature_cols=feature_cols)
    peak_proba = peak_model.predict(X_test)
    trough_proba = trough_model.predict(X_test)

    eval_results = full_evaluation(test_all, peak_proba, trough_proba)

    logger.info("=== Full Evaluation Report (Incremental) ===")
    logger.info(json.dumps(eval_results, indent=2, default=str))

    with mlflow.start_run(run_name="evaluation", nested=True):
        flat_metrics: dict[str, float] = {}
        for side in ("peak", "trough"):
            pr = eval_results.get(side, {}).get("pr_metrics", {})
            if "pr_auc" in pr:
                flat_metrics[f"{side}_pr_auc"] = pr["pr_auc"]
        bt = eval_results.get("backtest", {})
        for key in ("n_trades", "total_return", "buy_hold_return", "win_rate",
                    "avg_win", "avg_loss", "profit_factor", "max_drawdown", "sharpe_approx"):
            if key in bt and isinstance(bt[key], (int, float)):
                flat_metrics[f"backtest_{key}"] = bt[key]
        if flat_metrics:
            mlflow.log_metrics(flat_metrics)

    del test_all


def _train_lgb_model(
    split, target_label, label_name, feature_cols, market, models_dir,
    optimize: bool = False,
):
    """Train and save a LightGBM model."""
    from src.model.train_gbm import save_model as save_lgb
    from src.model.train_gbm import train_lgb

    if optimize:
        from src.model.train_gbm import optimize_lgb

        logger.info(f"Optimizing LightGBM for {label_name} (Optuna)...")
        model, metrics, best_params = optimize_lgb(
            split, target_label, feature_cols, n_trials=50, timeout=3600,
        )

        # Save best params for reproducibility
        params_path = models_dir / f"lgb_{market}_{label_name}_params.json"
        params_path.write_text(json.dumps(best_params, indent=2))
        logger.info(f"Best params saved to {params_path}")
        lgb_params = best_params
    else:
        # Use saved Optuna params if available, otherwise defaults
        params_path = models_dir / f"lgb_{market}_{label_name}_params.json"
        if params_path.exists():
            from config.settings import LGB_PARAMS
            saved_params = json.loads(params_path.read_text())
            lgb_params = {**LGB_PARAMS, **saved_params}
            logger.info(f"Training LightGBM for {label_name} (using saved Optuna params)...")
            model, metrics = train_lgb(split, target_label, feature_cols, params=lgb_params)
        else:
            from config.settings import LGB_PARAMS
            lgb_params = LGB_PARAMS
            logger.info(f"Training LightGBM for {label_name}...")
            model, metrics = train_lgb(split, target_label, feature_cols)

    model_path = models_dir / f"lgb_{market}_{label_name}.txt"
    save_lgb(model, model_path)

    logger.info(f"LightGBM {label_name} metrics:")
    for k, v in metrics.items():
        if k != "top_features":
            logger.info(f"  {k}: {v}")
    if "top_features" in metrics:
        logger.info("  Top 10 features:")
        for name, imp in metrics["top_features"][:10]:
            logger.info(f"    {name}: {imp:.1f}")

    with mlflow.start_run(run_name=f"gbm_{label_name}", nested=True):
        # Log LGB hyperparameters (exclude internal/non-numeric options)
        loggable_params = {
            k: v for k, v in lgb_params.items()
            if k not in ("verbosity", "num_threads") and not isinstance(v, (list, dict))
        }
        mlflow.log_params(loggable_params)

        # Log metrics (exclude non-scalar fields)
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if k != "top_features" and isinstance(v, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)

        # Log model file
        mlflow.log_artifact(str(model_path))
        # Log params JSON if it exists (Optuna or pre-saved)
        if params_path.exists():
            mlflow.log_artifact(str(params_path))


def _train_lstm_model(
    split, target_label, label_name, feature_cols, market, models_dir,
    lstm_lookback=None, fill_method="drop",
):
    """Train and save an LSTM model."""
    from config.settings import (
        LOOKBACK_WINDOW,
        LSTM_BATCH_SIZE,
        LSTM_DROPOUT,
        LSTM_HIDDEN_SIZE,
        LSTM_LR,
        LSTM_NUM_LAYERS,
    )
    from src.model.train_lstm import save_model as save_lstm
    from src.model.train_lstm import train_lstm

    if lstm_lookback is None:
        lstm_lookback = LOOKBACK_WINDOW

    logger.info(f"Training LSTM for {label_name} (lookback={lstm_lookback}, fill={fill_method})...")
    model, metrics = train_lstm(
        split, target_label, feature_cols,
        lookback=lstm_lookback,
    )

    model_path = models_dir / f"lstm_{market}_{label_name}.pt"
    save_lstm(model, model_path)

    logger.info(f"LSTM {label_name} metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    with mlflow.start_run(run_name=f"lstm_{label_name}", nested=True):
        mlflow.log_params({
            "hidden_size": LSTM_HIDDEN_SIZE,
            "num_layers": LSTM_NUM_LAYERS,
            "dropout": LSTM_DROPOUT,
            "lr": LSTM_LR,
            "batch_size": LSTM_BATCH_SIZE,
            "lookback": lstm_lookback,
            "fill_method": fill_method,
        })

        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)

        mlflow.log_artifact(str(model_path))


def _run_full_evaluation(split, feature_cols, market, models_dir):
    """Run full evaluation using LightGBM predictions on test set."""
    from src.model.dataset import prepare_xy
    from src.model.evaluate import full_evaluation
    from src.model.train_gbm import load_model as load_lgb

    peak_model_path = models_dir / f"lgb_{market}_peak.txt"
    trough_model_path = models_dir / f"lgb_{market}_trough.txt"

    if not peak_model_path.exists() or not trough_model_path.exists():
        logger.warning("Skipping full evaluation: LightGBM models not found")
        return

    peak_model = load_lgb(peak_model_path)
    trough_model = load_lgb(trough_model_path)

    X_test, _ = prepare_xy(split.test, target_label=1, feature_cols=feature_cols)
    peak_proba = peak_model.predict(X_test)
    trough_proba = trough_model.predict(X_test)

    eval_results = full_evaluation(split.test, peak_proba, trough_proba)

    logger.info("=== Full Evaluation Report ===")
    logger.info(json.dumps(eval_results, indent=2, default=str))

    with mlflow.start_run(run_name="evaluation", nested=True):
        flat_metrics: dict[str, float] = {}

        for side in ("peak", "trough"):
            pr = eval_results.get(side, {}).get("pr_metrics", {})
            if "pr_auc" in pr:
                flat_metrics[f"{side}_pr_auc"] = pr["pr_auc"]
            if "positive_rate" in pr:
                flat_metrics[f"{side}_positive_rate"] = pr["positive_rate"]

        bt = eval_results.get("backtest", {})
        for key in ("n_trades", "total_return", "buy_hold_return", "win_rate",
                    "avg_win", "avg_loss", "profit_factor", "max_drawdown", "sharpe_approx"):
            if key in bt and isinstance(bt[key], (int, float)):
                flat_metrics[f"backtest_{key}"] = bt[key]

        if flat_metrics:
            mlflow.log_metrics(flat_metrics)


# ── Ensemble ─────────────────────────────────────────────


def _get_lstm_aligned_rows(df: pd.DataFrame, lookback: int, fill_method: str) -> pd.DataFrame:
    """Return the subset of df that LSTM TimeSeriesDataset includes.

    For fill_method='0fill': all rows.
    For fill_method='drop': skip first (lookback-1) rows per day.
    """
    if fill_method == "0fill":
        return df
    subsets = []
    for _, day_df in df.groupby("date"):
        subsets.append(day_df.iloc[lookback - 1:])
    return pd.concat(subsets).reset_index(drop=True) if subsets else df.iloc[0:0]


def run_ensemble(
    markets: list[str],
    label_config: str = "L2",
    model_config: str = "M3",
    timeframe: str = "1m",
) -> None:
    """Calibrate LSTM and find optimal GBM/LSTM ensemble weights.

    Steps per market:
      1. Load saved val/test splits from models_dir/splits/
      2. For each label (peak, trough):
         a. Run LSTM on val → fit IsotonicRegression calibrator → save
         b. Run GBM on val (same row subset)
         c. Grid search for optimal w_gbm that maximises val PR-AUC → save
      3. Evaluate GBM-only vs Ensemble on test set → log to MLflow

    Artifacts saved under data/models/{label_config}/{model_config}/:
      - lstm_{market}_{label}_calibrator.joblib
      - ensemble_{market}_weights.json

    Note:
      Ensemble is designed for US market. KR LSTM PR-AUC (0.13~0.33) is
      too low to benefit from ensembling — the calibration step will still
      run but the optimal weight will likely be ~1.0 (GBM-only).
    """
    import torch

    from src.features.feature_pipeline import get_all_feature_columns, get_base_feature_columns
    from src.model.calibrate import apply_calibration, fit_calibrator, save_calibrator
    from src.model.dataset import TimeSeriesDataset, prepare_xy
    from src.model.ensemble import (
        ensemble_predict,
        find_optimal_weight,
        save_weights,
    )
    from src.model.evaluate import full_evaluation
    from src.model.train_gbm import load_model as load_lgb
    from src.model.train_lstm import load_model as load_lstm
    from src.model.train_lstm import predict as lstm_predict

    model_configs_dict = get_model_configs(timeframe)
    mcfg = model_configs_dict[model_config]
    lstm_lookback = mcfg["lstm_lookback"]
    fill_method = mcfg["fill_method"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    for market in markets:
        if market == "kr":
            logger.warning(
                "KR LSTM PR-AUC is typically 0.13~0.33 — ensemble may not improve over GBM alone. "
                "Proceeding anyway; optimal weight will likely be w_gbm≈1.0."
            )

        models_dir = DATA_DIR / "models" / timeframe / label_config / model_config
        splits_dir = models_dir / "splits"

        # ── Load splits ──────────────────────────────────────────────────────
        val_path = splits_dir / f"{market}_val.parquet"
        test_path = splits_dir / f"{market}_test.parquet"
        if not val_path.exists() or not test_path.exists():
            logger.error(
                f"Splits not found at {splits_dir}. "
                f"Run model first: ./optionmeme model --market {market} "
                f"--label-config {label_config} --model-config {model_config}"
            )
            continue

        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        logger.info(
            f"=== Ensemble [{market}] {label_config}/{model_config} | "
            f"val={len(val_df)}, test={len(test_df)} rows ==="
        )

        # ── Feature columns ──────────────────────────────────────────────────
        feature_cols = get_all_feature_columns(val_df)
        lstm_feature_cols = get_base_feature_columns(val_df)
        if not feature_cols or not lstm_feature_cols:
            logger.error("No feature columns found in val split — skipping")
            continue

        # ── Load GBM models ──────────────────────────────────────────────────
        peak_gbm_path = models_dir / f"lgb_{market}_peak.txt"
        trough_gbm_path = models_dir / f"lgb_{market}_trough.txt"
        if not peak_gbm_path.exists() or not trough_gbm_path.exists():
            logger.error(f"GBM models not found in {models_dir}")
            continue
        peak_gbm = load_lgb(peak_gbm_path)
        trough_gbm = load_lgb(trough_gbm_path)

        # ── Load LSTM models ─────────────────────────────────────────────────
        peak_lstm_path = models_dir / f"lstm_{market}_peak.pt"
        trough_lstm_path = models_dir / f"lstm_{market}_trough.pt"
        if not peak_lstm_path.exists() or not trough_lstm_path.exists():
            logger.error(f"LSTM models not found in {models_dir}")
            continue
        n_lstm_features = len(lstm_feature_cols)
        peak_lstm = load_lstm(peak_lstm_path, n_features=n_lstm_features)
        trough_lstm = load_lstm(trough_lstm_path, n_features=n_lstm_features)

        ensemble_weights = {}

        mlflow.set_experiment(f"option-meme/{market}")
        with mlflow.start_run(run_name=f"ensemble_{label_config}_{model_config}"):
            mlflow.log_params({
                "market": market,
                "label_config": label_config,
                "model_config": model_config,
                "lstm_lookback": lstm_lookback,
                "fill_method": fill_method,
                "calibration": "isotonic",
            })

            for target_label, label_name in [(1, "peak"), (2, "trough")]:
                logger.info(f"--- {label_name} ---")

                # ── LSTM val predictions via TimeSeriesDataset ───────────────
                # TimeSeriesDataset handles day-boundaries; labels are aligned.
                lstm_model = peak_lstm if label_name == "peak" else trough_lstm
                val_ds = TimeSeriesDataset(
                    val_df, target_label=target_label, lookback=lstm_lookback,
                    feature_cols=lstm_feature_cols, fill_method=fill_method,
                )
                lstm_val_proba = lstm_predict(lstm_model, val_ds, device=device)
                lstm_val_labels = val_ds.targets  # already aligned

                logger.info(
                    f"LSTM val: n={len(lstm_val_proba)}, "
                    f"pos_rate={lstm_val_labels.mean():.4f}, "
                    f"pred_range=[{lstm_val_proba.min():.4f}, {lstm_val_proba.max():.4f}]"
                )

                # ── Fit calibrator on val set ────────────────────────────────
                calibrator = fit_calibrator(lstm_val_proba, lstm_val_labels)
                cal_path = models_dir / f"lstm_{market}_{label_name}_calibrator.joblib"
                save_calibrator(calibrator, cal_path)
                mlflow.log_artifact(str(cal_path))

                # Calibrated val proba
                lstm_val_cal = apply_calibration(calibrator, lstm_val_proba)

                # ── GBM val predictions (same row subset as LSTM) ────────────
                val_subset = _get_lstm_aligned_rows(val_df, lstm_lookback, fill_method)
                X_val_sub, y_val_sub = prepare_xy(val_subset, target_label, feature_cols)
                gbm_model = peak_gbm if label_name == "peak" else trough_gbm
                gbm_val_proba = gbm_model.predict(X_val_sub)

                # Align lengths (may differ by 1 at day boundaries in rare edge cases)
                n_common = min(len(gbm_val_proba), len(lstm_val_cal))
                gbm_val_proba = gbm_val_proba[:n_common]
                lstm_val_cal_aligned = lstm_val_cal[:n_common]
                y_val_labels = y_val_sub[:n_common]

                # ── Find optimal weight ──────────────────────────────────────
                best_w_gbm, best_val_auc = find_optimal_weight(
                    gbm_val_proba, lstm_val_cal_aligned, y_val_labels,
                )
                ensemble_weights[label_name] = {
                    "w_gbm": round(best_w_gbm, 4),
                    "val_pr_auc": round(best_val_auc, 4),
                }

                with mlflow.start_run(run_name=f"ensemble_{label_name}", nested=True):
                    mlflow.log_metrics({
                        f"best_w_gbm": best_w_gbm,
                        f"ensemble_val_pr_auc": best_val_auc,
                    })

                logger.info(
                    f"{label_name}: w_gbm={best_w_gbm:.2f}, val PR-AUC={best_val_auc:.4f}"
                )

            # ── Save ensemble weights ────────────────────────────────────────
            weights_path = models_dir / f"ensemble_{market}_weights.json"
            save_weights(ensemble_weights, weights_path)
            mlflow.log_artifact(str(weights_path))

            # ── Test set evaluation: GBM-only vs Ensemble ────────────────────
            logger.info("=== Test set evaluation: GBM-only vs Ensemble ===")

            X_test, _ = prepare_xy(test_df, target_label=1, feature_cols=feature_cols)
            gbm_peak_test = peak_gbm.predict(X_test)
            X_test_t, _ = prepare_xy(test_df, target_label=2, feature_cols=feature_cols)
            gbm_trough_test = trough_gbm.predict(X_test_t)

            # LSTM test predictions (per label, calibrated)
            from src.model.calibrate import load_calibrator

            def _lstm_test_calibrated(lstm_model, label_name_key, target_lbl):
                test_ds = TimeSeriesDataset(
                    test_df, target_label=target_lbl, lookback=lstm_lookback,
                    feature_cols=lstm_feature_cols, fill_method=fill_method,
                )
                raw_proba = lstm_predict(lstm_model, test_ds, device=device)
                cal = load_calibrator(models_dir / f"lstm_{market}_{label_name_key}_calibrator.joblib")
                cal_proba = apply_calibration(cal, raw_proba)
                # Pad front to match test_df length (for fill_method='drop')
                n_test = len(test_df)
                if len(cal_proba) < n_test:
                    cal_proba = np.concatenate([np.zeros(n_test - len(cal_proba)), cal_proba])
                return cal_proba[:n_test]

            lstm_peak_test_cal = _lstm_test_calibrated(peak_lstm, "peak", 1)
            lstm_trough_test_cal = _lstm_test_calibrated(trough_lstm, "trough", 2)

            w_peak = ensemble_weights["peak"]["w_gbm"]
            w_trough = ensemble_weights["trough"]["w_gbm"]
            ens_peak_test = ensemble_predict(gbm_peak_test, lstm_peak_test_cal, w_gbm=w_peak)
            ens_trough_test = ensemble_predict(gbm_trough_test, lstm_trough_test_cal, w_gbm=w_trough)

            gbm_eval = full_evaluation(test_df, gbm_peak_test, gbm_trough_test)
            ens_eval = full_evaluation(test_df, ens_peak_test, ens_trough_test)

            gbm_peak_auc = gbm_eval["peak"]["pr_metrics"]["pr_auc"]
            gbm_trough_auc = gbm_eval["trough"]["pr_metrics"]["pr_auc"]
            ens_peak_auc = ens_eval["peak"]["pr_metrics"]["pr_auc"]
            ens_trough_auc = ens_eval["trough"]["pr_metrics"]["pr_auc"]

            logger.info(
                f"Test PR-AUC | GBM: peak={gbm_peak_auc:.4f}, trough={gbm_trough_auc:.4f} | "
                f"Ensemble: peak={ens_peak_auc:.4f}, trough={ens_trough_auc:.4f}"
            )
            logger.info(
                f"Delta      | peak={ens_peak_auc - gbm_peak_auc:+.4f}, "
                f"trough={ens_trough_auc - gbm_trough_auc:+.4f}"
            )

            with mlflow.start_run(run_name="ensemble_evaluation", nested=True):
                mlflow.log_metrics({
                    "gbm_peak_test_pr_auc": gbm_peak_auc,
                    "gbm_trough_test_pr_auc": gbm_trough_auc,
                    "ensemble_peak_test_pr_auc": ens_peak_auc,
                    "ensemble_trough_test_pr_auc": ens_trough_auc,
                    "delta_peak": ens_peak_auc - gbm_peak_auc,
                    "delta_trough": ens_trough_auc - gbm_trough_auc,
                })


# ── Batch Predict ────────────────────────────────────────


def run_batch_predict(
    markets: list[str],
    model_type: str = "gbm",
    threshold: float = 0.5,
    label_config: str | None = None,
    model_config: str | None = None,
    timeframe: str = "1m",
) -> None:
    """Run batch prediction for all symbols/dates in featured data."""
    from src.inference.predict import predict_all

    model_types = ["gbm", "lstm", "ensemble"] if model_type == "all" else [model_type]

    for mt in model_types:
        for market in markets:
            logger.info(f"=== Batch predicting {market} [{label_config}/{model_config}] model={mt} (timeframe={timeframe}) ===")
            try:
                result = predict_all(
                    market=market,
                    model_type=mt,
                    threshold=threshold,
                    label_config=label_config,
                    model_config=model_config,
                    timeframe=timeframe,
                )
                n_peaks = int((result["label"] == 1).sum())
                n_troughs = int((result["label"] == 2).sum())
                logger.info(
                    f"Done {market}/{mt}: {len(result)} rows, "
                    f"peaks={n_peaks}, troughs={n_troughs}"
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Batch predict failed for {market}/{mt}: {e}")


# ── Predict ──────────────────────────────────────────────


def run_predict(
    markets: list[str],
    symbols: list[str],
    model_type: str = "gbm",
    threshold: float = 0.5,
    date: str | None = None,
    label_config: str | None = None,
    model_config: str | None = None,
    timeframe: str = "1m",
) -> None:
    """Run inference for specified symbols."""
    from src.inference.predict import predict_symbol

    for market in markets:
        for symbol in symbols:
            logger.info(f"=== Predicting {market}/{symbol} [{label_config}/{model_config}] (timeframe={timeframe}) ===")
            try:
                result = predict_symbol(
                    market=market,
                    symbol=symbol,
                    model_type=model_type,
                    threshold=threshold,
                    date=date,
                    label_config=label_config,
                    model_config=model_config,
                    timeframe=timeframe,
                )
                logger.info(
                    f"Done: {result['date']}, "
                    f"{sum(p['n_peaks'] for p in result['predictions'])} peaks, "
                    f"{sum(p['n_troughs'] for p in result['predictions'])} troughs"
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Predict failed for {market}/{symbol}: {e}")


# ── Trade ────────────────────────────────────────────────


def run_trade(
    market: str,
    symbols: list[str],
    model_type: str = "gbm",
    threshold: float = 0.5,
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    quantity: int = 1,
    broker_type: str = "mock",
    label_config: str = "L2",
    model_config: str = "M3",
    timeframe: str = "1m",
) -> None:
    """Run trading simulation for one or more symbols.

    Single-day: --date YYYY-MM-DD (or latest if omitted and no --date-from)
    Multi-day:  --date-from YYYY-MM-DD [--date-to YYYY-MM-DD]
    """
    from src.trading.broker.mock_broker import MockBroker
    from src.trading.broker.historical_broker import HistoricalBroker
    from src.trading.datafeed.historical_feed import HistoricalDataFeed
    from src.trading.engine import TradingEngine
    from src.trading.notifier.console import ConsoleNotifier
    from src.trading.signal_detector import SignalDetector
    from src.trading.trade_tracker import TradeTracker

    # Determine replay dates
    if date_from:
        early_data = {}
        all_dates = None
        for symbol in symbols:
            early_df, dates = HistoricalDataFeed.get_available_dates(
                market, symbol, date_from=date_from, date_to=date_to,
            )
            early_data[symbol] = early_df
            if all_dates is None:
                all_dates = set(dates)
            else:
                all_dates &= set(dates)
        replay_dates = sorted(all_dates)
    elif date:
        replay_dates = [date]
        early_data = {s: None for s in symbols}
    else:
        replay_dates = [None]  # None = latest available
        early_data = {s: None for s in symbols}

    if not replay_dates:
        logger.warning("No trading dates found in the given range")
        return

    # Shared state across all days
    if broker_type == "historical":
        broker = HistoricalBroker(market=market)
        broker.load_symbols(symbols)
    else:
        broker = MockBroker()

    detector = SignalDetector(
        market=market, model_type=model_type, threshold=threshold,
        label_config=label_config, model_config=model_config,
    )
    tracker = TradeTracker()

    multi_day = date_from is not None
    if multi_day:
        print(f"\n=== Multi-day backtest: {len(replay_dates)} days ===")
        print(f"Period: {replay_dates[0]} ~ {replay_dates[-1]}")
        print(f"Symbols: {', '.join(symbols)}\n")

    for day_date in replay_dates:
        feeds = {
            symbol: HistoricalDataFeed(
                market=market, symbol=symbol, date=day_date,
                early_df=early_data.get(symbol),
            )
            for symbol in symbols
        }

        engine = TradingEngine(
            feeds=feeds,
            broker=broker,
            detector=detector,
            symbols=symbols,
            quantity=quantity,
            notifiers=[ConsoleNotifier()] if not multi_day else [],
            tracker=tracker,
        )
        result = engine.run()

        if multi_day:
            for sym in symbols:
                s = result.get(sym, {})
                buys, sells = s.get("buys", 0), s.get("sells", 0)
                pnl = s.get("net_pnl", 0)
                cash = s.get("cash_balance", 0)
                print(
                    f"  [{day_date}] {sym}: "
                    f"buys={buys} sells={sells} pnl={pnl:+,.0f} cash={cash:,.0f}"
                )

    # Save tracker & print summary
    if tracker:
        date_label = f"{replay_dates[0]}_{replay_dates[-1]}" if multi_day else (replay_dates[0] or "latest")
        tracker.save(date_label)
        summary = tracker.summary()
        if summary:
            ret = summary.get("return_pct", 0)
            mdd = summary.get("max_drawdown_pct", 0)
            print(f"\n=== Backtest Summary ===")
            print(f"Period: {replay_dates[0] or 'latest'} ~ {replay_dates[-1] or 'latest'} ({len(replay_dates)} days)")
            print(f"Return: {ret:+.2%} | Max drawdown: {mdd:.2%}")
            print(f"Final equity: {summary.get('final_equity', 0):,.0f}")



# ── Backtest ─────────────────────────────────────────


def list_backtest_symbols(market: str = "us") -> list[str]:
    """List symbols that have options data for backtesting."""
    from config.settings import RAW_OPTIONS_DIR

    options_dir = RAW_OPTIONS_DIR / market
    if not options_dir.exists():
        return []
    return sorted(d.name for d in options_dir.iterdir() if d.is_dir())


def _load_prediction_data(
    market: str,
    symbols: list[str],
    timeframe: str,
    label_config: str,
    model_config: str,
    model_type: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Load prediction data for selected symbols from partitioned parquet files.

    Path: data/predictions/labeled/{model_type}/{timeframe}/{label_config}/{model_config}/{market}/{symbol}/*.parquet
    """
    pred_base = PREDICTIONS_DIR / model_type / timeframe / label_config / model_config / market
    all_dfs: list[pd.DataFrame] = []

    for symbol in symbols:
        symbol_dir = pred_base / symbol
        if not symbol_dir.exists():
            logger.warning(f"No prediction data for {symbol} at {symbol_dir}, skipping")
            continue

        parquet_files = sorted(symbol_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files for {symbol} at {symbol_dir}, skipping")
            continue

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            df["symbol"] = symbol
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    pred_df = pd.concat(all_dfs, ignore_index=True)
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])

    # Filter by date range (strip tz for comparison)
    if date_from or date_to:
        dt_col = pred_df["datetime"]
        if dt_col.dt.tz is not None:
            dt_naive = dt_col.dt.tz_localize(None)
        else:
            dt_naive = dt_col
        if date_from:
            pred_df = pred_df[dt_naive >= pd.Timestamp(date_from)]
            dt_naive = dt_naive[pred_df.index]
        if date_to:
            pred_df = pred_df[dt_naive <= pd.Timestamp(date_to) + pd.Timedelta(days=1)]

    return pred_df


def run_backtest(
    market: str,
    symbols: list[str],
    timeframe: str = "1m",
    label_config: str = "L2",
    model_config: str = "M3",
    model_type: str = "gbm",
    threshold: float = 0.3,
    tp_pct: float = 0.10,
    sl_pct: float = -0.05,
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    """Run backtest for specified symbols using historical options data."""
    from config.settings import KR_SESSION_MINUTES, RAW_OPTIONS_DIR, US_SESSION_MINUTES
    from src.backtest.analyzer import Analyzer
    from src.backtest.engine import BacktestEngine
    from src.backtest.executor.backtest import BacktestExecutor
    from src.backtest.strategy import Strategy, StrategyConfig

    # 1. Load prediction data
    pred_df = _load_prediction_data(
        market, symbols, timeframe, label_config, model_config, model_type,
        date_from, date_to,
    )
    if pred_df.empty:
        logger.error("No prediction data found. Run batch_predict first.")
        return

    # 2. Check which symbols have options data
    symbols_with_options = []
    for symbol in symbols:
        options_dir = RAW_OPTIONS_DIR / market / symbol
        if options_dir.exists() and any(options_dir.iterdir()):
            symbols_with_options.append(symbol)
        else:
            logger.warning(f"No options data for {symbol}, skipping")

    if not symbols_with_options:
        logger.error("No symbols with options data found.")
        return

    # Filter pred_df to only symbols with options data
    pred_df = pred_df[pred_df["symbol"].isin(symbols_with_options)]
    if pred_df.empty:
        logger.error("No prediction data for symbols with options data.")
        return

    # 3. Create StrategyConfig
    config = StrategyConfig(
        threshold=threshold,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )

    # 4. Create BacktestExecutor and load data
    executor = BacktestExecutor(symbols=symbols_with_options, market=market)
    executor.load_data()

    # 5. Create BacktestEngine
    strategy = Strategy(config)
    engine = BacktestEngine(strategy, executor)

    # 6. Run backtest
    session_minutes = KR_SESSION_MINUTES if market == "kr" else US_SESSION_MINUTES
    result = engine.run(pred_df, market, session_minutes)

    # 7. Compute metrics
    analyzer = Analyzer()
    metrics = analyzer.compute_metrics(result)

    # 8. Print summary
    date_min = pred_df["datetime"].min().strftime("%Y-%m-%d")
    date_max = pred_df["datetime"].max().strftime("%Y-%m-%d")
    symbols_str = ", ".join(symbols_with_options)

    print(f"\n=== Backtest: {market} / {symbols_str} [{timeframe} {label_config}/{model_config}] ===")
    print(f"Period: {date_min} ~ {date_max}")
    print(f"Strategy: threshold={threshold}, TP={tp_pct:+.0%}, SL={sl_pct:+.0%}")
    print()
    print("Results:")
    print(f"  Total Return:    {metrics['total_return']:+.1%}")
    wins = len([t for t in result.trades if t.pnl > 0])
    total = metrics["total_trades"]
    print(f"  Win Rate:        {metrics['win_rate']:.1%} ({wins}/{total})")
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Profit Factor:   {pf_str}")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:+.1%}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Total Trades:    {total}")
    print(f"  Avg Holding:     {metrics['avg_holding_minutes']:.0f} min")

    # 9. Save result
    results_dir = DATA_DIR / "backtest" / "results" / market / f"{label_config}_{model_config}" / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    result.save(results_dir)
    print(f"\nSaved to: {results_dir}")


def run_backtest_grid(
    market: str,
    symbols: list[str],
    timeframe: str = "1m",
    label_config: str = "L2",
    model_config: str = "M3",
    model_type: str = "gbm",
    thresholds: list[float] | None = None,
    tp_pcts: list[float] | None = None,
    sl_pcts: list[float] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    """Run backtest grid search over strategy parameter combinations."""
    from config.settings import KR_SESSION_MINUTES, RAW_OPTIONS_DIR, US_SESSION_MINUTES
    from src.backtest.analyzer import Analyzer
    from src.backtest.engine import BacktestEngine
    from src.backtest.executor.backtest import BacktestExecutor
    from src.backtest.strategy import Strategy, StrategyConfig

    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4]
    if tp_pcts is None:
        tp_pcts = [0.05, 0.10, 0.15]
    if sl_pcts is None:
        sl_pcts = [-0.03, -0.05, -0.10]

    # 1. Load prediction data
    pred_df = _load_prediction_data(
        market, symbols, timeframe, label_config, model_config, model_type,
        date_from, date_to,
    )
    if pred_df.empty:
        logger.error("No prediction data found. Run batch_predict first.")
        return

    # Check which symbols have options data
    symbols_with_options = []
    for symbol in symbols:
        options_dir = RAW_OPTIONS_DIR / market / symbol
        if options_dir.exists() and any(options_dir.iterdir()):
            symbols_with_options.append(symbol)
        else:
            logger.warning(f"No options data for {symbol}, skipping")

    if not symbols_with_options:
        logger.error("No symbols with options data found.")
        return

    pred_df = pred_df[pred_df["symbol"].isin(symbols_with_options)]
    if pred_df.empty:
        logger.error("No prediction data for symbols with options data.")
        return

    # 2. Generate configs
    configs = []
    for thr in thresholds:
        for tp in tp_pcts:
            for sl in sl_pcts:
                configs.append(StrategyConfig(threshold=thr, tp_pct=tp, sl_pct=sl))

    n_combos = len(configs)
    symbols_str = ", ".join(symbols_with_options)
    print(f"\n=== Backtest Grid: {market} / {symbols_str} [{timeframe} {label_config}/{model_config}] ===")
    print(f"{n_combos} combinations to test\n")

    # 3. Create executor and engine
    executor = BacktestExecutor(symbols=symbols_with_options, market=market)
    executor.load_data()

    strategy = Strategy(configs[0])
    engine = BacktestEngine(strategy, executor)

    # 4. Run grid
    session_minutes = KR_SESSION_MINUTES if market == "kr" else US_SESSION_MINUTES
    results = engine.run_grid(pred_df, market, configs, session_minutes)

    # 5. Compare results
    analyzer = Analyzer()
    comparison_df = analyzer.compare(results)

    # 6. Print comparison table
    if not comparison_df.empty:
        # Format for display
        fmt_df = comparison_df.copy()
        fmt_df["return"] = fmt_df["total_return"].apply(lambda x: f"{x:+.1%}")
        fmt_df["win_rate"] = fmt_df["win_rate"].apply(lambda x: f"{x:.1%}")
        fmt_df["mdd"] = fmt_df["max_drawdown_pct"].apply(lambda x: f"{x:+.1%}")
        fmt_df["sharpe"] = fmt_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
        fmt_df["pf"] = fmt_df["profit_factor"].apply(
            lambda x: f"{x:.2f}" if x != float("inf") else "inf"
        )
        fmt_df["trades"] = fmt_df["total_trades"].astype(int)

        display_cols = ["threshold", "tp_pct", "sl_pct", "return", "win_rate", "mdd", "sharpe", "pf", "trades"]
        print(fmt_df[display_cols].to_string(index=False))

        # Best by return
        best = comparison_df.iloc[0]
        print(
            f"\nBest by return: threshold={best['threshold']}, "
            f"tp={best['tp_pct']}, sl={best['sl_pct']} "
            f"-> {best['total_return']:+.1%}"
        )

    # 7. Save grid results
    results_dir = DATA_DIR / "backtest" / "grid" / market / f"{label_config}_{model_config}" / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    if not comparison_df.empty:
        comparison_df.to_parquet(results_dir / "grid_results.parquet", index=False)
        comparison_df.to_csv(results_dir / "grid_results.csv", index=False)
    # Save best result
    if results:
        results[0].save(results_dir / "best")
    print(f"\nSaved to: {results_dir}")


# ── CLI ──────────────────────────────────────────────────


STAGES = {
    "collector": "Data collection (yfinance → tvDatafeed)",
    "labeler": "Peak/trough labeling",
    "features": "Feature engineering",
    "model": "Model training & evaluation",
    "ensemble": "LSTM calibration + GBM/LSTM ensemble (US only)",
    "predict": "Inference on latest data",
    "batch_predict": "Batch prediction (all symbols → labeled parquet)",
    "trade": "Mock trading simulation",
    "backtest": "Options backtest with historical OHLCV",
    "backtest_grid": "Grid search over backtest strategy parameters",
    "all": "Full pipeline (collector → labeler → features → model)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="option-meme ML pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ./optionmeme all                      Full pipeline (all markets)
  ./optionmeme collector --market kr    KR data collection (incremental)
  ./optionmeme collector --full         Full re-collection
  ./optionmeme labeler --market us      US labeling
  ./optionmeme features                 Feature engineering (all markets)
  ./optionmeme model --model gbm       LightGBM training only
  ./optionmeme all --market kr          KR full pipeline
  ./optionmeme predict --market kr --symbol 5930 --model gbm
  ./optionmeme predict --market us --symbol AAPL --model all --threshold 0.3
  ./optionmeme predict --market kr --symbol 5930 --date 2026-02-20
  ./optionmeme trade --market kr --symbol 5930 --model gbm
  ./optionmeme trade --market kr --symbol 5930 660 --model gbm
  ./optionmeme trade --market kr --symbol 5930 --model gbm --quantity 2
  ./optionmeme trade --market kr --symbol 5930 --date 2026-02-19
  ./optionmeme batch_predict --market all --model gbm --threshold 0.3
  ./optionmeme labeler --label-config L2
  ./optionmeme features --label-config L2 --model-config M3
  ./optionmeme model --label-config L2 --model-config M3
  ./optionmeme batch_predict --label-config all --model-config all
  ./optionmeme all --timeframe 5m                  5-minute pipeline
  ./optionmeme labeler --timeframe 5m --label-config L1
  ./optionmeme backtest --market us --symbol AAPL --threshold 0.3
  ./optionmeme backtest --market us --symbol AAPL SPY --tp 0.15 --sl -0.03
  ./optionmeme backtest --list-symbols --market us
  ./optionmeme backtest_grid --market us --symbol AAPL
  ./optionmeme backtest_grid --market us --symbol AAPL --threshold-grid 0.2,0.3 --tp-grid 0.05,0.10
""",
    )

    parser.add_argument(
        "stage",
        choices=list(STAGES.keys()),
        help="Pipeline stage to run: " + ", ".join(
            f"{k} ({v})" for k, v in STAGES.items()
        ),
    )
    parser.add_argument(
        "--market",
        choices=["kr", "us", "all"],
        default="all",
        help="Market to process (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["gbm", "lstm", "ensemble", "all"],
        default="all",
        dest="model_type",
        help="Model type to train/predict (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full re-collection instead of incremental (collector only)",
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        help="Specific symbol(s). Required for predict. e.g. --symbol 5930 660",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for signal detection (predict only, default: 0.5)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date YYYY-MM-DD (predict/trade, default: latest trading day)",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Number of option contracts per trade (trade only, default: 1)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Backtest start date YYYY-MM-DD (trade only, enables multi-day mode)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="Backtest end date YYYY-MM-DD (trade only, default: latest)",
    )
    parser.add_argument(
        "--broker",
        choices=["mock", "historical"],
        default="mock",
        help="Broker type for trading: mock (Black-Scholes) or historical (real OHLCV) (default: mock)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization for LightGBM (model stage only)",
    )
    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1m",
        help="Timeframe for the pipeline: 1m or 5m (default: 1m)",
    )
    # Label choices: union of all timeframe configs (L1, L2, L3)
    _all_label_keys = sorted(set(LABEL_CONFIGS.keys()) | set(get_label_configs("5m").keys()))
    parser.add_argument(
        "--label-config",
        choices=_all_label_keys + ["all"],
        default="all",
        help="Label configuration variant (default: all)",
    )
    _all_model_keys = sorted(set(MODEL_CONFIGS.keys()) | set(get_model_configs("5m").keys()))
    parser.add_argument(
        "--model-config",
        choices=_all_model_keys + ["all"],
        default="all",
        help="Model configuration variant (default: all)",
    )

    # ── Backtest arguments ──────────────────────────────
    parser.add_argument(
        "--tp",
        type=float,
        default=0.10,
        help="Take profit percentage (backtest, default: 0.10)",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=-0.05,
        help="Stop loss percentage (backtest, default: -0.05)",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List symbols with options data (backtest only)",
    )
    # Grid search arguments
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="0.2,0.3,0.4",
        help="Comma-separated thresholds for grid search (backtest_grid, default: 0.2,0.3,0.4)",
    )
    parser.add_argument(
        "--tp-grid",
        type=str,
        default="0.05,0.10,0.15",
        help="Comma-separated TP values for grid search (backtest_grid, default: 0.05,0.10,0.15)",
    )
    parser.add_argument(
        "--sl-grid",
        type=str,
        default="-0.03,-0.05,-0.10",
        help="Comma-separated SL values for grid search (backtest_grid, default: -0.03,-0.05,-0.10)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    markets = _resolve_markets(args.market)
    stage = args.stage
    timeframe = args.timeframe
    label_configs = _resolve_label_configs(args.label_config, timeframe)
    model_configs = _resolve_model_configs(args.model_config, timeframe)

    logger.info(
        f"Pipeline start: stage={stage}, markets={markets}, timeframe={timeframe}, "
        f"label_configs={label_configs}, model_configs={model_configs}"
    )
    start_time = datetime.now()

    try:
        if stage in ("collector", "all"):
            run_collector(markets, full=args.full, symbols=args.symbol, timeframe=timeframe)

        if stage in ("labeler", "all"):
            for lc in label_configs:
                run_labeler(markets, label_config=lc, timeframe=timeframe)

        if stage in ("features", "all"):
            for lc in label_configs:
                for mc in model_configs:
                    run_features(markets, label_config=lc, model_config=mc, timeframe=timeframe)

        if stage in ("model", "all"):
            for lc in label_configs:
                for mc in model_configs:
                    run_model(
                        markets,
                        model_type=args.model_type,
                        label_config=lc,
                        model_config=mc,
                        optimize=args.optimize,
                        timeframe=timeframe,
                    )

        if stage == "ensemble":
            for lc in label_configs:
                for mc in model_configs:
                    run_ensemble(markets, label_config=lc, model_config=mc, timeframe=timeframe)

        if stage == "batch_predict":
            for lc in label_configs:
                for mc in model_configs:
                    run_batch_predict(
                        markets,
                        model_type=args.model_type,
                        threshold=args.threshold,
                        label_config=lc,
                        model_config=mc,
                        timeframe=timeframe,
                    )

        if stage == "predict":
            if not args.symbol:
                parser.error("predict stage requires --symbol")
            for lc in label_configs:
                for mc in model_configs:
                    run_predict(
                        markets,
                        symbols=args.symbol,
                        model_type=args.model_type,
                        threshold=args.threshold,
                        date=args.date,
                        label_config=lc,
                        model_config=mc,
                        timeframe=timeframe,
                    )

        if stage == "trade":
            if not args.symbol:
                parser.error("trade stage requires --symbol")
            if args.market == "all":
                parser.error("trade stage requires a specific --market (kr or us)")
            model_type = args.model_type if args.model_type != "all" else "gbm"
            label_config = args.label_config if args.label_config != "all" else "L2"
            model_config = args.model_config if args.model_config != "all" else "M3"
            run_trade(
                market=args.market,
                symbols=args.symbol,
                model_type=model_type,
                threshold=args.threshold,
                date=args.date,
                date_from=args.date_from,
                date_to=args.date_to,
                quantity=args.quantity,
                broker_type=args.broker,
                label_config=label_config,
                model_config=model_config,
                timeframe=timeframe,
            )

        if stage == "backtest":
            if args.list_symbols:
                bt_market = args.market if args.market != "all" else "us"
                symbols = list_backtest_symbols(bt_market)
                print(f"Symbols with options data ({bt_market}): {', '.join(symbols)}")
                return
            if not args.symbol:
                parser.error("backtest requires --symbol")
            if args.market == "all":
                parser.error("backtest requires a specific --market (kr or us)")
            model_type = args.model_type if args.model_type != "all" else "gbm"
            label_config = args.label_config if args.label_config != "all" else "L2"
            model_config = args.model_config if args.model_config != "all" else "M3"
            run_backtest(
                market=args.market,
                symbols=args.symbol,
                timeframe=timeframe,
                label_config=label_config,
                model_config=model_config,
                model_type=model_type,
                threshold=args.threshold,
                tp_pct=args.tp,
                sl_pct=args.sl,
                date_from=args.date_from,
                date_to=args.date_to,
            )

        if stage == "backtest_grid":
            if not args.symbol:
                parser.error("backtest_grid requires --symbol")
            if args.market == "all":
                parser.error("backtest_grid requires a specific --market (kr or us)")
            model_type = args.model_type if args.model_type != "all" else "gbm"
            label_config = args.label_config if args.label_config != "all" else "L2"
            model_config = args.model_config if args.model_config != "all" else "M3"
            thresholds = [float(x) for x in args.threshold_grid.split(",")]
            tp_pcts = [float(x) for x in args.tp_grid.split(",")]
            sl_pcts = [float(x) for x in args.sl_grid.split(",")]
            run_backtest_grid(
                market=args.market,
                symbols=args.symbol,
                timeframe=timeframe,
                label_config=label_config,
                model_config=model_config,
                model_type=model_type,
                thresholds=thresholds,
                tp_pcts=tp_pcts,
                sl_pcts=sl_pcts,
                date_from=args.date_from,
                date_to=args.date_to,
            )

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    elapsed = datetime.now() - start_time
    logger.info(f"Pipeline complete in {elapsed}")


if __name__ == "__main__":
    main()
