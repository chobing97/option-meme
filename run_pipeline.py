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
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import DATA_DIR, LABELED_DIR, PREDICTIONS_DIR, PROCESSED_DIR


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


def run_collector(markets: list[str], full: bool = False, symbols: list[str] | None = None) -> None:
    """Run data collection for specified markets.

    Default (incremental): fetch from last collected date to today.
    --full: fetch entire available history (~60 days).
    --symbol: collect specific symbol(s) only.
    """
    from src.collector.bar_fetcher import BarFetcher, fetch_yfinance, load_symbol_list
    from src.collector.storage import get_symbol_date_range

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
                    # Full collection: let collect_single handle everything
                    fetcher.collect_single(ticker, exchange, market)
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
                                from src.collector.storage import save_bars

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


# ── Labeler ──────────────────────────────────────────────


def run_labeler(markets: list[str]) -> None:
    """Run labeling for all symbols in specified markets."""
    from src.labeler.label_generator import (
        apply_manual_overrides,
        label_all_symbols,
        label_statistics,
        save_labeled,
    )

    for market in markets:
        logger.info(f"=== Labeling {market} ===")
        labeled_df = label_all_symbols(market, save=True)

        if labeled_df.empty:
            logger.warning(f"No data to label for {market}")
            continue

        # 수작업 레이블 오버라이드 적용 후 재저장
        labeled_df = apply_manual_overrides(labeled_df, market)
        save_labeled(labeled_df, market)

        stats = label_statistics(labeled_df)
        logger.info(f"Label statistics for {market}:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")


# ── Features ─────────────────────────────────────────────


def run_features(markets: list[str]) -> None:
    """Build features from labeled data."""
    from src.features.feature_pipeline import (
        build_features,
        build_lookback_features,
        clean_features,
        get_feature_columns,
    )
    from src.labeler.label_generator import load_labeled

    featured_dir = PROCESSED_DIR / "featured"
    featured_dir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        logger.info(f"=== Building features for {market} ===")
        df = load_labeled(market)

        if df.empty:
            logger.warning(f"No labeled data for {market}, skipping")
            continue

        logger.info(f"Loaded {len(df)} labeled bars")

        df = build_features(df)
        df = build_lookback_features(df)
        df = clean_features(df)

        feature_cols = get_feature_columns(df)
        logger.info(f"Total features: {len(feature_cols)}")

        output_path = featured_dir / f"{market}_featured.parquet"
        df.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"Saved featured data to {output_path} ({len(df)} rows)")


# ── Model ────────────────────────────────────────────────


def run_model(markets: list[str], model_type: str = "all") -> None:
    """Train and evaluate models."""
    from src.features.feature_pipeline import get_all_feature_columns
    from src.model.dataset import SplitResult, prepare_xy, time_based_split
    from src.model.evaluate import full_evaluation

    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        logger.info(f"=== Training models for {market} ===")

        featured_path = PROCESSED_DIR / "featured" / f"{market}_featured.parquet"
        if not featured_path.exists():
            logger.warning(f"No featured data at {featured_path}, skipping")
            continue

        df = pd.read_parquet(featured_path)
        logger.info(f"Loaded {len(df)} featured bars")

        feature_cols = get_all_feature_columns(df)
        logger.info(f"Feature columns: {len(feature_cols)}")

        split = time_based_split(df)
        logger.info(
            f"Split: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}"
        )

        # Track probabilities for full_evaluation
        peak_proba_test = None
        trough_proba_test = None

        for target_label, label_name in [(1, "peak"), (2, "trough")]:
            logger.info(f"--- Target: {label_name} (label={target_label}) ---")

            # LightGBM
            if model_type in ("gbm", "all"):
                _train_lgb_model(
                    split, target_label, label_name, feature_cols,
                    market, models_dir,
                )

            # LSTM
            if model_type in ("lstm", "all"):
                _train_lstm_model(
                    split, target_label, label_name, feature_cols,
                    market, models_dir,
                )

        # Full evaluation with LightGBM predictions (if available)
        if model_type in ("gbm", "all"):
            _run_full_evaluation(split, feature_cols, market, models_dir)


def _train_lgb_model(
    split, target_label, label_name, feature_cols, market, models_dir,
):
    """Train and save a LightGBM model."""
    from src.model.train_gbm import save_model as save_lgb
    from src.model.train_gbm import train_lgb

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


def _train_lstm_model(
    split, target_label, label_name, feature_cols, market, models_dir,
):
    """Train and save an LSTM model."""
    from src.model.train_lstm import save_model as save_lstm
    from src.model.train_lstm import train_lstm

    logger.info(f"Training LSTM for {label_name}...")
    model, metrics = train_lstm(split, target_label, feature_cols)

    model_path = models_dir / f"lstm_{market}_{label_name}.pt"
    save_lstm(model, model_path)

    logger.info(f"LSTM {label_name} metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


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


# ── Batch Predict ────────────────────────────────────────


def run_batch_predict(
    markets: list[str],
    model_type: str = "gbm",
    threshold: float = 0.5,
) -> None:
    """Run batch prediction for all symbols/dates in featured data."""
    from src.inference.predict import predict_all

    for market in markets:
        logger.info(f"=== Batch predicting {market} ===")
        try:
            result = predict_all(
                market=market,
                model_type=model_type,
                threshold=threshold,
            )
            n_peaks = int((result["label"] == 1).sum())
            n_troughs = int((result["label"] == 2).sum())
            logger.info(
                f"Done {market}: {len(result)} rows, "
                f"peaks={n_peaks}, troughs={n_troughs}"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Batch predict failed for {market}: {e}")


# ── Predict ──────────────────────────────────────────────


def run_predict(
    markets: list[str],
    symbols: list[str],
    model_type: str = "gbm",
    threshold: float = 0.5,
    date: str | None = None,
) -> None:
    """Run inference for specified symbols."""
    from src.inference.predict import predict_symbol

    for market in markets:
        for symbol in symbols:
            logger.info(f"=== Predicting {market}/{symbol} ===")
            try:
                result = predict_symbol(
                    market=market,
                    symbol=symbol,
                    model_type=model_type,
                    threshold=threshold,
                    date=date,
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
    quantity: int = 1,
) -> None:
    """Run mock trading simulation for one or more symbols."""
    from src.trading.broker.mock_broker import MockBroker
    from src.trading.datafeed.mock_feed import MockDataFeed
    from src.trading.engine import TradingEngine
    from src.trading.notifier.console import ConsoleNotifier
    from src.trading.signal_detector import SignalDetector
    from src.trading.trade_db import TradeDB

    feeds = {
        symbol: MockDataFeed(market=market, symbol=symbol, date=date)
        for symbol in symbols
    }
    broker = MockBroker()
    detector = SignalDetector(market=market, model_type=model_type, threshold=threshold)
    notifiers = [ConsoleNotifier()]
    trade_db = TradeDB()

    engine = TradingEngine(
        feeds=feeds,
        broker=broker,
        detector=detector,
        symbols=symbols,
        quantity=quantity,
        notifiers=notifiers,
        trade_db=trade_db,
    )
    engine.run()
    trade_db.close()


# ── CLI ──────────────────────────────────────────────────


STAGES = {
    "collector": "Data collection (yfinance → tvDatafeed)",
    "labeler": "Peak/trough labeling",
    "features": "Feature engineering",
    "model": "Model training & evaluation",
    "predict": "Inference on latest data",
    "batch_predict": "Batch prediction (all symbols → labeled parquet)",
    "trade": "Mock trading simulation",
    "all": "Full pipeline (collector → labeler → features → model)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="option-meme ML pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ./run.sh all                      Full pipeline (all markets)
  ./run.sh collector --market kr    KR data collection (incremental)
  ./run.sh collector --full         Full re-collection
  ./run.sh labeler --market us      US labeling
  ./run.sh features                 Feature engineering (all markets)
  ./run.sh model --model gbm       LightGBM training only
  ./run.sh all --market kr          KR full pipeline
  ./run.sh predict --market kr --symbol 5930 --model gbm
  ./run.sh predict --market us --symbol AAPL --model all --threshold 0.3
  ./run.sh predict --market kr --symbol 5930 --date 2026-02-20
  ./run.sh trade --market kr --symbol 5930 --model gbm
  ./run.sh trade --market kr --symbol 5930 660 --model gbm
  ./run.sh trade --market kr --symbol 5930 --model gbm --quantity 2
  ./run.sh trade --market kr --symbol 5930 --date 2026-02-19
  ./run.sh batch_predict --market all --model gbm --threshold 0.3
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
        choices=["gbm", "lstm", "all"],
        default="all",
        dest="model_type",
        help="Model type to train (default: all)",
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    markets = _resolve_markets(args.market)
    stage = args.stage

    logger.info(f"Pipeline start: stage={stage}, markets={markets}")
    start_time = datetime.now()

    try:
        if stage in ("collector", "all"):
            run_collector(markets, full=args.full, symbols=args.symbol)

        if stage in ("labeler", "all"):
            run_labeler(markets)

        if stage in ("features", "all"):
            run_features(markets)

        if stage in ("model", "all"):
            run_model(markets, model_type=args.model_type)

        if stage == "batch_predict":
            run_batch_predict(
                markets,
                model_type=args.model_type,
                threshold=args.threshold,
            )

        if stage == "predict":
            if not args.symbol:
                parser.error("predict stage requires --symbol")
            run_predict(
                markets,
                symbols=args.symbol,
                model_type=args.model_type,
                threshold=args.threshold,
                date=args.date,
            )

        if stage == "trade":
            if not args.symbol:
                parser.error("trade stage requires --symbol")
            if args.market == "all":
                parser.error("trade stage requires a specific --market (kr or us)")
            run_trade(
                market=args.market,
                symbols=args.symbol,
                model_type=args.model_type,
                threshold=args.threshold,
                date=args.date,
                quantity=args.quantity,
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
