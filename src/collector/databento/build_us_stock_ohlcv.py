"""
미국 주식 1분봉 통합 ETL 스크립트

1. Databento 20종목: DBN 다운로드 확인 → 3거래소 합산 → Parquet 저장
2. 나머지 종목: yfinance + tvDatafeed (기존 로직)

사용법:
    # 전체 실행
    python -m src.collector.databento.build_us_stock_ohlcv

    # Databento 종목만
    python -m src.collector.databento.build_us_stock_ohlcv --databento-only

    # 나머지 종목만 (yfinance + TV)
    python -m src.collector.databento.build_us_stock_ohlcv --legacy-only

    # dry-run
    python -m src.collector.databento.build_us_stock_ohlcv --dry-run
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from re import search as re_search

import databento as db
import pandas as pd
from loguru import logger

from config.settings import LOG_DIR, RAW_STOCK_DIR
from src.collector.collection_tracker import CollectionTracker
from src.collector.storage import save_bars


# ── 로그 ──────────────────────────────────────────────────────


class Tee:
    """stdout을 콘솔과 로그 파일 양쪽에 동시 출력."""

    def __init__(self, log_path: Path):
        self._console = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")

    def write(self, data: str) -> int:
        self._console.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        sys.stdout = self._console


def setup_logging() -> Path:
    """로그 파일 설정. 로그 파일 경로 반환."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"build_us_ohlcv_{timestamp}.log"

    # stdout → Tee (콘솔 + 파일)
    tee = Tee(log_path)
    sys.stdout = tee

    # loguru도 같은 파일에 기록
    logger.add(str(log_path), level="DEBUG", mode="a")

    return log_path

# ── 상수 ──────────────────────────────────────────────────────

DATABENTO_DIR = RAW_STOCK_DIR / "databento" / "us"

DATABENTO_SYMBOLS = [
    "TSLA", "AAPL", "NVDA", "AMZN", "AMD", "META", "MSFT", "GOOGL", "PLTR", "MARA", "AMC",
    "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "TLT", "SLV",
    "BAC", "BABA",
]

DATASETS = ["XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR"]

DATABENTO_START = "2020-01-01"


# ── 유틸 ──────────────────────────────────────────────────────


def parse_available_end(error_msg: str) -> str | None:
    """Databento 에러에서 사용 가능 종료일 추출."""
    m = re_search(r"data available up to '(\d{4}-\d{2}-\d{2})", str(error_msg))
    return m.group(1) if m else None


def find_dbn_files(symbol: str) -> dict[str, list[Path]]:
    """심볼의 거래소별 DBN 파일 목록 반환. {dataset: [파일들]}"""
    symbol_dir = DATABENTO_DIR / symbol
    if not symbol_dir.exists():
        return {}
    result = {}
    for ds in DATASETS:
        files = sorted(symbol_dir.glob(f"{symbol}_{ds}_ohlcv-1m_*.dbn.zst"))
        if files:
            result[ds] = files
    return result


def get_dbn_end_date(symbol: str) -> str | None:
    """심볼의 DBN 파일에서 가장 최신 end 날짜 추출."""
    files = find_dbn_files(symbol)
    if not files:
        return None
    latest = None
    for file_list in files.values():
        for f in file_list:
            # 파일명: TSLA_XNAS.ITCH_ohlcv-1m_20200101_20260224.dbn.zst
            parts = f.stem.replace(".dbn", "").split("_")
            end_str = parts[-1]  # 20260224
            if len(end_str) == 8:
                date_str = f"{end_str[:4]}-{end_str[4:6]}-{end_str[6:8]}"
                if latest is None or date_str > latest:
                    latest = date_str
    return latest


# ── Step 1: Databento 다운로드 확인 & 추가 다운로드 ───────────


def check_databento_downloads(dry_run: bool = False, auto_yes: bool = False) -> dict[str, str]:
    """Databento 다운로드 상태 확인. 추가 다운로드 필요 시 비용 확인 후 진행.

    Returns:
        {symbol: end_date} 최종 다운로드 완료 상태
    """
    today = datetime.now().strftime("%Y-%m-%d")
    status = {}
    need_download = []

    print(f"\n=== Databento 다운로드 확인 ({len(DATABENTO_SYMBOLS)}종목) ===")

    for sym in DATABENTO_SYMBOLS:
        end = get_dbn_end_date(sym)
        if end is None:
            print(f"  {sym:<6} 미다운로드 → {DATABENTO_START} ~ {today}")
            need_download.append((sym, DATABENTO_START, today))
        elif end < today:
            # end의 다음날부터 오늘까지 추가 다운로드 필요
            next_day = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  {sym:<6} {end}까지 존재 → {next_day} ~ {today} 추가 필요")
            need_download.append((sym, next_day, today))
        else:
            print(f"  {sym:<6} {end}까지 존재 (최신)")
            status[sym] = end

    if not need_download:
        print("  모든 종목 최신 상태")
        return status

    if dry_run:
        print(f"\n  [dry-run] {len(need_download)}건 다운로드 필요")
        for sym, start, end in need_download:
            print(f"    {sym:<6} {start} ~ {end}")
        return status

    # 비용 확인
    print(f"\n=== 추가 다운로드 비용 확인 ({len(need_download)}종목) ===")
    try:
        client = db.Historical()
    except Exception as e:
        print(f"  Databento 클라이언트 생성 실패: {e}")
        print(f"  DATABENTO_API_KEY 환경변수를 확인하세요.")
        return status

    total_cost = 0.0
    job_costs = []
    for sym, start, end in need_download:
        sym_cost = 0.0
        for ds in DATASETS:
            try:
                cost = client.metadata.get_cost(
                    dataset=ds, symbols=[sym], stype_in="raw_symbol",
                    schema="ohlcv-1m", start=start, end=end,
                )
                sym_cost += cost
            except db.common.error.BentoClientError as e:
                avail = parse_available_end(str(e))
                if avail and "end_after_available" in str(e):
                    try:
                        cost = client.metadata.get_cost(
                            dataset=ds, symbols=[sym], stype_in="raw_symbol",
                            schema="ohlcv-1m", start=start, end=avail,
                        )
                        sym_cost += cost
                    except Exception:
                        pass
                # 기타 에러는 무시 (비용 0으로 처리)
            except Exception:
                pass
        job_costs.append((sym, start, end, sym_cost))
        total_cost += sym_cost
        print(f"  {sym:<6} {start} ~ {end}  ${sym_cost:.4f}")

    print(f"\n  총 예상 비용: ${total_cost:.4f}")

    # 사용자 확인
    if auto_yes:
        print("\n  [--yes] 자동 승인")
    else:
        answer = input(f"\n  다운로드를 진행할까요? (y/n): ").strip().lower()
        if answer != "y":
            print("  다운로드 취소")
            return status

    # 다운로드 실행
    print(f"\n=== Databento 다운로드 시작 ===")
    for sym, start, end in need_download:
        for ds in DATASETS:
            start_compact = start.replace("-", "")
            end_compact = end.replace("-", "")
            filename = f"{sym}_{ds}_ohlcv-1m_{start_compact}_{end_compact}.dbn.zst"
            path = DATABENTO_DIR / sym / filename

            if path.exists():
                print(f"  {sym:<6} {ds:<15} skip")
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  {sym:<6} {ds:<15}", end="")

            actual_end = end
            try:
                data = client.timeseries.get_range(
                    dataset=ds, symbols=[sym], stype_in="raw_symbol",
                    schema="ohlcv-1m", start=start, end=end,
                )
            except db.common.error.BentoClientError as e:
                avail = parse_available_end(str(e))
                if avail and "end_after_available" in str(e) and avail > start:
                    actual_end = avail
                    try:
                        data = client.timeseries.get_range(
                            dataset=ds, symbols=[sym], stype_in="raw_symbol",
                            schema="ohlcv-1m", start=start, end=avail,
                        )
                        # 조정된 end로 파일명 변경
                        end_compact = avail.replace("-", "")
                        filename = f"{sym}_{ds}_ohlcv-1m_{start_compact}_{end_compact}.dbn.zst"
                        path = DATABENTO_DIR / sym / filename
                    except Exception as e2:
                        print(f"  FAIL: {e2}")
                        continue
                else:
                    print(f"  FAIL: {e}")
                    continue
            except Exception as e:
                print(f"  FAIL: {e}")
                continue

            data.to_file(str(path))
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ok ({size_mb:.1f} MB)")

        status[sym] = get_dbn_end_date(sym) or end

    return status


# ── Step 2: DBN → Parquet 변환 (3거래소 합산) ─────────────────


def read_dbn_to_df(path: Path) -> pd.DataFrame:
    """DBN 파일을 OHLCV DataFrame으로 변환.

    Databento ohlcv-1m 구조:
    - 인덱스: ts_event (DatetimeIndex, UTC)
    - 컬럼: open, high, low, close (float64), volume (uint64)
    - UTC → US/Eastern 변환 후 tz 제거
    """
    store = db.DBNStore.from_file(str(path))
    df = store.to_df()
    if df.empty:
        return pd.DataFrame()

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    # 인덱스(ts_event, UTC) → US/Eastern → tz 제거
    df = df.reset_index()
    ts_col = df.columns[0]  # ts_event
    df = df.rename(columns={ts_col: "datetime"})
    df["datetime"] = (
        pd.to_datetime(df["datetime"])
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    df = df[["datetime"] + required].copy()
    df["volume"] = df["volume"].astype("int64")

    return df


def merge_exchange_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """3거래소 DataFrame을 분 단위로 합산.

    open=first, high=max, low=min, close=last, volume=sum
    """
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["open", "high", "low", "close"])
    combined = combined[combined["volume"] > 0]

    if combined.empty:
        return pd.DataFrame()

    # 분 단위로 truncate
    combined["minute"] = combined["datetime"].dt.floor("min")

    merged = combined.groupby("minute").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    merged = merged.rename(columns={"minute": "datetime"})
    merged = merged.sort_values("datetime").reset_index(drop=True)

    return merged


def _find_legacy_data(symbols: list[str], tracker: CollectionTracker) -> dict[str, list[dict]]:
    """Databento 종목 중 yfinance/TV 데이터가 존재하는 종목 탐색.

    Returns:
        {symbol: [tracker_entries]} yfinance/TV 소스의 tracker 항목
    """
    legacy_sources = {"yfinance", "tvdatafeed", "combined"}
    result = {}
    for sym in symbols:
        entries = tracker.get_entries_for_symbol(sym, "us")
        legacy = [e for e in entries if e["source"] in legacy_sources]
        if legacy:
            result[sym] = legacy
    return result


def _purge_legacy_parquets(symbol: str, tracker: CollectionTracker, legacy_entries: list[dict]):
    """심볼의 기존 Parquet 파일과 legacy tracker 항목 삭제."""
    # Parquet 파일 삭제
    symbol_dir = RAW_STOCK_DIR / "us" / symbol
    if symbol_dir.exists():
        for pf in symbol_dir.glob("*.parquet"):
            pf.unlink()

    # legacy tracker 항목 삭제
    for entry in legacy_entries:
        tracker.delete_entry(symbol, entry["exchange"], entry["source"])

    # databento tracker 항목도 삭제 (재변환을 위해)
    db_entry = tracker.get_status(symbol, "MULTI", "databento")
    if db_entry:
        tracker.delete_entry(symbol, "MULTI", "databento")


def convert_databento_to_parquet(
    symbols: list[str],
    tracker: CollectionTracker,
    dry_run: bool = False,
    auto_yes: bool = False,
):
    """Databento DBN 파일을 Parquet으로 변환."""
    print(f"\n=== DBN → Parquet 변환 ({len(symbols)}종목) ===")

    # yfinance/TV 데이터가 있는 종목 확인 → 교체 제안
    has_dbn = [s for s in symbols if find_dbn_files(s)]
    legacy_map = _find_legacy_data(has_dbn, tracker)
    overwrite_syms: set[str] = set()

    if legacy_map:
        print(f"\n  다음 {len(legacy_map)}종목에 yfinance/TV 데이터가 존재합니다:")
        for sym, entries in legacy_map.items():
            sources = ", ".join(f"{e['source']}({e.get('end_date', '?')[:10]})" for e in entries)
            print(f"    {sym:<6} {sources}")

        if not dry_run:
            if auto_yes:
                print("\n  [--yes] 자동 승인")
                answer = "y"
            else:
                answer = input("\n  Databento 데이터로 교체할까요? (y/n): ").strip().lower()
            if answer == "y":
                overwrite_syms = set(legacy_map.keys())
                for sym in overwrite_syms:
                    _purge_legacy_parquets(sym, tracker, legacy_map[sym])
                print(f"  {len(overwrite_syms)}종목 기존 데이터 삭제 완료 → Databento로 재변환")
            else:
                print("  기존 데이터 유지 (Databento 데이터와 병합)")
        else:
            print(f"  [dry-run] 교체 대상 {len(legacy_map)}종목")

        print()

    for sym in symbols:
        files_by_ds = find_dbn_files(sym)
        if not files_by_ds:
            print(f"  {sym:<6} DBN 파일 없음, 스킵")
            continue

        # 교체 대상이 아닌 경우에만 스킵 검사
        if sym not in overwrite_syms:
            dbn_end = get_dbn_end_date(sym)
            existing = tracker.get_status(sym, "MULTI", "databento")
            if existing and existing["status"] == "complete" and existing.get("end_date") and dbn_end:
                existing_end = existing["end_date"][:10]
                # dbn_end는 exclusive end date → 실제 데이터 마지막일은 dbn_end - 1일
                dbn_data_end = (
                    datetime.strptime(dbn_end, "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if existing_end >= dbn_data_end:
                    print(f"  {sym:<6} 변환 완료 ({existing_end}까지), 스킵")
                    continue

        if dry_run:
            total_files = sum(len(v) for v in files_by_ds.values())
            label = " [교체]" if sym in overwrite_syms else ""
            print(f"  {sym:<6} {total_files}개 DBN 파일 → 변환 예정{label}")
            continue

        # 모든 거래소의 모든 DBN 파일 읽기
        all_dfs = []
        total_files = 0
        for ds, file_list in files_by_ds.items():
            for f in file_list:
                try:
                    df = read_dbn_to_df(f)
                    if not df.empty:
                        all_dfs.append(df)
                        total_files += 1
                except Exception as e:
                    logger.warning(f"  {sym} {f.name} 읽기 실패: {e}")

        if not all_dfs:
            print(f"  {sym:<6} 읽을 수 있는 데이터 없음")
            continue

        # 3거래소 합산
        merged = merge_exchange_dfs(all_dfs)
        if merged.empty:
            print(f"  {sym:<6} 합산 결과 빈 DataFrame")
            continue

        # Parquet 저장
        saved = save_bars(merged, market="us", symbol=sym, source="databento")
        total_bars = sum(saved.values())

        # 트래커 업데이트
        tracker.upsert(
            symbol=sym,
            exchange="MULTI",
            market="us",
            source="databento",
            start_date=str(merged["datetime"].min()),
            end_date=str(merged["datetime"].max()),
            bar_count=total_bars,
            status="complete",
        )

        years = sorted(saved.keys())
        label = " (교체)" if sym in overwrite_syms else ""
        print(f"  {sym:<6} {total_files} DBN → {total_bars:,} bars ({years[0]}~{years[-1]}){label}")


# ── Step 3: 나머지 종목 yfinance + tvDatafeed ─────────────────


def collect_legacy_symbols(
    tracker: CollectionTracker,
    dry_run: bool = False,
    fallback_symbols: list[str] | None = None,
):
    """Databento 제외 종목 + fallback 종목을 yfinance + tvDatafeed로 수집.

    Args:
        fallback_symbols: Databento 데이터가 없는 종목 (다운로드 거부 등)
    """
    from src.collector.bar_fetcher import BarFetcher, load_symbol_list

    # 전체 US 종목 로드
    stocks_df = load_symbol_list("us_stocks")
    etfs_df = load_symbol_list("us_etf_index")
    all_symbols = pd.concat([stocks_df, etfs_df], ignore_index=True)

    # Databento 종목 중 fallback 대상은 제외하지 않음
    fallback_set = set(fallback_symbols or [])
    exclude_set = set(DATABENTO_SYMBOLS) - fallback_set
    legacy = all_symbols[~all_symbols["ticker"].isin(exclude_set)]

    if fallback_set:
        print(f"\n=== yfinance + tvDatafeed 수집 ({len(legacy)}종목) ===")
        print(f"  (Databento 완료 {len(exclude_set)}종목 제외, fallback {len(fallback_set)}종목 포함)")
    else:
        print(f"\n=== yfinance + tvDatafeed 수집 ({len(legacy)}종목) ===")
        print(f"  (Databento {len(exclude_set)}종목 제외)")

    if dry_run:
        for _, row in legacy.iterrows():
            ticker = str(row["ticker"])
            existing = tracker.get_status(ticker, row.get("exchange", ""), "yfinance")
            end = existing["end_date"][:10] if existing and existing.get("end_date") else "없음"
            print(f"  {ticker:<6} 마지막 수집: {end}")
        return

    fetcher = BarFetcher()
    total = len(legacy)

    for i, (_, row) in enumerate(legacy.iterrows()):
        ticker = str(row["ticker"])
        exchange = row.get("exchange", "")
        tv_symbol = row.get("tv_symbol", "")
        if tv_symbol and ":" in tv_symbol:
            exchange = tv_symbol

        print(f"  [{i+1:>3}/{total}] {ticker:<6}", end=" ")
        try:
            result = fetcher.collect_single(ticker, exchange, market="us")
            if result is not None:
                print(f"ok ({len(result)} bars)")
            else:
                print("데이터 없음")
        except Exception as e:
            print(f"FAIL: {e}")
            tracker.mark_error(ticker, exchange, "combined", str(e))


# ── 메인 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="미국 주식 1분봉 통합 ETL")
    parser.add_argument("--databento-only", action="store_true",
                        help="Databento 20종목만 처리")
    parser.add_argument("--legacy-only", action="store_true",
                        help="나머지 종목만 처리 (yfinance + TV)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 작업 없이 상태만 확인")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="모든 확인 프롬프트에 자동 yes")
    args = parser.parse_args()

    log_path = setup_logging()
    tracker = CollectionTracker()

    print("=" * 50)
    print("  미국 주식 1분봉 통합 ETL")
    print(f"  로그: {log_path}")
    print("=" * 50)

    fallback_symbols = []

    if not args.legacy_only:
        # Step 1: Databento 다운로드 확인
        check_databento_downloads(dry_run=args.dry_run, auto_yes=args.yes)

        # Step 2: DBN → Parquet 변환
        convert_databento_to_parquet(DATABENTO_SYMBOLS, tracker, dry_run=args.dry_run, auto_yes=args.yes)

        # Databento 데이터가 없는 종목 → yfinance/TV fallback 대상
        for sym in DATABENTO_SYMBOLS:
            if not find_dbn_files(sym):
                fallback_symbols.append(sym)
        if fallback_symbols:
            print(f"\n  Databento 미수집 → yfinance/TV fallback: {', '.join(fallback_symbols)}")

    if not args.databento_only:
        # Step 3: 나머지 종목 + fallback 종목 수집
        collect_legacy_symbols(tracker, dry_run=args.dry_run, fallback_symbols=fallback_symbols)

    print(f"\n{'=' * 50}")
    print("  완료")
    print(f"  로그: {log_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
