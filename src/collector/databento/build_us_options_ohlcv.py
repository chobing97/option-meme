"""
Databento OPRA dbn.zst → Parquet 변환

download_us_options_ohlcv.py 가 주별로 저장한 dbn.zst 파일을 읽어
publisher 합산 후 연도별 Parquet + contracts 메타데이터로 저장한다.

■ 입력: data/raw/options/databento/us/{SYMBOL}/*.dbn.zst
■ 출력:
    data/raw/options/us/{SYMBOL}/contracts.parquet   계약 메타 (OCC 심볼별 1행)
    data/raw/options/us/{SYMBOL}/{year}.parquet       분봉 OHLCV

■ contracts.parquet 스키마
    symbol       : OCC 심볼 (PK, OHLCV JOIN 키)
    underlying   : 기초자산 (TSLA)
    expiry       : 만기일 (date)
    cp           : P / C
    strike       : 행사가 (float)
    period_start : 해당 주 시작일 (pick-one 선택 시점)
    stock_close  : 선택 기준 주가

■ {year}.parquet 스키마
    datetime : timestamp (US/Eastern, tz-naive)
    symbol   : OCC 심볼
    open, high, low, close : float64
    volume   : int64
    source   : "databento"

■ 합산 로직
    같은 분 + 같은 OCC 심볼에 대해 여러 publisher(거래소)의 데이터를
    open=first, high=max, low=min, close=last, volume=sum 으로 합산.

사용법:
    # dry-run: 변환 대상 파일 확인
    python -m src.collector.databento.build_us_options_ohlcv --symbol TSLA --dry-run

    # 변환 실행
    python -m src.collector.databento.build_us_options_ohlcv --symbol TSLA
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import databento as db
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import RAW_DIR, RAW_OPTIONS_DIR

# ── 경로 ──────────────────────────────────────────────────

DBN_DIR = RAW_DIR / "options" / "databento" / "us"
OUT_DIR = RAW_OPTIONS_DIR / "us"

# ── 캐시 경로 (contracts 메타 소스) ───────────────────────

CACHE_DIR = Path(__file__).parent / ".cache" / "opra"


# ── OCC 심볼 파싱 ─────────────────────────────────────────


def parse_occ_symbol(sym: str) -> dict | None:
    """OCC 심볼 파싱: 'TSLA  260320P00350000' → dict"""
    m = re.match(r"^(\S+?)\s+(\d{6})([CP])(\d{8})$", sym.strip())
    if not m:
        return None
    underlying, expiry, cp, strike_raw = m.groups()
    return {
        "underlying": underlying,
        "expiry": f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}",
        "cp": cp,
        "strike": int(strike_raw) / 1000,
        "symbol": sym.strip(),
    }


# ── DBN → DataFrame ──────────────────────────────────────


def read_dbn_to_df(path: Path) -> pd.DataFrame:
    """DBN 파일을 DataFrame 으로 변환. symbol 컬럼 보존."""
    store = db.DBNStore.from_file(str(path))
    df = store.to_df()
    if df.empty:
        return pd.DataFrame()

    required = ["open", "high", "low", "close", "volume", "symbol"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    # ts_event (UTC) → US/Eastern → tz-naive
    df = df.reset_index()
    ts_col = df.columns[0]  # ts_event
    df = df.rename(columns={ts_col: "datetime"})
    df["datetime"] = (
        pd.to_datetime(df["datetime"])
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    return df[["datetime", "symbol", "open", "high", "low", "close", "volume"]].copy()


# ── Publisher 합산 ────────────────────────────────────────


def merge_publishers(df: pd.DataFrame) -> pd.DataFrame:
    """같은 분 + 같은 OCC 심볼에 대해 publisher 합산.

    open=first, high=max, low=min, close=last, volume=sum
    """
    if df.empty:
        return df

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    if df.empty:
        return df

    df["minute"] = df["datetime"].dt.floor("min")

    merged = df.groupby(["minute", "symbol"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    merged = merged.rename(columns={"minute": "datetime"})
    merged = merged.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return merged


# ── Contracts 메타 생성 ───────────────────────────────────


def build_contracts_meta(
    symbols: set[str], symbol: str
) -> pd.DataFrame:
    """OHLCV 에 등장한 OCC 심볼 + 캐시에서 stock_close/period_start 매칭."""
    import json

    # 캐시에서 주별 stock_close, period_start 로드
    cache_map = {}  # occ_symbol → {period_start, stock_close}
    cache_dir = CACHE_DIR / symbol
    if cache_dir.exists():
        for jf in sorted(cache_dir.glob("symbols_*.json")):
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            ps = data.get("period_start", "")
            sc = data.get("stock_close")
            for sym in data.get("filtered_symbols", []):
                cache_map[sym.strip()] = {"period_start": ps, "stock_close": sc}

    rows = []
    for occ in sorted(symbols):
        p = parse_occ_symbol(occ)
        if not p:
            continue
        meta = cache_map.get(occ, {})
        rows.append({
            "symbol": occ,
            "underlying": p["underlying"],
            "expiry": p["expiry"],
            "cp": p["cp"],
            "strike": p["strike"],
            "period_start": meta.get("period_start", ""),
            "stock_close": meta.get("stock_close"),
        })

    return pd.DataFrame(rows)


# ── Parquet 저장 ──────────────────────────────────────────

OHLCV_SCHEMA = pa.schema([
    ("datetime", pa.timestamp("us")),
    ("symbol", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("source", pa.string()),
])

CONTRACTS_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("underlying", pa.string()),
    ("expiry", pa.string()),
    ("cp", pa.string()),
    ("strike", pa.float64()),
    ("period_start", pa.string()),
    ("stock_close", pa.float64()),
])


def save_ohlcv(df: pd.DataFrame, symbol: str) -> dict[int, int]:
    """연도별 Parquet 저장. Returns {year: bar_count}."""
    df["source"] = "databento"
    df["volume"] = df["volume"].fillna(0).astype("int64")

    out_dir = OUT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for year, ydf in df.groupby(df["datetime"].dt.year):
        path = out_dir / f"{year}.parquet"
        ydf = ydf.sort_values(["symbol", "datetime"]).reset_index(drop=True)

        # 기존 파일과 머지
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                ydf = pd.concat([existing, ydf], ignore_index=True)
                ydf = ydf.drop_duplicates(subset=["datetime", "symbol"], keep="last")
                ydf = ydf.sort_values(["symbol", "datetime"]).reset_index(drop=True)
            except Exception:
                pass

        table = pa.Table.from_pandas(ydf, schema=OHLCV_SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        results[year] = len(ydf)

    return results


def save_contracts(df: pd.DataFrame, symbol: str) -> Path:
    """contracts.parquet 저장."""
    out_dir = OUT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "contracts.parquet"

    # 기존 파일과 머지
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["symbol"], keep="last")
            df = df.sort_values("symbol").reset_index(drop=True)
        except Exception:
            pass

    table = pa.Table.from_pandas(df, schema=CONTRACTS_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return path


# ── 메인 ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Databento OPRA dbn.zst → Parquet 변환"
    )
    parser.add_argument("--symbol", required=True, help="기초자산 (예: TSLA)")
    parser.add_argument(
        "--dry-run", action="store_true", help="변환 대상 파일만 확인"
    )
    args = parser.parse_args()

    dbn_dir = DBN_DIR / args.symbol
    if not dbn_dir.exists():
        print(f"  오류: dbn.zst 디렉토리 없음 — {dbn_dir}", file=sys.stderr)
        sys.exit(1)

    dbn_files = sorted(dbn_dir.glob("*.dbn.zst"))
    print(f"=== OPRA dbn.zst → Parquet 변환 ===")
    print(f"  기초자산 : {args.symbol}")
    print(f"  입력     : {dbn_dir}/ ({len(dbn_files)}개 파일)")
    print(f"  출력     : {OUT_DIR / args.symbol}/")
    print()

    if not dbn_files:
        print("  변환할 파일이 없습니다.")
        return

    if args.dry_run:
        total_size = 0
        for f in dbn_files:
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name}  ({size / 1024:.1f} KB)")
        print(f"\n  총 {len(dbn_files)}개, {total_size / (1024*1024):.1f} MB")
        return

    # 1. 모든 dbn.zst 읽기
    all_dfs = []
    for i, f in enumerate(dbn_files):
        print(f"  [{i+1:>3}/{len(dbn_files)}] {f.name} ... ", end="")
        try:
            df = read_dbn_to_df(f)
            if df.empty:
                print("빈 파일")
            else:
                all_dfs.append(df)
                print(f"{len(df):,} rows")
        except Exception as e:
            print(f"FAIL: {e}")

    if not all_dfs:
        print("\n  읽을 수 있는 데이터가 없습니다.")
        return

    # 2. 전체 합치기 + publisher 합산
    print(f"\n  합산 중 ... ", end="")
    combined = pd.concat(all_dfs, ignore_index=True)
    merged = merge_publishers(combined)
    unique_symbols = merged["symbol"].nunique()
    print(f"{len(combined):,} → {len(merged):,} bars ({unique_symbols}개 계약)")

    # 3. contracts 메타 생성 + 저장
    occ_symbols = set(merged["symbol"].unique())
    contracts_df = build_contracts_meta(occ_symbols, args.symbol)
    contracts_path = save_contracts(contracts_df, args.symbol)
    print(f"\n  contracts: {len(contracts_df)}개 → {contracts_path.name}")

    # 4. OHLCV Parquet 저장
    results = save_ohlcv(merged, args.symbol)
    total_bars = sum(results.values())
    years = sorted(results.keys())
    print(f"  OHLCV: {total_bars:,} bars → {', '.join(f'{y}.parquet({results[y]:,})' for y in years)}")

    print(f"\n=== 완료 ===")
    print(f"  {OUT_DIR / args.symbol}/")


if __name__ == "__main__":
    main()
