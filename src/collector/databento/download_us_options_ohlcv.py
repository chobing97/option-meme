"""
Databento OPRA 미국 주식옵션 1분봉 다운로드 (DBN 원본 포맷)

거래일 단위로 ATM 풋옵션 1종목을 선택하여 ohlcv-1m 데이터를 다운로드한다.
캐시(주 단위 definition)가 없는 주는 자동으로 조회 → 캐시 저장 후 진행.

■ 핵심 로직 — 일 단위 다운로드
    각 거래일마다 당일 주가 평균((O+H+L+C)/4)을 기준으로
    ATM 풋옵션 1종목을 선택하여 당일 1분봉만 다운로드한다.
    포지션이 당일 청산되므로 당일 데이터만 필요.

■ pick-one (기본 ON)
    거래일당 1개의 ATM 풋옵션만 선택하여 다운로드 비용을 극소화한다.
    선택 기준:
    1) 만기가 다음주 금요일에 가장 가까운 계약 우선
    2) 동일 만기 중 행사가가 당일 평균가에 가장 가까운 1개
    예상 비용: ~252거래일 × 1심볼 × 1일 ≈ $3~5/년

■ 캐시 — 주 단위 자동 수집
    캐시: src/collector/databento/.cache/opra/{SYMBOL}/symbols_{START}_{END}.json
    캐시 수집 기준: ±$20, 만기 30일, Put only (넓게 수집)
    다운로드 시 더 좁은 조건(예: ±$5)으로 로컬 재필터.
    이미 캐시된 주는 재조회하지 않음 → 비용 $0

■ 동작 모드
    --dry-run   : 일별 필터 결과(심볼 목록) 출력. API 호출 없음.     비용 $0
    (기본)      : 캐시 자동 수집 + Databento get_cost 비용 산정.     비용 ~$0
    --download  : 캐시 자동 수집 + 실제 ohlcv-1m 다운로드.          비용 ~$4 추정

■ 저장 경로: data/raw/options/databento/us/{SYMBOL}/
    파일명: {SYMBOL}_OPRA.PILLAR_ohlcv-1m_{DATE}_{NEXT_DATE}.dbn.zst
    거래일 단위로 파일 분할 저장.

사용법:
    # dry-run: 일별 필터된 심볼 목록 확인 (캐시 없으면 자동 수집)
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --dry-run

    # 비용 산정 (기본 모드)
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA

    # 실제 다운로드
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --download

    # 기간 지정
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --start 2025-06-01 --end 2025-12-31

    # 파라미터 조정
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --download --strike-range 10

    # 전체 심볼 다운로드 (pick-one 해제)
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --download --pick-all

    # 캐시 무시 (재수집)
    python -m src.collector.databento.download_us_options_ohlcv --symbol TSLA --no-cache --dry-run
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import yfinance as yf

from config.settings import RAW_DIR

# ── 경로 ──────────────────────────────────────────────────

DATA_DIR = RAW_DIR / "options" / "databento" / "us"
CACHE_DIR = Path(__file__).parent / ".cache" / "opra"
COST_BATCH_SIZE = 2000


# ── OCC 심볼 파싱 ─────────────────────────────────────────


def parse_occ_symbol(sym: str) -> dict | None:
    """OCC 심볼 파싱: 'TSLA  260320P00350000' → {underlying, expiry, cp, strike, raw}"""
    m = re.match(r"^(\S+?)\s+(\d{6})([CP])(\d{8})$", sym.strip())
    if not m:
        return None
    underlying, expiry, cp, strike_raw = m.groups()
    strike = int(strike_raw) / 1000
    return {
        "underlying": underlying,
        "expiry": expiry,
        "cp": cp,
        "strike": strike,
        "raw": sym.strip(),
    }


# ── 에러 유틸 ─────────────────────────────────────────────


def parse_available_end(error_msg: str) -> str | None:
    """Databento 'data_end_after_available_end' 에러에서 사용 가능 종료일 추출."""
    m = re.search(r"data available up to '(\d{4}-\d{2}-\d{2})", str(error_msg))
    return m.group(1) if m else None


# ── 캐시 수집 (definition 조회) ──────────────────────────
#    캐시가 없는 주만 Databento definition 을 조회하여 캐시에 저장.
#    수집 기준: ±$20, 만기 30일, Put only (넓게 수집 → 로컬 재필터)

CACHE_STRIKE_RANGE = 20.0
CACHE_EXPIRY_DAYS = 30
CACHE_CP_FILTER = "P"


def generate_weekly_periods(start: str, end: str) -> list[tuple[str, str]]:
    """월요일~월요일 기준 주 단위 구간 리스트 생성. (캐시 수집용)"""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    # 첫 번째 월요일로 맞추기
    days_since_monday = start_dt.weekday()
    if days_since_monday != 0:
        start_dt = start_dt - timedelta(days=days_since_monday)
    periods = []
    current = start_dt
    while current < end_dt:
        next_monday = current + timedelta(days=7)
        p_end = min(next_monday, end_dt)
        periods.append((current.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")))
        current = next_monday
    return periods


def generate_daily_periods(
    prices: dict[str, dict],
) -> list[tuple[str, str]]:
    """거래일 단위 구간 리스트 생성. prices 키(거래일)를 정렬하여 (당일, 익거래일) 쌍 반환."""
    trading_days = sorted(prices.keys())
    periods = []
    for i, day in enumerate(trading_days):
        if i + 1 < len(trading_days):
            next_day = trading_days[i + 1]
        else:
            # 마지막 거래일: 다음 날을 end 로
            dt = datetime.strptime(day, "%Y-%m-%d")
            next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        periods.append((day, next_day))
    return periods


def download_stock_prices(symbol: str, start: str, end: str) -> dict[str, dict]:
    """yfinance로 일봉 OHLC 다운로드. {날짜문자열: {open, high, low, close, avg}} 반환."""
    import pandas as pd

    print(f"  주가 다운로드 중... ({symbol} {start} ~ {end})")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"  오류: {symbol} 주가 데이터를 가져올 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    # MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        ohlc = {}
        for col in ("Open", "High", "Low", "Close"):
            s = df[col][symbol] if symbol in df[col].columns else df[col].iloc[:, 0]
            ohlc[col] = s
    else:
        ohlc = {col: df[col] for col in ("Open", "High", "Low", "Close")}

    prices = {}
    for dt in ohlc["Close"].index:
        key = dt.strftime("%Y-%m-%d")
        o, h, l, c = (
            float(ohlc["Open"].loc[dt]),
            float(ohlc["High"].loc[dt]),
            float(ohlc["Low"].loc[dt]),
            float(ohlc["Close"].loc[dt]),
        )
        prices[key] = {
            "open": o, "high": h, "low": l, "close": c,
            "avg": (o + h + l + c) / 4,
        }
    print(f"  → {len(prices)}개 거래일 주가 확인")
    return prices


def get_stock_close_for_period(
    prices: dict[str, dict], period_start: str
) -> float | None:
    """해당 주 시작일 기준 가장 가까운 종가 반환. (캐시 수집용)"""
    start_dt = datetime.strptime(period_start, "%Y-%m-%d")
    for offset in range(7):
        key = (start_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        if key in prices:
            return prices[key]["close"]
    return None


def get_stock_avg_for_date(
    prices: dict[str, dict], date: str
) -> float | None:
    """해당 거래일의 OHLC 평균가 반환."""
    p = prices.get(date)
    return p["avg"] if p else None


def cache_path(symbol: str, period_start: str, period_end: str) -> Path:
    return CACHE_DIR / symbol / f"symbols_{period_start}_{period_end}.json"


def load_cache(symbol: str, period_start: str, period_end: str) -> dict | None:
    """캐시 파일 로드. 존재하면 반환, 없으면 None."""
    path = cache_path(symbol, period_start, period_end)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(
    symbol: str,
    period_start: str,
    period_end: str,
    stock_close: float,
    total_contracts: int,
    filtered_symbols: list[str],
):
    path = cache_path(symbol, period_start, period_end)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "symbol": symbol,
        "period_start": period_start,
        "period_end": period_end,
        "stock_close": stock_close,
        "strike_range": CACHE_STRIKE_RANGE,
        "expiry_days": CACHE_EXPIRY_DAYS,
        "cp_filter": CACHE_CP_FILTER,
        "total_contracts": total_contracts,
        "filtered_count": len(filtered_symbols),
        "query_timestamp": datetime.now().isoformat(timespec="seconds"),
        "filtered_symbols": filtered_symbols,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def filter_symbols_from_definitions(
    df,
    stock_close: float,
    period_start: str,
) -> tuple[int, list[str]]:
    """definition DataFrame에서 캐시 기준(±$20, 30일, Put)으로 필터링."""
    start_dt = datetime.strptime(period_start, "%Y-%m-%d")
    expiry_limit = start_dt + timedelta(days=CACHE_EXPIRY_DAYS)

    total = 0
    filtered = []
    for sym in df["raw_symbol"].unique():
        p = parse_occ_symbol(sym)
        if not p:
            continue
        total += 1
        if CACHE_CP_FILTER != "CP" and p["cp"] not in CACHE_CP_FILTER:
            continue
        if abs(p["strike"] - stock_close) > CACHE_STRIKE_RANGE:
            continue
        try:
            expiry_dt = datetime.strptime("20" + p["expiry"], "%Y%m%d")
        except ValueError:
            continue
        if not (start_dt < expiry_dt <= expiry_limit):
            continue
        filtered.append(p["raw"])
    return total, filtered


def collect_cache_for_week(
    client,
    symbol: str,
    period_start: str,
    period_end: str,
    stock_close: float,
) -> list[str]:
    """한 주의 Databento definition 조회 → 캐시 저장. 필터된 심볼 반환."""
    log_request("timeseries.get_range(definition)", dataset="OPRA.PILLAR",
                symbols=[f"{symbol}.OPT"], schema="definition",
                start=period_start, end=period_end)
    try:
        data = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            symbols=[f"{symbol}.OPT"],
            stype_in="parent",
            schema="definition",
            start=period_start,
            end=period_end,
        )
        df = data.to_df()
    except db.common.error.BentoClientError as e:
        avail = parse_available_end(str(e))
        if avail and "end_after_available" in str(e) and avail > period_start:
            log_response("definition", False, f"end 조정 → {avail}")
            try:
                data = client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    symbols=[f"{symbol}.OPT"],
                    stype_in="parent",
                    schema="definition",
                    start=period_start,
                    end=avail,
                )
                df = data.to_df()
                period_end = avail
            except (db.common.error.BentoClientError, db.common.error.BentoServerError) as e2:
                log_response("definition", False, str(e2))
                return []
        else:
            log_response("definition", False, str(e))
            return []
    except db.common.error.BentoServerError as e:
        log_response("definition", False, str(e))
        return []

    if df.empty:
        save_cache(symbol, period_start, period_end, stock_close, 0, [])
        log_response("definition", True, "결과 없음")
        return []

    total, filtered = filter_symbols_from_definitions(df, stock_close, period_start)
    save_cache(symbol, period_start, period_end, stock_close, total, filtered)
    log_response("definition", True, f"{total}→{len(filtered)}개")
    return filtered


def ensure_cache(
    symbol: str,
    start: str,
    end: str,
    prices: dict[str, dict],
    api_key: str | None = None,
    no_cache: bool = False,
) -> int:
    """캐시가 없는 주를 자동 수집. 수집한 주 수 반환.

    이미 캐시된 주는 건너뛴다 (--no-cache 시 전체 재수집).
    """
    periods = generate_weekly_periods(start, end)
    cache_dir = CACHE_DIR / symbol

    # 캐시 확인: 전부 있으면 바로 리턴
    if not no_cache and cache_dir.exists():
        missing = [
            (ps, pe) for ps, pe in periods
            if not cache_path(symbol, ps, pe).exists()
        ]
        if not missing:
            return 0
    else:
        missing = periods

    print(f"\n  캐시 수집 필요: {len(missing)}/{len(periods)}주")
    print(f"  캐시 기준: ±${CACHE_STRIKE_RANGE:.0f}, 만기 {CACHE_EXPIRY_DAYS}일, {CACHE_CP_FILTER}")

    # Databento 클라이언트
    try:
        client = db.Historical(api_key) if api_key else db.Historical()
    except Exception:
        print(f"  경고: Databento 클라이언트 생성 실패 — {len(missing)}주 캐시 미수집")
        print(f"         DATABENTO_API_KEY 환경변수를 설정하거나 --api-key를 전달하세요.")
        return 0

    collected = 0
    for i, (ps, pe) in enumerate(missing):
        stock_close = get_stock_close_for_period(prices, ps)
        if stock_close is None:
            print(f"  [{i+1:>3}/{len(missing)}] {ps} ~ {pe}  주가 없음, 건너뜀")
            continue

        print(f"  [{i+1:>3}/{len(missing)}] {ps} ~ {pe}  주가 ${stock_close:,.1f}  ", end="")
        cached = load_cache(symbol, ps, pe) if not no_cache else None
        if cached is not None:
            print(f"cache {cached['filtered_count']}개")
        else:
            syms = collect_cache_for_week(client, symbol, ps, pe, stock_close)
            print(f"→ {len(syms)}개")
            collected += 1

    print(f"  캐시 수집 완료: {collected}주 신규 조회\n")
    return collected


# ── 일별 캐시 로드 + 재필터 ───────────────────────────────


def _load_all_cache(symbol: str) -> list[dict]:
    """모든 주간 캐시 JSON 을 로드."""
    cache_dir = CACHE_DIR / symbol
    if not cache_dir.exists():
        return []
    caches = []
    for jf in sorted(cache_dir.glob("symbols_*.json")):
        try:
            caches.append(json.loads(jf.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return caches


def _find_cache_for_date(caches: list[dict], date: str) -> dict | None:
    """특정 거래일을 포함하는 주간 캐시를 찾는다."""
    for c in caches:
        ps = c.get("period_start", "")
        pe = c.get("period_end", "")
        if ps and pe and ps <= date < pe:
            return c
    return None


def load_daily_jobs(
    symbol: str,
    strike_range: float,
    expiry_days: int,
    prices: dict[str, dict],
) -> list[dict]:
    """주간 캐시를 거래일 단위로 분해하여 재필터. API 호출 없음.

    각 거래일마다 해당 날짜를 포함하는 주 캐시에서 심볼을 필터링한다.
    기준가격은 당일 평균가((O+H+L+C)/4).

    Returns:
        일별 작업 리스트. 각 항목:
        {
            "date": "2026-03-11",
            "download_end": "2026-03-12",  # 익거래일
            "stock_avg": 280.50,
            "symbols": ["TSLA  260320P00280000", ...],
        }
    """
    caches = _load_all_cache(symbol)
    if not caches:
        return []

    daily_periods = generate_daily_periods(prices)
    jobs = []

    for date, next_date in daily_periods:
        cache = _find_cache_for_date(caches, date)
        if cache is None:
            continue

        stock_avg = get_stock_avg_for_date(prices, date)
        if stock_avg is None:
            continue

        date_dt = datetime.strptime(date, "%Y-%m-%d")
        expiry_limit = date_dt + timedelta(days=expiry_days)

        filtered = []
        for sym in cache.get("filtered_symbols", []):
            p = parse_occ_symbol(sym)
            if not p:
                continue
            # 행사가 재필터: 당일 평균가 ±strike_range
            if abs(p["strike"] - stock_avg) > strike_range:
                continue
            # 만기 재필터: date < 만기 ≤ date + expiry_days
            try:
                expiry_dt = datetime.strptime("20" + p["expiry"], "%Y%m%d")
            except ValueError:
                continue
            if not (date_dt < expiry_dt <= expiry_limit):
                continue
            filtered.append(p["raw"])

        if not filtered:
            continue

        jobs.append({
            "date": date,
            "download_end": next_date,
            "stock_avg": stock_avg,
            "symbols": sorted(filtered),
        })

    return jobs


# ── pick-one 필터 ────────────────────────────────────────


def _next_friday(dt: datetime) -> datetime:
    """주어진 날짜의 다음주 금요일을 반환.

    예) 월(0)→+11, 화(1)→+10, 수(2)→+9, 목(3)→+8, 금(4)→+7, 토(5)→+6, 일(6)→+5
    """
    days_ahead = (4 - dt.weekday()) % 7 + 7  # 다음주 금요일까지 일수
    return dt + timedelta(days=days_ahead)


def pick_one_symbol(job: dict) -> dict:
    """일별 job 에서 ATM 풋 1개만 선택.

    선택 기준:
    1) 만기가 다음주 금요일에 가장 가까운 계약 우선
    2) 동일 만기 중 행사가가 당일 평균가에 가장 가까운 1개

    Returns:
        symbols 가 1개로 축소된 새 job dict.
    """
    stock_avg = job["stock_avg"]
    date_dt = datetime.strptime(job["date"], "%Y-%m-%d")
    target_friday = _next_friday(date_dt)

    # 심볼 파싱
    parsed = []
    for sym in job["symbols"]:
        p = parse_occ_symbol(sym)
        if p:
            try:
                p["expiry_dt"] = datetime.strptime("20" + p["expiry"], "%Y%m%d")
            except ValueError:
                continue
            parsed.append(p)

    if not parsed:
        return {**job, "symbols": []}

    # 1단계: 만기가 다음주 금요일에 가장 가까운 그룹
    best_expiry_dist = min(abs((p["expiry_dt"] - target_friday).days) for p in parsed)
    candidates = [
        p for p in parsed
        if abs((p["expiry_dt"] - target_friday).days) == best_expiry_dist
    ]

    # 2단계: 행사가가 당일 평균가에 가장 가까운 1개
    best = min(candidates, key=lambda p: abs(p["strike"] - stock_avg))

    return {
        **job,
        "symbols": [best["raw"]],
    }


# ── 출력 경로 ─────────────────────────────────────────────


def output_path(symbol: str, start: str, end: str) -> Path:
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")
    filename = f"{symbol}_OPRA.PILLAR_ohlcv-1m_{start_compact}_{end_compact}.dbn.zst"
    return DATA_DIR / symbol / filename


# ── 로그 ──────────────────────────────────────────────────


def log_request(method: str, **kwargs):
    """Databento API 요청 로그 출력."""
    symbols = kwargs.get("symbols", [])
    sym_preview = ", ".join(symbols[:3])
    if len(symbols) > 3:
        sym_preview += f" ... (+{len(symbols) - 3}개)"
    print(f"    [API] {method}")
    print(f"           dataset={kwargs.get('dataset', '?')}, schema={kwargs.get('schema', '?')}")
    print(f"           symbols={len(symbols)}개 [{sym_preview}]")
    print(f"           period={kwargs.get('start', '?')} ~ {kwargs.get('end', '?')}")


def log_response(method: str, success: bool, detail: str):
    """Databento API 응답 로그 출력."""
    status = "OK" if success else "FAIL"
    print(f"    [API] {method} → {status}: {detail}")


# ── 비용 산정 ─────────────────────────────────────────────


def estimate_cost_for_day(
    client, symbols: list[str], start: str, end: str
) -> float:
    """한 거래일의 심볼에 대한 비용 산정."""
    params = dict(
        dataset="OPRA.PILLAR",
        symbols=symbols,
        stype_in="raw_symbol",
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    log_request("metadata.get_cost", **params)

    try:
        cost = client.metadata.get_cost(**params)
        log_response("metadata.get_cost", True, f"${cost:.4f}")
        return cost
    except db.common.error.BentoClientError as e:
        avail = parse_available_end(str(e))
        if avail and "end_after_available" in str(e):
            log_response("metadata.get_cost", False, f"end 조정 {end}→{avail}")
            params["end"] = avail
            log_request("metadata.get_cost (retry)", **params)
            try:
                cost = client.metadata.get_cost(**params)
                log_response("metadata.get_cost (retry)", True, f"${cost:.4f}")
                return cost
            except (
                db.common.error.BentoClientError,
                db.common.error.BentoServerError,
            ) as e2:
                log_response("metadata.get_cost (retry)", False, str(e2))
        else:
            log_response("metadata.get_cost", False, str(e))
    except db.common.error.BentoServerError as e:
        log_response("metadata.get_cost", False, str(e))

    return 0.0


# ── 다운로드 ──────────────────────────────────────────────


def download_day(
    client, symbol: str, symbols: list[str], start: str, end: str
) -> str:
    """한 거래일의 필터된 OCC 심볼들의 ohlcv-1m 데이터를 dbn.zst 로 다운로드.

    Returns:
        결과 문자열: 'ok', 'skip', 'fail'
    """
    path = output_path(symbol, start, end)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        log_response("timeseries.get_range", True, f"skip ({size_mb:.1f} MB)")
        return "skip"

    path.parent.mkdir(parents=True, exist_ok=True)

    params = dict(
        dataset="OPRA.PILLAR",
        symbols=symbols,
        stype_in="raw_symbol",
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    log_request("timeseries.get_range", **params)

    try:
        data = client.timeseries.get_range(**params)
    except db.common.error.BentoClientError as e:
        avail = parse_available_end(str(e))
        if avail and "end_after_available" in str(e) and avail > start:
            log_response("timeseries.get_range", False, f"end 조정 {end}→{avail}")
            params["end"] = avail
            path = output_path(symbol, start, avail)
            path.parent.mkdir(parents=True, exist_ok=True)
            log_request("timeseries.get_range (retry)", **params)
            try:
                data = client.timeseries.get_range(**params)
            except (
                db.common.error.BentoClientError,
                db.common.error.BentoServerError,
            ) as e2:
                log_response("timeseries.get_range (retry)", False, str(e2))
                return "fail"
        else:
            log_response("timeseries.get_range", False, str(e))
            return "fail"
    except db.common.error.BentoServerError as e:
        log_response("timeseries.get_range", False, str(e))
        return "fail"

    data.to_file(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    log_response("timeseries.get_range", True, f"{size_mb:.1f} MB → {path.name}")
    return "ok"


# ── 메인 ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Databento OPRA 미국 주식옵션 1분봉 일별 다운로드"
    )
    parser.add_argument("--symbol", required=True, help="기초자산 (예: TSLA)")
    parser.add_argument("--start", default=None, help="시작일 YYYY-MM-DD (기본: 1년 전)")
    parser.add_argument("--end", default=None, help="종료일 YYYY-MM-DD (기본: 오늘)")
    parser.add_argument(
        "--strike-range",
        type=float,
        default=5.0,
        help="주가 대비 행사가 허용 범위 $ (기본: 5.0)",
    )
    parser.add_argument(
        "--expiry-days",
        type=int,
        default=21,
        help="만기일까지 최대 일수 (기본: 21, 다음주 금요일 커버)",
    )
    parser.add_argument("--api-key", default=None, help="Databento API 키")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="일별 필터 결과만 출력 (캐시 없으면 자동 수집)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="실제 ohlcv-1m 데이터 다운로드 (미지정 시 비용 산정만)",
    )
    parser.add_argument(
        "--pick-all",
        action="store_true",
        help="거래일당 전체 필터 심볼 다운로드 (기본: pick-one, 일당 ATM 1개만)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="캐시 무시, definition 전체 재수집",
    )
    args = parser.parse_args()

    # 기간 설정
    end_str = args.end or datetime.now().strftime("%Y-%m-%d")
    if args.start:
        start_str = args.start
    else:
        start_str = (datetime.strptime(end_str, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

    mode = "다운로드" if args.download else "dry-run" if args.dry_run else "비용 산정"
    pick_one = not args.pick_all
    print(f"=== OPRA 옵션 1분봉 일별 다운로드 ===")
    print(f"  기초자산    : {args.symbol}")
    print(f"  기간        : {start_str} ~ {end_str}")
    print(f"  행사가 범위 : ±${args.strike_range:.1f}")
    print(f"  만기 일수   : {args.expiry_days}일")
    print(f"  pick-one    : {'ON — 일당 ATM 1개' if pick_one else 'OFF — 전체 심볼'}")
    print(f"  모드        : {mode}")
    print(f"  저장        : {DATA_DIR / args.symbol}/")

    # 1. yfinance 주가 OHLC 다운로드
    prices = download_stock_prices(args.symbol, start_str, end_str)

    # 2. 캐시 확인 + 자동 수집 (없는 주만 definition 조회)
    ensure_cache(args.symbol, start_str, end_str, prices, args.api_key, args.no_cache)

    # 3. 일별 캐시 로드 + 재필터 (API 호출 없음)
    jobs = load_daily_jobs(args.symbol, args.strike_range, args.expiry_days, prices)
    total_symbols = sum(len(j["symbols"]) for j in jobs)
    unique_symbols = len({s for j in jobs for s in j["symbols"]})
    print(f"  거래일: {len(jobs)}일 (심볼 합계 {total_symbols}개, 고유 {unique_symbols}개)")

    # pick-one: 일당 1개 ATM 풋만 선택 (기본 ON)
    if pick_one:
        jobs = [pick_one_symbol(j) for j in jobs]
        jobs = [j for j in jobs if j["symbols"]]  # 빈 날 제거
        pick_total = sum(len(j["symbols"]) for j in jobs)
        print(f"  pick-one 적용: {pick_total}개 심볼 ({len(jobs)}일)")

    print()

    if not jobs:
        print("  필터 조건에 맞는 심볼이 없습니다.")
        return

    # dry-run: 일별 심볼 목록 출력
    if args.dry_run:
        for i, job in enumerate(jobs):
            print(
                f"  [{i+1:>3}/{len(jobs)}] {job['date']}"
                f"  평균가 ${job['stock_avg']:,.1f}"
                f"  심볼 {len(job['symbols'])}개"
            )
            for sym in job["symbols"]:
                p = parse_occ_symbol(sym)
                if p:
                    print(f"         {sym}  (${p['strike']:.0f} {p['cp']} exp {p['expiry']})")
        print(f"\n  dry-run 완료. --download 추가 후 실행하세요.")
        return

    # Databento 클라이언트 생성
    try:
        if args.api_key:
            client = db.Historical(args.api_key)
        else:
            client = db.Historical()
    except Exception as e:
        print(f"  오류: Databento 클라이언트 생성 실패: {e}", file=sys.stderr)
        print(f"  DATABENTO_API_KEY 환경변수를 설정하거나 --api-key를 전달하세요.")
        sys.exit(1)

    started = datetime.now()
    total_cost = 0.0
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for i, job in enumerate(jobs):
        date = job["date"]
        dl_end = job["download_end"]
        syms = job["symbols"]
        print(
            f"\n  [{i+1:>3}/{len(jobs)}] {date}"
            f"  평균가 ${job['stock_avg']:,.1f}"
            f"  심볼 {len(syms)}개"
        )

        if args.download:
            result = download_day(client, args.symbol, syms, date, dl_end)
            counts[result] += 1
        else:
            cost = estimate_cost_for_day(client, syms, date, dl_end)
            total_cost += cost

    elapsed = datetime.now() - started
    print(f"\n=== 완료 ({elapsed.seconds // 60}분 {elapsed.seconds % 60}초) ===")

    if args.download:
        print(f"  성공: {counts['ok']}건, 스킵: {counts['skip']}건, 실패: {counts['fail']}건")
    else:
        print(f"  총 예상 비용: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
