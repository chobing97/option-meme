"""
Databento 미국 주식 1분봉 일괄 다운로드 (DBN 원본 포맷)

20개 종목 × 3개 거래소 ohlcv-1m 데이터를 다운로드한다.
--download 없이 실행하면 비용만 계산하고, --download 를 붙이면 실제 다운로드.

사용법:
    # 비용 확인만 (기본)
    python -m src.collector.databento.download_us_stock_ohlcv

    # 실제 다운로드
    python -m src.collector.databento.download_us_stock_ohlcv --download

    # 특정 종목 비용 확인
    python -m src.collector.databento.download_us_stock_ohlcv --symbol TSLA

    # dry-run: 작업 목록만 확인 (API 호출 없음)
    python -m src.collector.databento.download_us_stock_ohlcv --dry-run
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import databento as db

from config.settings import RAW_STOCK_DIR

DATA_DIR = RAW_STOCK_DIR / "databento" / "us"

SYMBOLS = [
    "TSLA", "AAPL", "NVDA", "AMZN", "AMD", "META", "MSFT", "GOOGL", "PLTR", "MARA", "AMC",
    "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "TLT", "SLV",
    "BAC", "BABA",
]

DATASETS = ["XNAS.ITCH", "XNYS.PILLAR", "ARCX.PILLAR"]


def parse_available_end(error_msg: str) -> str | None:
    """Databento 'data_end_after_available_end' 에러에서 사용 가능 종료일 추출."""
    m = re.search(r"data available up to '(\d{4}-\d{2}-\d{2})", str(error_msg))
    return m.group(1) if m else None


def output_path(symbol: str, dataset: str, schema: str, start: str, end: str) -> Path:
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")
    filename = f"{symbol}_{dataset}_{schema}_{start_compact}_{end_compact}.dbn.zst"
    return DATA_DIR / symbol / filename


def estimate_cost(client, symbol: str, dataset: str, schema: str, start: str, end: str) -> float:
    """한 건의 예상 비용 조회. 실패 시 0.0 반환."""
    try:
        return client.metadata.get_cost(
            dataset=dataset, symbols=[symbol], stype_in="raw_symbol",
            schema=schema, start=start, end=end,
        )
    except db.common.error.BentoClientError as e:
        avail = parse_available_end(str(e))
        if avail and "end_after_available" in str(e) and avail > start:
            try:
                return client.metadata.get_cost(
                    dataset=dataset, symbols=[symbol], stype_in="raw_symbol",
                    schema=schema, start=start, end=avail,
                )
            except Exception:
                return 0.0
        return 0.0
    except Exception:
        return 0.0


def download_one(client, symbol: str, dataset: str, schema: str, start: str, end: str) -> str:
    """한 건 다운로드. 결과 상태 문자열 반환: 'ok', 'skip', 'fail'."""
    path = output_path(symbol, dataset, schema, start, end)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  skip ({size_mb:.1f} MB)")
        return "skip"

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            stype_in="raw_symbol",
            schema=schema,
            start=start,
            end=end,
        )
    except db.common.error.BentoClientError as e:
        avail = parse_available_end(str(e))
        if avail and "end_after_available" in str(e) and avail > start:
            try:
                data = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    stype_in="raw_symbol",
                    schema=schema,
                    start=start,
                    end=avail,
                )
                # 조정된 end로 파일명 변경
                path = output_path(symbol, dataset, schema, start, avail)
                path.parent.mkdir(parents=True, exist_ok=True)
            except (db.common.error.BentoClientError, db.common.error.BentoServerError) as e2:
                print(f"  FAIL: {e2}")
                return "fail"
        else:
            print(f"  FAIL: {e}")
            return "fail"
    except db.common.error.BentoServerError as e:
        print(f"  FAIL: {e}")
        return "fail"

    data.to_file(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ok ({size_mb:.1f} MB)")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Databento 미국 주식 1분봉 일괄 다운로드")
    parser.add_argument("--start", default="2020-01-01", help="시작일 (기본: 2020-01-01)")
    parser.add_argument("--end", default="2026-02-24", help="종료일 (기본: 2026-02-24)")
    parser.add_argument("--schema", default="ohlcv-1m", help="스키마 (기본: ohlcv-1m)")
    parser.add_argument("--api-key", default=None, help="Databento API 키")
    parser.add_argument("--download", action="store_true", help="실제 다운로드 실행 (미지정 시 비용 계산만)")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 작업 목록만 출력")
    parser.add_argument("--symbol", default=None, help="특정 종목만 (미지정 시 전체 20개)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS
    jobs = [(sym, ds) for sym in symbols for ds in DATASETS]

    mode = "다운로드" if args.download else "비용 확인"
    print(f"=== Databento 주식 1분봉 {mode} ===")
    print(f"  종목    : {len(symbols)}개")
    print(f"  거래소  : {len(DATASETS)}개 ({', '.join(DATASETS)})")
    print(f"  작업 수 : {len(jobs)}건")
    print(f"  기간    : {args.start} ~ {args.end}")
    print(f"  스키마  : {args.schema}")
    print(f"  저장    : {DATA_DIR}/")
    print()

    if args.dry_run:
        for i, (sym, ds) in enumerate(jobs):
            path = output_path(sym, ds, args.schema, args.start, args.end)
            exists = "존재" if path.exists() else "미존재"
            print(f"  [{i+1:>3}/{len(jobs)}] {sym:<6} {ds:<15} ({exists})")
        print(f"\n  dry-run 완료.")
        return

    try:
        if args.api_key:
            client = db.Historical(args.api_key)
        else:
            client = db.Historical()
    except Exception as e:
        print(f"  오류: Databento 클라이언트 생성 실패: {e}", file=sys.stderr)
        print(f"  DATABENTO_API_KEY 환경변수를 설정하거나 --api-key를 전달하세요.")
        sys.exit(1)

    if not args.download:
        # 비용 계산 모드
        total_cost = 0.0
        need_download = 0
        for i, (sym, ds) in enumerate(jobs):
            path = output_path(sym, ds, args.schema, args.start, args.end)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  [{i+1:>3}/{len(jobs)}] {sym:<6} {ds:<15} 존재 ({size_mb:.1f} MB)")
                continue
            cost = estimate_cost(client, sym, ds, args.schema, args.start, args.end)
            total_cost += cost
            need_download += 1
            print(f"  [{i+1:>3}/{len(jobs)}] {sym:<6} {ds:<15} ${cost:.4f}")

        print(f"\n=== 비용 요약 ===")
        print(f"  다운로드 필요: {need_download}건 / {len(jobs)}건")
        print(f"  총 예상 비용 : ${total_cost:.4f}")
        if need_download > 0:
            print(f"\n  실제 다운로드: --download 옵션을 추가하세요.")
        return

    # 다운로드 모드
    counts = {"ok": 0, "skip": 0, "fail": 0}
    started = datetime.now()

    for i, (sym, ds) in enumerate(jobs):
        print(f"  [{i+1:>3}/{len(jobs)}] {sym:<6} {ds:<15}", end="")
        result = download_one(client, sym, ds, args.schema, args.start, args.end)
        counts[result] += 1

    elapsed = datetime.now() - started
    print(f"\n=== 완료 ({elapsed.seconds // 60}분 {elapsed.seconds % 60}초) ===")
    print(f"  성공: {counts['ok']}건, 스킵: {counts['skip']}건, 실패: {counts['fail']}건")


if __name__ == "__main__":
    main()
