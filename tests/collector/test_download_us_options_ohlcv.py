"""Tests for src.collector.databento.download_us_options_ohlcv — 일별 옵션 다운로드 로직."""

import json
from datetime import datetime

import pytest

from src.collector.databento.download_us_options_ohlcv import (
    _find_cache_for_date,
    _load_all_cache,
    _next_friday,
    generate_daily_periods,
    generate_weekly_periods,
    get_stock_avg_for_date,
    get_stock_close_for_period,
    load_daily_jobs,
    output_path,
    parse_occ_symbol,
    pick_one_symbol,
)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_prices() -> dict[str, dict]:
    """5 거래일 OHLC + avg 데이터."""
    return {
        "2026-03-09": {"open": 280, "high": 285, "low": 278, "close": 282, "avg": 281.25},
        "2026-03-10": {"open": 282, "high": 288, "low": 280, "close": 286, "avg": 284.0},
        "2026-03-11": {"open": 286, "high": 290, "low": 284, "close": 288, "avg": 287.0},
        "2026-03-12": {"open": 288, "high": 292, "low": 286, "close": 290, "avg": 289.0},
        "2026-03-13": {"open": 290, "high": 294, "low": 288, "close": 292, "avg": 291.0},
    }


@pytest.fixture
def sample_cache_data() -> dict:
    """주간 캐시 JSON 1건 (2026-03-09 ~ 2026-03-16)."""
    return {
        "symbol": "SPY",
        "period_start": "2026-03-09",
        "period_end": "2026-03-16",
        "stock_close": 282.0,
        "strike_range": 20.0,
        "expiry_days": 30,
        "cp_filter": "P",
        "total_contracts": 100,
        "filtered_count": 6,
        "filtered_symbols": [
            "SPY   260313P00280000",  # $280 P exp 03/13
            "SPY   260313P00285000",  # $285 P exp 03/13
            "SPY   260320P00280000",  # $280 P exp 03/20
            "SPY   260320P00285000",  # $285 P exp 03/20
            "SPY   260320P00290000",  # $290 P exp 03/20
            "SPY   260327P00285000",  # $285 P exp 03/27
        ],
    }


@pytest.fixture
def cache_dir(tmp_path, sample_cache_data):
    """tmp_path 에 캐시 파일을 생성."""
    d = tmp_path / "SPY"
    d.mkdir()
    path = d / "symbols_2026-03-09_2026-03-16.json"
    path.write_text(json.dumps(sample_cache_data))
    return tmp_path


# ── parse_occ_symbol ──────────────────────────────────────


class TestParseOccSymbol:
    def test_valid_put(self):
        result = parse_occ_symbol("TSLA  260320P00350000")
        assert result is not None
        assert result["underlying"] == "TSLA"
        assert result["expiry"] == "260320"
        assert result["cp"] == "P"
        assert result["strike"] == 350.0
        assert result["raw"] == "TSLA  260320P00350000"

    def test_valid_call(self):
        result = parse_occ_symbol("SPY   260313C00280000")
        assert result is not None
        assert result["cp"] == "C"
        assert result["strike"] == 280.0

    def test_fractional_strike(self):
        result = parse_occ_symbol("AAPL  260320P00175500")
        assert result is not None
        assert result["strike"] == 175.5

    def test_invalid_format_returns_none(self):
        assert parse_occ_symbol("INVALID") is None
        assert parse_occ_symbol("") is None
        assert parse_occ_symbol("TSLA 260320X00350000") is None  # X is not C or P


# ── _next_friday ──────────────────────────────────────────


class TestNextFriday:
    @pytest.mark.parametrize(
        "date_str, expected_friday",
        [
            ("2026-03-09", "2026-03-20"),  # Mon → next Fri +11
            ("2026-03-10", "2026-03-20"),  # Tue → next Fri +10
            ("2026-03-11", "2026-03-20"),  # Wed → next Fri +9
            ("2026-03-12", "2026-03-20"),  # Thu → next Fri +8
            ("2026-03-13", "2026-03-20"),  # Fri → next Fri +7
            ("2026-03-14", "2026-03-27"),  # Sat → next Fri +13 (토 기준 다음주 금)
            ("2026-03-15", "2026-03-27"),  # Sun → next Fri +12 (일 기준 다음주 금)
            ("2026-03-16", "2026-03-27"),  # Mon → next Fri +11
        ],
    )
    def test_next_friday(self, date_str, expected_friday):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        result = _next_friday(dt)
        assert result.strftime("%Y-%m-%d") == expected_friday
        assert result.weekday() == 4  # always Friday


# ── generate_weekly_periods ───────────────────────────────


class TestGenerateWeeklyPeriods:
    def test_basic(self):
        periods = generate_weekly_periods("2026-03-02", "2026-03-16")
        assert len(periods) == 2
        assert periods[0] == ("2026-03-02", "2026-03-09")
        assert periods[1] == ("2026-03-09", "2026-03-16")

    def test_start_not_monday(self):
        """시작일이 수요일이면 그 주 월요일로 맞춘다."""
        periods = generate_weekly_periods("2026-03-04", "2026-03-16")
        assert periods[0][0] == "2026-03-02"  # 수요일 → 월요일

    def test_end_mid_week(self):
        periods = generate_weekly_periods("2026-03-02", "2026-03-12")
        assert len(periods) == 2
        assert periods[1][1] == "2026-03-12"  # end 가 주 중간


# ── generate_daily_periods ────────────────────────────────


class TestGenerateDailyPeriods:
    def test_basic(self, sample_prices):
        periods = generate_daily_periods(sample_prices)
        assert len(periods) == 5
        # 첫째: 당일 → 익거래일
        assert periods[0] == ("2026-03-09", "2026-03-10")
        assert periods[1] == ("2026-03-10", "2026-03-11")
        # 마지막: 익거래일 없으면 다음 날
        assert periods[4][0] == "2026-03-13"
        assert periods[4][1] == "2026-03-14"

    def test_single_day(self):
        prices = {"2026-03-11": {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "avg": 1.25}}
        periods = generate_daily_periods(prices)
        assert len(periods) == 1
        assert periods[0] == ("2026-03-11", "2026-03-12")

    def test_empty(self):
        assert generate_daily_periods({}) == []


# ── get_stock_avg_for_date ────────────────────────────────


class TestGetStockAvgForDate:
    def test_existing_date(self, sample_prices):
        avg = get_stock_avg_for_date(sample_prices, "2026-03-11")
        assert avg == 287.0

    def test_missing_date(self, sample_prices):
        assert get_stock_avg_for_date(sample_prices, "2026-03-15") is None


# ── get_stock_close_for_period ────────────────────────────


class TestGetStockCloseForPeriod:
    def test_exact_date(self, sample_prices):
        close = get_stock_close_for_period(sample_prices, "2026-03-09")
        assert close == 282.0

    def test_offset_lookup(self, sample_prices):
        """월요일에 주가가 없으면 이전 날짜에서 찾는다."""
        # 2026-03-14 (토)은 없음 → 2026-03-13 (금) 으로 fallback
        close = get_stock_close_for_period(sample_prices, "2026-03-14")
        assert close == 292.0

    def test_no_match(self):
        prices = {"2026-01-01": {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "avg": 1.25}}
        assert get_stock_close_for_period(prices, "2026-03-09") is None


# ── _find_cache_for_date ──────────────────────────────────


class TestFindCacheForDate:
    def test_found(self, sample_cache_data):
        caches = [sample_cache_data]
        result = _find_cache_for_date(caches, "2026-03-11")
        assert result is not None
        assert result["period_start"] == "2026-03-09"

    def test_boundary_start(self, sample_cache_data):
        """period_start 자체는 포함."""
        caches = [sample_cache_data]
        result = _find_cache_for_date(caches, "2026-03-09")
        assert result is not None

    def test_boundary_end_excluded(self, sample_cache_data):
        """period_end 는 미포함 (< pe)."""
        caches = [sample_cache_data]
        result = _find_cache_for_date(caches, "2026-03-16")
        assert result is None

    def test_not_found(self, sample_cache_data):
        caches = [sample_cache_data]
        assert _find_cache_for_date(caches, "2026-04-01") is None


# ── _load_all_cache ───────────────────────────────────────


class TestLoadAllCache:
    def test_load(self, cache_dir, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", cache_dir
        )
        caches = _load_all_cache("SPY")
        assert len(caches) == 1
        assert caches[0]["symbol"] == "SPY"

    def test_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", tmp_path
        )
        assert _load_all_cache("NONEXIST") == []

    def test_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", tmp_path
        )
        d = tmp_path / "SPY"
        d.mkdir()
        (d / "symbols_2026-03-09_2026-03-16.json").write_text("NOT JSON")
        assert _load_all_cache("SPY") == []


# ── load_daily_jobs ───────────────────────────────────────


class TestLoadDailyJobs:
    def test_basic(self, cache_dir, sample_prices, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", cache_dir
        )
        jobs = load_daily_jobs("SPY", strike_range=5.0, expiry_days=21, prices=sample_prices)
        # 5 거래일 모두 캐시 범위(03/09~03/16) 안에 있으므로 5개 job
        assert len(jobs) == 5
        # 각 job 에 필수 키 존재
        for job in jobs:
            assert "date" in job
            assert "download_end" in job
            assert "stock_avg" in job
            assert "symbols" in job
            assert len(job["symbols"]) > 0

    def test_strike_range_filter(self, cache_dir, sample_prices, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", cache_dir
        )
        # strike_range=1 → 평균가와 $1 이내인 심볼만 통과
        jobs = load_daily_jobs("SPY", strike_range=1.0, expiry_days=21, prices=sample_prices)
        for job in jobs:
            avg = job["stock_avg"]
            for sym in job["symbols"]:
                p = parse_occ_symbol(sym)
                assert abs(p["strike"] - avg) <= 1.0

    def test_expiry_days_filter(self, cache_dir, sample_prices, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", cache_dir
        )
        # expiry_days=5 → 03/09 기준 만기 03/14까지만 → 03/13 만기만 통과
        jobs = load_daily_jobs("SPY", strike_range=20.0, expiry_days=5, prices=sample_prices)
        for job in jobs:
            date_dt = datetime.strptime(job["date"], "%Y-%m-%d")
            for sym in job["symbols"]:
                p = parse_occ_symbol(sym)
                assert p["expiry"] == "260313"  # 03/13 만기만

    def test_no_cache(self, tmp_path, sample_prices, monkeypatch):
        monkeypatch.setattr(
            "src.collector.databento.download_us_options_ohlcv.CACHE_DIR", tmp_path
        )
        jobs = load_daily_jobs("SPY", strike_range=5.0, expiry_days=21, prices=sample_prices)
        assert jobs == []


# ── pick_one_symbol ───────────────────────────────────────


class TestPickOneSymbol:
    def test_picks_next_friday_expiry(self):
        """2026-03-11 (수) → 다음주 금 = 03/20. 03/20 만기가 선택되어야 함."""
        job = {
            "date": "2026-03-11",
            "download_end": "2026-03-12",
            "stock_avg": 285.0,
            "symbols": [
                "SPY   260313P00285000",  # exp 03/13 (이번주 금)
                "SPY   260320P00285000",  # exp 03/20 (다음주 금) ← 타겟
                "SPY   260327P00285000",  # exp 03/27 (다다음주 금)
            ],
        }
        result = pick_one_symbol(job)
        assert len(result["symbols"]) == 1
        p = parse_occ_symbol(result["symbols"][0])
        assert p["expiry"] == "260320"  # 다음주 금요일

    def test_picks_closest_strike_to_avg(self):
        """같은 만기 중 평균가에 가장 가까운 행사가 선택."""
        job = {
            "date": "2026-03-11",
            "download_end": "2026-03-12",
            "stock_avg": 287.0,
            "symbols": [
                "SPY   260320P00280000",  # $280, 거리 7
                "SPY   260320P00285000",  # $285, 거리 2 ← 가장 가까움
                "SPY   260320P00290000",  # $290, 거리 3
            ],
        }
        result = pick_one_symbol(job)
        assert len(result["symbols"]) == 1
        p = parse_occ_symbol(result["symbols"][0])
        assert p["strike"] == 285.0

    def test_empty_symbols(self):
        job = {
            "date": "2026-03-11",
            "download_end": "2026-03-12",
            "stock_avg": 285.0,
            "symbols": [],
        }
        result = pick_one_symbol(job)
        assert result["symbols"] == []

    def test_single_symbol(self):
        job = {
            "date": "2026-03-11",
            "download_end": "2026-03-12",
            "stock_avg": 285.0,
            "symbols": ["SPY   260320P00285000"],
        }
        result = pick_one_symbol(job)
        assert len(result["symbols"]) == 1

    def test_monday_targets_next_week_friday(self):
        """월요일(03/16) → 다음주 금 = 03/27."""
        job = {
            "date": "2026-03-16",
            "download_end": "2026-03-17",
            "stock_avg": 290.0,
            "symbols": [
                "SPY   260320P00290000",  # exp 03/20 (이번주 금)
                "SPY   260327P00290000",  # exp 03/27 (다음주 금) ← 타겟
            ],
        }
        result = pick_one_symbol(job)
        p = parse_occ_symbol(result["symbols"][0])
        assert p["expiry"] == "260327"

    def test_friday_targets_next_week_friday(self):
        """금요일(03/13) → 다음주 금 = 03/20."""
        job = {
            "date": "2026-03-13",
            "download_end": "2026-03-16",
            "stock_avg": 290.0,
            "symbols": [
                "SPY   260320P00290000",  # exp 03/20 (다음주 금) ← 타겟
                "SPY   260327P00290000",  # exp 03/27
            ],
        }
        result = pick_one_symbol(job)
        p = parse_occ_symbol(result["symbols"][0])
        assert p["expiry"] == "260320"


# ── output_path ───────────────────────────────────────────


class TestOutputPath:
    def test_format(self):
        p = output_path("SPY", "2026-03-11", "2026-03-12")
        assert p.name == "SPY_OPRA.PILLAR_ohlcv-1m_20260311_20260312.dbn.zst"
        assert "SPY" in p.parts
