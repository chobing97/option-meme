"""Tests for src.labeler.session_extractor."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.labeler.session_extractor import (
    extract_early_session,
    extract_session,
    get_trading_days,
    split_by_day,
    validate_session_data,
)


# -- extract_session --------------------------------------------------------


class TestExtractSession:
    def test_kr_filters_full_session(self, raw_ohlcv_kr):
        result = extract_session(raw_ohlcv_kr, "kr")

        assert len(result) == 390
        times = result["datetime"].dt.strftime("%H:%M")
        assert (times >= "09:00").all()
        assert (times < "15:30").all()

    def test_us_filters_full_session(self, raw_ohlcv_us):
        result = extract_session(raw_ohlcv_us, "us")

        assert len(result) == 390
        times = result["datetime"].dt.strftime("%H:%M")
        assert (times >= "09:30").all()
        assert (times < "16:00").all()

    def test_adds_date_and_minutes_columns(self, raw_ohlcv_kr):
        result = extract_session(raw_ohlcv_kr, "kr")

        assert "date" in result.columns
        assert "minutes_from_open" in result.columns

    def test_minutes_from_open_range_kr(self, raw_ohlcv_kr):
        result = extract_session(raw_ohlcv_kr, "kr")

        assert result["minutes_from_open"].min() == 0
        assert result["minutes_from_open"].max() == 389

    def test_minutes_from_open_range_us(self, raw_ohlcv_us):
        result = extract_session(raw_ohlcv_us, "us")

        assert result["minutes_from_open"].min() == 0
        assert result["minutes_from_open"].max() == 389

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        result = extract_session(empty, "kr")

        assert result.empty

    def test_unknown_market_raises(self, raw_ohlcv_kr):
        with pytest.raises(ValueError, match="Unknown market"):
            extract_session(raw_ohlcv_kr, "jp")

    def test_tz_aware_input(self, raw_ohlcv_kr):
        """tz-aware datetime should be tz_convert'd and filtered correctly."""
        df = raw_ohlcv_kr.copy()
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("Asia/Seoul")

        result = extract_session(df, "kr")

        assert len(result) == 390
        times = result["datetime"].dt.strftime("%H:%M")
        assert (times >= "09:00").all()
        assert (times < "15:30").all()


class TestExtractEarlySessionAlias:
    """Verify backward-compatible alias still works."""

    def test_alias_is_same_function(self):
        assert extract_early_session is extract_session

    def test_alias_works(self, raw_ohlcv_kr):
        result = extract_early_session(raw_ohlcv_kr, "kr")

        assert len(result) == 390
        assert "date" in result.columns
        assert "minutes_from_open" in result.columns


# -- get_trading_days -------------------------------------------------------


class TestGetTradingDays:
    def test_returns_trading_day_list(self):
        mock_session = MagicMock()
        mock_dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"])
        mock_session.sessions_in_range.return_value = mock_dates

        mock_cal = MagicMock(return_value=mock_session)
        with patch("src.labeler.session_extractor.xcals.get_calendar", mock_cal):
            result = get_trading_days("kr", "2025-01-01", "2025-01-07")

        assert len(result) == 3
        import datetime
        assert all(isinstance(d, datetime.date) for d in result)
        mock_cal.assert_called_once_with("XKRX")


# -- split_by_day -----------------------------------------------------------


class TestSplitByDay:
    def test_splits_by_date(self, session_kr):
        # Add a second day to test splitting
        import datetime

        df2 = session_kr.copy()
        df2["date"] = datetime.date(2025, 1, 3)
        df2["datetime"] = df2["datetime"] + pd.Timedelta(days=1)
        combined = pd.concat([session_kr, df2], ignore_index=True)

        result = split_by_day(combined)

        assert len(result) == 2
        for day_df in result.values():
            assert len(day_df) == 390

    def test_no_date_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="date"):
            split_by_day(df)


# -- validate_session_data --------------------------------------------------


class TestValidateSessionData:
    def test_full_day_valid(self, session_kr):
        result = validate_session_data(session_kr, "kr")

        assert result["valid"] is True
        assert result["full_days"] == result["total_days"]
        assert result["short_days"] == 0

    def test_empty_df_invalid(self):
        empty = pd.DataFrame()
        result = validate_session_data(empty, "kr")

        assert result["valid"] is False

    def test_custom_expected_bars(self, session_kr):
        """Custom expected_bars should override the default."""
        result = validate_session_data(session_kr, "kr", expected_bars=60)

        assert result["valid"] is True
        # With expected_bars=60, 390 bars/day is way above 80% threshold
        assert result["full_days"] == result["total_days"]

    def test_default_expected_bars_uses_session_minutes(self, session_kr):
        """Without expected_bars, should use KR_SESSION_MINUTES (390)."""
        result = validate_session_data(session_kr, "kr")

        assert result["valid"] is True
        assert result["avg_bars_per_day"] == 390
