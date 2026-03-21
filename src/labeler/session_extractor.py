"""Extract regular session data from raw bar data."""

from typing import Optional

import exchange_calendars as xcals
import pandas as pd
from loguru import logger

from config.settings import (
    KR_MARKET_CLOSE,
    KR_MARKET_OPEN,
    KR_SESSION_MINUTES,
    KR_TIMEZONE,
    US_MARKET_CLOSE,
    US_MARKET_OPEN,
    US_SESSION_MINUTES,
    US_TIMEZONE,
)

# Exchange calendar identifiers
KR_CALENDAR = "XKRX"  # Korea Exchange
US_CALENDAR = "XNYS"  # NYSE (covers NASDAQ trading days too)


def extract_session(
    df: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    """Extract bars from the full regular session.

    Args:
        df: DataFrame with 'datetime' column and OHLCV data.
        market: 'kr' or 'us'

    Returns:
        Filtered DataFrame with only regular session bars, plus 'date' and
        'minutes_from_open' columns.
    """
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if market == "kr":
        tz = KR_TIMEZONE
        open_time = KR_MARKET_OPEN
        close_time = KR_MARKET_CLOSE
    elif market == "us":
        tz = US_TIMEZONE
        open_time = US_MARKET_OPEN
        close_time = US_MARKET_CLOSE
    else:
        raise ValueError(f"Unknown market: {market}")

    # Localize to market timezone if naive
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(tz)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(tz)

    # Extract time components for filtering
    df["_time"] = df["datetime"].dt.strftime("%H:%M")
    df["date"] = df["datetime"].dt.date

    # Filter to full session window: open_time <= time < close_time
    mask = (df["_time"] >= open_time) & (df["_time"] < close_time)
    session = df[mask].copy()

    if session.empty:
        return pd.DataFrame()

    # Calculate minutes from market open
    open_h, open_m = map(int, open_time.split(":"))
    session["minutes_from_open"] = (
        session["datetime"].dt.hour * 60
        + session["datetime"].dt.minute
        - (open_h * 60 + open_m)
    )

    session = session.drop(columns=["_time"])
    session = session.sort_values("datetime").reset_index(drop=True)

    logger.debug(
        f"Extracted {len(session)} session bars across "
        f"{session['date'].nunique()} trading days"
    )

    return session


# Backward-compatible alias
extract_early_session = extract_session


def get_trading_days(
    market: str,
    start_date: str,
    end_date: str,
) -> list:
    """Get list of trading days for a market using exchange_calendars.

    Args:
        market: 'kr' or 'us'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of datetime.date objects.
    """
    cal_id = KR_CALENDAR if market == "kr" else US_CALENDAR
    cal = xcals.get_calendar(cal_id)
    sessions = cal.sessions_in_range(start_date, end_date)
    return [s.date() for s in sessions]


def split_by_day(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split early session DataFrame into per-day DataFrames.

    Args:
        df: DataFrame with 'date' column (from extract_early_session).

    Returns:
        Dict mapping date string to that day's DataFrame.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must have 'date' column. Use extract_early_session first.")

    result = {}
    for date_val, group in df.groupby("date"):
        date_str = str(date_val)
        day_df = group.sort_values("datetime").reset_index(drop=True)
        result[date_str] = day_df

    return result


def validate_session_data(
    df: pd.DataFrame,
    market: str,
    expected_bars: Optional[int] = None,
) -> dict:
    """Validate session data quality.

    Args:
        df: Output from extract_session
        market: 'kr' or 'us'
        expected_bars: Expected bars per day. If None, uses session minutes
            for the market (KR: 390, US: 390 for 1-minute bars).

    Returns:
        Validation report dict.
    """
    if expected_bars is None:
        expected_bars = KR_SESSION_MINUTES if market == "kr" else US_SESSION_MINUTES
    if df.empty:
        return {"valid": False, "error": "empty dataframe"}

    days = split_by_day(df)
    bars_per_day = {d: len(g) for d, g in days.items()}

    short_days = {d: n for d, n in bars_per_day.items() if n < expected_bars * 0.8}
    full_days = {d: n for d, n in bars_per_day.items() if n >= expected_bars * 0.8}

    return {
        "valid": True,
        "total_days": len(days),
        "full_days": len(full_days),
        "short_days": len(short_days),
        "short_day_details": short_days,
        "avg_bars_per_day": sum(bars_per_day.values()) / len(bars_per_day) if bars_per_day else 0,
        "total_bars": len(df),
    }
