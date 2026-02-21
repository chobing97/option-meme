"""Extract early regular session data (first 60 minutes) from raw bar data."""

from typing import Optional

import exchange_calendars as xcals
import pandas as pd
from loguru import logger

from config.settings import (
    EARLY_SESSION_MINUTES,
    KR_EARLY_END,
    KR_MARKET_OPEN,
    KR_TIMEZONE,
    US_EARLY_END,
    US_MARKET_OPEN,
    US_TIMEZONE,
)

# Exchange calendar identifiers
KR_CALENDAR = "XKRX"  # Korea Exchange
US_CALENDAR = "XNYS"  # NYSE (covers NASDAQ trading days too)


def extract_early_session(
    df: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    """Extract bars from the early regular session (first 60 minutes).

    Args:
        df: DataFrame with 'datetime' column and OHLCV data.
        market: 'kr' or 'us'

    Returns:
        Filtered DataFrame with only early session bars, plus 'date' and
        'minutes_from_open' columns.
    """
    if df.empty:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if market == "kr":
        tz = KR_TIMEZONE
        open_time = KR_MARKET_OPEN
        end_time = KR_EARLY_END
        cal_id = KR_CALENDAR
    elif market == "us":
        tz = US_TIMEZONE
        open_time = US_MARKET_OPEN
        end_time = US_EARLY_END
        cal_id = US_CALENDAR
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

    # Filter to early session window
    mask = (df["_time"] >= open_time) & (df["_time"] < end_time)
    early = df[mask].copy()

    if early.empty:
        return pd.DataFrame()

    # Calculate minutes from market open
    open_h, open_m = map(int, open_time.split(":"))
    early["minutes_from_open"] = (
        early["datetime"].dt.hour * 60
        + early["datetime"].dt.minute
        - (open_h * 60 + open_m)
    )

    early = early.drop(columns=["_time"])
    early = early.sort_values("datetime").reset_index(drop=True)

    logger.debug(
        f"Extracted {len(early)} early session bars across "
        f"{early['date'].nunique()} trading days"
    )

    return early


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
    expected_bars: int = EARLY_SESSION_MINUTES,
) -> dict:
    """Validate early session data quality.

    Args:
        df: Output from extract_early_session
        market: 'kr' or 'us'
        expected_bars: Expected bars per day (default: 60)

    Returns:
        Validation report dict.
    """
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
