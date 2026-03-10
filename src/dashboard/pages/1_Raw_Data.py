"""Phase 0: Raw OHLCV data explorer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import timedelta

import streamlit as st

from dashboard.components.charts import make_candlestick
from dashboard.components.filters import (
    date_range_selector, kb_nav_apply_date, kb_nav_apply_symbol, kb_nav_read,
    market_selector, reload_button, symbol_selector,
)
from dashboard.data_loader import get_raw_date_range, get_raw_symbols, get_raw_trading_dates, get_stock_name_map, load_raw_bars

st.set_page_config(page_title="Raw Data", layout="wide")
st.title("Phase 0: Raw OHLCV Data")

kb_dir = kb_nav_read()

# ── Sidebar filters ───────────────────────────────────────

reload_button()
market = market_selector(key="raw_market")
symbols = get_raw_symbols(market)

if not symbols:
    st.warning(f"No raw data for **{market.upper()}**. Run: `python run_pipeline.py collector --market {market}`")
    st.stop()

name_map = get_stock_name_map(market)
kb_nav_apply_symbol(kb_dir, symbols, "raw_symbol", "raw_chart_date")
symbol = symbol_selector(symbols, key="raw_symbol", name_map=name_map)
if symbol is None:
    st.stop()

# Get date range without loading all data
date_range = get_raw_date_range(market, symbol)
if date_range is None:
    st.warning(f"No bars for {symbol}")
    st.stop()

min_dt, max_dt = date_range[0].date(), date_range[1].date()
start, end = date_range_selector(min_dt, max_dt, key="raw_dates")

# Get trading dates (lightweight — datetime column only)
all_dates = get_raw_trading_dates(market, symbol)
dates = [d for d in all_dates if start <= d <= end]
if not dates:
    st.info("No data in selected range.")
    st.stop()

stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol

# ── Summary metrics ───────────────────────────────────────

cols = st.columns(4)
cols[0].metric("Symbol", stock_label)
cols[1].metric("Trading Days", f"{len(dates):,}")
cols[2].metric("From", str(start))
cols[3].metric("To", str(end))

# ── Per-day chart with date slider ────────────────────────

st.subheader(f"Intraday Chart — {stock_label}")

kb_nav_apply_date(kb_dir, dates, "raw_chart_date")
selected_date = st.select_slider(
    "Date", options=dates, value=dates[-1], key="raw_chart_date",
)

# Load only 1 day of data (end_date + 1 day because storage filters with <=)
next_day = selected_date + timedelta(days=1)
day_df = load_raw_bars(market, symbol, str(selected_date), str(next_day))

if day_df.empty:
    st.info(f"No data for {selected_date}")
else:
    st.plotly_chart(
        make_candlestick(day_df, f"{stock_label} ({market.upper()}) — {selected_date}"),
        use_container_width=True,
    )

# ── Data table (selected day) ────────────────────────────

with st.expander(f"Raw Data Table ({selected_date})"):
    if not day_df.empty:
        st.dataframe(day_df, use_container_width=True, height=400)
    else:
        st.info("No data.")
