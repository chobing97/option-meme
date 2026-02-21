"""Phase 0: Raw OHLCV data explorer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components.charts import make_candlestick
from dashboard.components.filters import date_range_selector, market_selector, symbol_selector
from dashboard.data_loader import get_raw_symbols, get_stock_name_map, load_raw_bars

st.set_page_config(page_title="Raw Data", layout="wide")
st.title("Phase 0: Raw OHLCV Data")

# ── Sidebar filters ───────────────────────────────────────

market = market_selector(key="raw_market")
symbols = get_raw_symbols(market)

if not symbols:
    st.warning(f"No raw data for **{market.upper()}**. Run: `python run_pipeline.py collector --market {market}`")
    st.stop()

name_map = get_stock_name_map(market)
symbol = symbol_selector(symbols, key="raw_symbol", name_map=name_map)
if symbol is None:
    st.stop()

# Load data to get date range
df_full = load_raw_bars(market, symbol)
if df_full.empty:
    st.warning(f"No bars for {symbol}")
    st.stop()

min_dt = df_full["datetime"].min().date()
max_dt = df_full["datetime"].max().date()
start, end = date_range_selector(min_dt, max_dt, key="raw_dates")

# ── Filter by date ────────────────────────────────────────

df = load_raw_bars(market, symbol, str(start), str(end))
if df.empty:
    st.info("No data in selected range.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────

cols = st.columns(4)
stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol
cols[0].metric("Symbol", stock_label)
cols[1].metric("Bars", f"{len(df):,}")
cols[2].metric("From", str(df["datetime"].min())[:10])
cols[3].metric("To", str(df["datetime"].max())[:10])

# ── Candlestick chart ─────────────────────────────────────

st.plotly_chart(make_candlestick(df, f"{stock_label} ({market.upper()})"), use_container_width=True)

# ── Data table ────────────────────────────────────────────

with st.expander("Raw Data Table"):
    st.dataframe(df, use_container_width=True, height=400)
