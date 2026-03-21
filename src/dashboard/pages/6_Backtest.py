"""Phase 6: Backtest results viewer — equity curve, trades, signals, drawdown."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import make_backtest_chart
from dashboard.components.filters import kb_nav_apply_date, kb_nav_apply_selectbox, kb_nav_read, reload_button, timeframe_selector
from datetime import timedelta

from dashboard.data_loader import (
    get_backtest_files, get_backtest_symbols, get_backtest_trading_dates,
    has_options_data, load_backtest, load_options_ohlcv_by_strike, load_raw_bars,
)

st.set_page_config(page_title="Backtest", layout="wide")
st.title("Phase 6: Backtest Results")

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

kb_dir = kb_nav_read()

# ── Sidebar ───────────────────────────────────────────────

reload_button()
timeframe = timeframe_selector(key="timeframe")

files = get_backtest_files()
if not files:
    st.warning(
        "No backtest data found. Run a backtest first:\n\n"
        "```\n./optionmeme trade --market us --date-from 2025-03-11 --date-to 2026-02-27 --broker historical\n```"
    )
    st.stop()

kb_nav_apply_selectbox(kb_dir, files, "bt_file")
selected = st.sidebar.selectbox("Backtest", files, key="bt_file")

# Symbol filter (lightweight)
symbols = get_backtest_symbols(selected)
if not symbols:
    st.warning("Selected backtest file is empty.")
    st.stop()

if len(symbols) > 1:
    symbol = st.sidebar.selectbox("Symbol", symbols, key="bt_symbol")
else:
    symbol = symbols[0]

# Trading dates (lightweight)
dates = get_backtest_trading_dates(selected, symbol)
if not dates:
    st.warning(f"No data for {symbol}.")
    st.stop()

# Date slider (skip slider if only 1 date to avoid Streamlit min==max bug)
kb_nav_apply_date(kb_dir, dates, "bt_chart_date")
if len(dates) > 1:
    selected_date = st.select_slider("Date", options=dates, value=dates[-1], key="bt_chart_date")
else:
    selected_date = dates[0]
day_name = DAY_NAMES[selected_date.weekday()]

# Load only selected symbol + date
day_df = load_backtest(selected, symbol=symbol, date_str=str(selected_date))

if day_df.empty:
    st.info(f"No data for {selected_date}")
    st.stop()

# ── KPI Summary ──────────────────────────────────────────

trades = day_df[day_df["action"] != ""]
buys = trades[trades["action"] == "BUY_PUT"]
sells = trades[trades["action"] == "SELL_PUT"]

initial_equity = day_df["equity"].iloc[0] if len(day_df) > 0 else 0
final_equity = day_df["equity"].iloc[-1] if len(day_df) > 0 else 0
return_pct = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
max_dd = day_df["drawdown_pct"].min() if len(day_df) > 0 else 0

sell_reasons = sells["reason"].value_counts().to_dict() if not sells.empty else {}

cols = st.columns(6)
cols[0].metric("Total Bars", f"{len(day_df):,}")
cols[1].metric("Buys", f"{len(buys)}")
cols[2].metric("Sells", f"{len(sells)}")
cols[3].metric("Return", f"{return_pct:+.2%}")
cols[4].metric("Max Drawdown", f"{max_dd:.2%}")
cols[5].metric("Final Equity", f"{final_equity:,.0f}")

if sell_reasons:
    reason_cols = st.columns(len(sell_reasons))
    for i, (reason, count) in enumerate(sorted(sell_reasons.items())):
        reason_cols[i].metric(reason.replace("_", " ").title(), count)

# ── Load real OHLCV data for candlestick charts ──────────

# Stock OHLCV
next_day = selected_date + timedelta(days=1)
bt_market = day_df["market"].iloc[0] if "market" in day_df.columns and not day_df.empty else "us"
stock_ohlcv = load_raw_bars(bt_market, symbol, str(selected_date), str(next_day), timeframe)

# Option OHLCV
option_ohlcv = None
if not buys.empty and has_options_data(bt_market, symbol):
    trade_strike = buys["strike"].iloc[0]
    option_ohlcv = load_options_ohlcv_by_strike(bt_market, symbol, str(selected_date), trade_strike)

# ── Main Chart ───────────────────────────────────────────

chart_title = f"{symbol} — {selected_date} ({day_name})"
fig = make_backtest_chart(day_df, title=chart_title, stock_ohlcv=stock_ohlcv, option_ohlcv=option_ohlcv)
st.plotly_chart(fig, use_container_width=True)

# ── Trade History Table ──────────────────────────────────

st.subheader("Trade History")
if trades.empty:
    st.info("No trades in selected period.")
else:
    display_cols = [
        "timestamp", "symbol", "action", "reason", "strike", "fill_price",
        "underlying_close", "position_qty", "cash", "equity", "drawdown_pct",
    ]
    available = [c for c in display_cols if c in trades.columns]
    trade_display = trades[available].copy()
    trade_display["drawdown_pct"] = trade_display["drawdown_pct"].map("{:.2%}".format)
    trade_display["cash"] = trade_display["cash"].map("{:,.0f}".format)
    trade_display["equity"] = trade_display["equity"].map("{:,.0f}".format)
    st.dataframe(trade_display, use_container_width=True, hide_index=True)
