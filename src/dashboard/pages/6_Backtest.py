"""Phase 6: Backtest results viewer — equity curve, trades, signals, drawdown."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import make_backtest_chart
from dashboard.components.filters import reload_button
from dashboard.data_loader import get_backtest_files, load_backtest

st.set_page_config(page_title="Backtest", layout="wide")
st.title("Phase 6: Backtest Results")

# ── Sidebar ───────────────────────────────────────────────

reload_button()

files = get_backtest_files()
if not files:
    st.warning(
        "No backtest data found. Run a backtest first:\n\n"
        "```\n./optionmeme trade --market us --date-from 2025-03-11 --date-to 2026-02-27 --broker historical\n```"
    )
    st.stop()

selected = st.sidebar.selectbox("Backtest", files, key="bt_file")
df = load_backtest(selected)

if df.empty:
    st.warning("Selected backtest file is empty.")
    st.stop()

# Symbol filter
symbols = sorted(df["symbol"].unique().tolist())
if len(symbols) > 1:
    symbol = st.sidebar.selectbox("Symbol", symbols, key="bt_symbol")
else:
    symbol = symbols[0]

sym_df = df[df["symbol"] == symbol].copy()

# Date filter for multi-day
dates = sorted(sym_df["timestamp"].dt.date.unique())
if len(dates) > 1:
    date_options = ["All"] + [str(d) for d in dates]
    selected_date = st.sidebar.selectbox("Date", date_options, key="bt_date")
    if selected_date != "All":
        import datetime
        sel_date = datetime.date.fromisoformat(selected_date)
        day_df = sym_df[sym_df["timestamp"].dt.date == sel_date]
    else:
        day_df = sym_df
else:
    day_df = sym_df
    selected_date = str(dates[0]) if dates else "N/A"

# ── KPI Summary ──────────────────────────────────────────

trades = day_df[day_df["action"] != ""]
buys = trades[trades["action"] == "BUY_PUT"]
sells = trades[trades["action"] == "SELL_PUT"]

initial_equity = day_df["equity"].iloc[0] if len(day_df) > 0 else 0
final_equity = day_df["equity"].iloc[-1] if len(day_df) > 0 else 0
return_pct = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
max_dd = day_df["drawdown_pct"].min() if len(day_df) > 0 else 0

# Win rate: match buys and sells by looking at equity change around trades
sell_reasons = sells["reason"].value_counts().to_dict() if not sells.empty else {}

cols = st.columns(6)
cols[0].metric("Total Bars", f"{len(day_df):,}")
cols[1].metric("Buys", f"{len(buys)}")
cols[2].metric("Sells", f"{len(sells)}")
cols[3].metric("Return", f"{return_pct:+.2%}")
cols[4].metric("Max Drawdown", f"{max_dd:.2%}")
cols[5].metric("Final Equity", f"{final_equity:,.0f}")

# Sell reason breakdown
if sell_reasons:
    reason_cols = st.columns(len(sell_reasons))
    for i, (reason, count) in enumerate(sorted(sell_reasons.items())):
        reason_cols[i].metric(reason.replace("_", " ").title(), count)

# ── Main Chart ───────────────────────────────────────────

period_label = selected_date if selected_date != "All" else selected
chart_title = f"{symbol} — {period_label}"
fig = make_backtest_chart(day_df, title=chart_title)
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

# ── Daily Summary (multi-day only) ───────────────────────

if len(dates) > 1 and selected_date == "All":
    st.subheader("Daily Summary")
    daily_rows = []
    for d in dates:
        ddf = sym_df[sym_df["timestamp"].dt.date == d]
        if ddf.empty:
            continue
        d_buys = len(ddf[ddf["action"] == "BUY_PUT"])
        d_sells = len(ddf[ddf["action"] == "SELL_PUT"])
        d_start = ddf["equity"].iloc[0]
        d_end = ddf["equity"].iloc[-1]
        d_ret = (d_end - d_start) / d_start if d_start > 0 else 0
        d_dd = ddf["drawdown_pct"].min()
        daily_rows.append({
            "date": str(d),
            "bars": len(ddf),
            "buys": d_buys,
            "sells": d_sells,
            "return": f"{d_ret:+.2%}",
            "max_dd": f"{d_dd:.2%}",
            "end_equity": f"{d_end:,.0f}",
        })

    if daily_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True)
