"""Phase 1: Labeled data explorer — peak/trough visualization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import make_editable_candlestick, make_label_distribution
from dashboard.components.filters import (
    date_range_selector, kb_nav_apply_date, kb_nav_apply_symbol, kb_nav_read,
    label_config_selector, market_selector, reload_button, symbol_selector,
)
from dashboard.data_loader import (
    get_labeled_date_range, get_labeled_symbol_stats, get_labeled_symbols,
    get_labeled_trading_dates, get_stock_name_map, load_labeled, save_label_edit,
)

st.set_page_config(page_title="Labels", layout="wide")
st.title("Phase 1: Peak/Trough Labels")

kb_dir = kb_nav_read()

# ── Sidebar ───────────────────────────────────────────────

reload_button()
market = market_selector(key="label_market")
label_config = label_config_selector(key="label_lc")

symbols = get_labeled_symbols(market, label_config)
if not symbols:
    st.warning(f"No labeled data for **{market.upper()}**. Run: `python run_pipeline.py labeler --market {market}`")
    st.stop()

name_map = get_stock_name_map(market)
kb_nav_apply_symbol(kb_dir, symbols, "label_symbol", "label_chart_date")
symbol = symbol_selector(symbols, key="label_symbol", name_map=name_map)
if symbol is None:
    st.stop()

# Date range (lightweight)
date_range = get_labeled_date_range(market, label_config, symbol)
if date_range is None:
    st.info(f"No labeled data for {symbol}")
    st.stop()

min_dt, max_dt = date_range[0].date(), date_range[1].date()
start, end = date_range_selector(min_dt, max_dt, key="label_dates")

# Trading dates (lightweight)
all_dates = get_labeled_trading_dates(market, label_config, symbol)
dates = [d for d in all_dates if start <= d <= end]
if not dates:
    st.info("No data in selected range.")
    st.stop()

stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol

# ── Label distribution ────────────────────────────────────

st.subheader("Label Distribution")
label_counts = get_labeled_symbol_stats(market, label_config, symbol)

cols = st.columns(4)
cols[0].metric("Total Bars", f"{sum(label_counts.values()):,}")
cols[1].metric("Peaks", f"{label_counts.get(1, 0):,}")
cols[2].metric("Troughs", f"{label_counts.get(2, 0):,}")
cols[3].metric("Neither", f"{label_counts.get(0, 0):,}")

st.plotly_chart(make_label_distribution(label_counts), use_container_width=True)

# ── Per-day chart (interactive editing) ───────────────────

st.subheader(f"Labeled Chart — {stock_label}")
st.caption("Click a bar to cycle label: **None → Peak → Trough → None**")

kb_nav_apply_date(kb_dir, dates, "label_chart_date")
selected_date = st.select_slider("Date", options=dates, value=dates[-1], key="label_chart_date")

# Load only 1 day of data for the selected symbol
day_df = load_labeled(market, label_config, symbol=symbol, date_str=str(selected_date))

if not day_df.empty:
    fig = make_editable_candlestick(day_df, f"{stock_label} — {selected_date}")
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("points",),
        key="label_editor",
    )

    # Handle selection event
    if event and event.selection and event.selection.get("points"):
        points = event.selection["points"]
        clicked_dt = None
        for pt in points:
            cd = pt.get("customdata")
            if cd is not None:
                clicked_dt = cd if isinstance(cd, str) else str(cd)
                break

        if clicked_dt:
            match = day_df[day_df["datetime"].astype(str) == clicked_dt]
            if not match.empty:
                current_label = int(match.iloc[0]["label"])
                new_label = (current_label + 1) % 3  # 0→1→2→0
                save_label_edit(market, symbol, clicked_dt, new_label, label_config)
                st.rerun()

# ── Label stats ───────────────────────────────────────────

with st.expander("Labeled Data Table"):
    if not day_df.empty:
        st.dataframe(day_df, use_container_width=True, height=400)
    else:
        st.info("No data.")
