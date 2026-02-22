"""Phase 1: Labeled data explorer — peak/trough visualization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import make_editable_candlestick, make_label_distribution
from dashboard.components.filters import (
    date_range_selector, kb_nav_apply_date, kb_nav_apply_symbol, kb_nav_read,
    market_selector, reload_button, symbol_selector,
)
from dashboard.data_loader import get_stock_name_map, load_labeled, save_label_edit

st.set_page_config(page_title="Labels", layout="wide")
st.title("Phase 1: Peak/Trough Labels")

kb_dir = kb_nav_read()

# ── Sidebar ───────────────────────────────────────────────

reload_button()
market = market_selector(key="label_market")
df = load_labeled(market)

if df.empty:
    st.warning(f"No labeled data for **{market.upper()}**. Run: `python run_pipeline.py labeler --market {market}`")
    st.stop()

symbols = sorted(df["symbol"].unique().tolist())
name_map = get_stock_name_map(market)
kb_nav_apply_symbol(kb_dir, symbols, "label_symbol", "label_chart_date")
symbol = symbol_selector(symbols, key="label_symbol", name_map=name_map)
if symbol is None:
    st.stop()

sym_df = df[df["symbol"] == symbol].copy()
if sym_df.empty:
    st.info(f"No labeled data for {symbol}")
    st.stop()

min_dt = sym_df["datetime"].min().date()
max_dt = sym_df["datetime"].max().date()
start, end = date_range_selector(min_dt, max_dt, key="label_dates")

sym_df = sym_df[(sym_df["datetime"].dt.date >= start) & (sym_df["datetime"].dt.date <= end)]
if sym_df.empty:
    st.info("No data in selected range.")
    st.stop()

# ── Label distribution ────────────────────────────────────

st.subheader("Label Distribution")
label_counts = sym_df["label"].value_counts().to_dict()

cols = st.columns(4)
cols[0].metric("Total Bars", f"{len(sym_df):,}")
cols[1].metric("Peaks", f"{label_counts.get(1, 0):,}")
cols[2].metric("Troughs", f"{label_counts.get(2, 0):,}")
cols[3].metric("Neither", f"{label_counts.get(0, 0):,}")

st.plotly_chart(make_label_distribution(label_counts), use_container_width=True)

# ── Per-day chart (interactive editing) ───────────────────

stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol
st.subheader(f"Labeled Chart — {stock_label}")
st.caption("Click a bar to cycle label: **None → Peak → Trough → None**")

dates = sorted(sym_df["date"].unique()) if "date" in sym_df.columns else []
kb_nav_apply_date(kb_dir, dates, "label_chart_date")
if dates:
    selected_date = st.select_slider("Date", options=dates, value=dates[-1], key="label_chart_date")
    day_df = sym_df[sym_df["date"] == selected_date]
else:
    day_df = sym_df

if not day_df.empty:
    stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol
    fig = make_editable_candlestick(day_df, f"{stock_label} — {selected_date if dates else ''}")
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
        # Find the first point that has customdata (from the clickable scatter trace)
        clicked_dt = None
        for pt in points:
            cd = pt.get("customdata")
            if cd is not None:
                clicked_dt = cd if isinstance(cd, str) else str(cd)
                break

        if clicked_dt:
            # Look up current label
            match = day_df[day_df["datetime"].astype(str) == clicked_dt]
            if not match.empty:
                current_label = int(match.iloc[0]["label"])
                new_label = (current_label + 1) % 3  # 0→1→2→0
                save_label_edit(market, symbol, clicked_dt, new_label)
                st.rerun()

# ── Label stats ───────────────────────────────────────────

with st.expander("Labeled Data Table"):
    st.dataframe(sym_df, use_container_width=True, height=400)
