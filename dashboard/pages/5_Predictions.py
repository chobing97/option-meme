"""Phase 5: Model predictions viewer with label comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import (
    make_candlestick_with_labels,
    make_candlestick_with_probs,
    make_label_distribution,
)
from dashboard.components.filters import (
    date_range_selector, kb_nav_apply_date, kb_nav_apply_symbol, kb_nav_read,
    label_config_selector, market_selector, model_config_selector, reload_button, symbol_selector,
)
from dashboard.data_loader import get_stock_name_map, load_labeled, load_predicted, load_split_dates

st.set_page_config(page_title="Predictions", layout="wide")
st.title("Phase 5: Predictions")

kb_dir = kb_nav_read()

# ── Sidebar ───────────────────────────────────────────────

reload_button()
market = market_selector(key="pred_market")
lc = label_config_selector(key="pred_lc")
mc = model_config_selector(key="pred_mc")
pred_df = load_predicted(market, lc, mc)

if pred_df.empty:
    st.warning(
        f"No prediction data for **{market.upper()}**. "
        f"Run: `python run_pipeline.py predict --market {market}`"
    )
    st.stop()

symbols = sorted(pred_df["symbol"].unique().tolist())
name_map = get_stock_name_map(market)
kb_nav_apply_symbol(kb_dir, symbols, "pred_symbol", "pred_chart_date")
symbol = symbol_selector(symbols, key="pred_symbol", name_map=name_map)
if symbol is None:
    st.stop()

sym_pred = pred_df[pred_df["symbol"] == symbol].copy()
if sym_pred.empty:
    st.info(f"No prediction data for {symbol}")
    st.stop()

min_dt = sym_pred["datetime"].min().date()
max_dt = sym_pred["datetime"].max().date()
start, end = date_range_selector(min_dt, max_dt, key="pred_dates")

sym_pred = sym_pred[(sym_pred["datetime"].dt.date >= start) & (sym_pred["datetime"].dt.date <= end)]
if sym_pred.empty:
    st.info("No data in selected range.")
    st.stop()

stock_label = f"{symbol}({name_map[symbol]})" if symbol in name_map else symbol

# ── Prediction distribution ──────────────────────────────

st.subheader("Prediction Distribution")
label_counts = sym_pred["label"].value_counts().to_dict()

cols = st.columns(4)
cols[0].metric("Total Bars", f"{len(sym_pred):,}")
cols[1].metric("Peaks", f"{label_counts.get(1, 0):,}")
cols[2].metric("Troughs", f"{label_counts.get(2, 0):,}")
cols[3].metric("Neither", f"{label_counts.get(0, 0):,}")

st.plotly_chart(make_label_distribution(label_counts), use_container_width=True)

# ── Label vs Prediction ──────────────────────────────────

st.subheader(f"Label vs Prediction — {stock_label}")

dates = sorted(sym_pred["date"].unique()) if "date" in sym_pred.columns else []
kb_nav_apply_date(kb_dir, dates, "pred_chart_date")
if dates:
    selected_date = st.select_slider("Date", options=dates, value=dates[-1], key="pred_chart_date")
    day_pred = sym_pred[sym_pred["date"] == selected_date]
else:
    selected_date = ""
    day_pred = sym_pred

# Show which split the selected date belongs to
split_dates = load_split_dates(market, lc, mc)
date_str = str(selected_date) if selected_date else ""
split_name = next((s for s, ds in split_dates.items() if date_str in ds), None)
if split_name:
    color = {"train": "blue", "val": "orange", "test": "red"}[split_name]
    st.markdown(f"Data split: :{color}[**{split_name.upper()}**]")

label_df = load_labeled(market, lc)
if label_df.empty:
    st.info("No labeled data available for comparison.")
else:
    sym_label = label_df[label_df["symbol"] == symbol].copy()
    if not sym_label.empty and dates:
        day_label = sym_label[sym_label["date"] == selected_date] if "date" in sym_label.columns else sym_label
    else:
        day_label = sym_label

    if day_label.empty:
        st.info(f"No labeled data for {stock_label} on {selected_date}")
    elif not day_pred.empty:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Ground Truth (Label)**")
            st.plotly_chart(
                make_candlestick_with_labels(day_label, f"{stock_label} — Label"),
                use_container_width=True,
            )
        with col_right:
            st.markdown("**Model Prediction**")
            st.plotly_chart(
                make_candlestick_with_probs(day_pred, f"{stock_label} — Prediction"),
                use_container_width=True,
            )

# ── Data table ───────────────────────────────────────────

with st.expander("Prediction Data Table"):
    st.dataframe(sym_pred, use_container_width=True, height=400)
