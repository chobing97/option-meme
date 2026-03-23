"""Phase 5: Model predictions viewer with label comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
    label_config_selector, load_from_query_params, market_selector,
    model_config_selector, model_type_selector, reload_button, symbol_selector,
    sync_to_query_params, timeframe_selector,
)
from dashboard.data_loader import (
    find_configs_for_model_type, get_available_model_types, get_stock_name_map,
    load_labeled, load_predicted, load_split_dates,
)

st.set_page_config(page_title="Predictions", layout="wide")
st.title("Phase 5: Predictions")

kb_dir = kb_nav_read()

# ── Query param persistence ──────────────────────────────
load_from_query_params("timeframe", cast=str)
load_from_query_params("pred_market", cast=str)
load_from_query_params("pred_mt", cast=str)
load_from_query_params("pred_lc", cast=str)
load_from_query_params("pred_mc", cast=str)
load_from_query_params("pred_symbol", cast=str)

# ── Sidebar ───────────────────────────────────────────────

reload_button()
timeframe = timeframe_selector(key="timeframe")
market = market_selector(key="pred_market")
mt = model_type_selector(key="pred_mt")

# Pre-check: if current config has no data for selected model_type, auto-switch before widgets
_cur_lc = st.session_state.get("pred_lc", "L1")
_cur_mc = st.session_state.get("pred_mc", "M1")
available_mt = get_available_model_types(market, _cur_lc, _cur_mc, timeframe)
if mt not in available_mt:
    valid_configs = find_configs_for_model_type(market, mt, timeframe)
    if valid_configs:
        st.session_state["pred_lc"] = valid_configs[0][0]
        st.session_state["pred_mc"] = valid_configs[0][1]

lc = label_config_selector(key="pred_lc", timeframe=timeframe)
mc = model_config_selector(key="pred_mc", timeframe=timeframe)

# Verify data exists
available_mt = get_available_model_types(market, lc, mc, timeframe)
if mt not in available_mt:
    mt_display = {"gbm": "GBM", "lstm": "LSTM", "ensemble": "Ensemble"}.get(mt, mt)
    st.warning(
        f"No **{mt_display}** prediction data for **{market.upper()}** [{timeframe}]. "
        f"Run: `./optionmeme batch_predict --market {market} --model {mt}`"
    )
    st.stop()

pred_df = load_predicted(market, lc, mc, mt, timeframe)

if pred_df.empty:
    st.warning(
        f"No prediction data for **{market.upper()}** ({mt.upper()}) [{timeframe}]. "
        f"Run: `./optionmeme batch_predict --market {market} --model {mt}`"
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

st.subheader(f"Prediction Distribution [{timeframe}]")
label_counts = sym_pred["label"].value_counts().to_dict()

cols = st.columns(4)
cols[0].metric("Total Bars", f"{len(sym_pred):,}")
cols[1].metric("Peaks", f"{label_counts.get(1, 0):,}")
cols[2].metric("Troughs", f"{label_counts.get(2, 0):,}")
cols[3].metric("Neither", f"{label_counts.get(0, 0):,}")

st.plotly_chart(make_label_distribution(label_counts), use_container_width=True)

# ── Label vs Prediction ──────────────────────────────────

mt_display = {"gbm": "GBM", "lstm": "LSTM", "ensemble": "Ensemble"}.get(mt, mt)
st.subheader(f"Label vs Prediction ({mt_display}) [{timeframe}] — {stock_label}")

dates = sorted(sym_pred["date"].unique()) if "date" in sym_pred.columns else []
kb_nav_apply_date(kb_dir, dates, "pred_chart_date")
if dates:
    selected_date = st.select_slider("Date", options=dates, value=dates[-1], key="pred_chart_date")
    day_pred = sym_pred[sym_pred["date"] == selected_date]
else:
    selected_date = ""
    day_pred = sym_pred

# Show which split the selected date belongs to
split_dates = load_split_dates(market, lc, mc, timeframe)
date_str = str(selected_date) if selected_date else ""
split_name = next((s for s, ds in split_dates.items() if date_str in ds), None)
if split_name:
    color = {"train": "blue", "val": "orange", "test": "red"}[split_name]
    st.markdown(f"Data split: :{color}[**{split_name.upper()}**]")

label_df = load_labeled(market, lc, symbol=symbol, timeframe=timeframe)
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
            st.markdown(f"**Model Prediction ({mt_display})**")
            st.plotly_chart(
                make_candlestick_with_probs(day_pred, f"{stock_label} — {mt_display}"),
                use_container_width=True,
            )

# ── Data table ───────────────────────────────────────────

with st.expander("Prediction Data Table"):
    st.dataframe(sym_pred, use_container_width=True, height=400)

sync_to_query_params(
    timeframe=timeframe, pred_market=market, pred_mt=mt,
    pred_lc=lc, pred_mc=mc, pred_symbol=symbol,
)
