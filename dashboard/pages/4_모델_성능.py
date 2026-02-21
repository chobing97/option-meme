"""Phase 3: Model performance — PR-AUC, backtest, feature importance."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    make_feature_importance_bar,
    make_pr_curve,
    make_time_error_chart,
)
from dashboard.components.filters import market_selector
from dashboard.components.metrics import backtest_summary
from dashboard.data_loader import (
    get_feature_importance,
    get_model_status,
    get_pr_curve_data,
    run_model_evaluation,
)

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Phase 3: Model Performance")

# ── Sidebar ───────────────────────────────────────────────

market = market_selector(key="model_market")

# ── Model file check ─────────────────────────────────────

model_status = get_model_status(market)
st.subheader("Model Files")

cols = st.columns(4)
for i, (name, exists) in enumerate(model_status.items()):
    with cols[i % 4]:
        if exists:
            st.success(name)
        else:
            st.error(f"{name}: missing")

if not any(model_status.values()):
    st.warning(f"No models found for **{market.upper()}**. Run: `python run_pipeline.py model --market {market}`")
    st.stop()

# ── Full evaluation ───────────────────────────────────────

lgb_peak = model_status.get("lgb_peak", False)
lgb_trough = model_status.get("lgb_trough", False)

if not (lgb_peak and lgb_trough):
    st.warning("LightGBM peak and trough models are required for evaluation.")
    st.stop()

st.subheader("Evaluation (LightGBM)")
st.caption("Results are cached for 1 hour.")

eval_result = run_model_evaluation(market)
if eval_result is None:
    st.warning("Evaluation failed — check that featured data and models exist.")
    st.stop()

# ── Split info ────────────────────────────────────────────

split = eval_result.get("split_info", {})
if split:
    scols = st.columns(3)
    scols[0].metric("Train", f"{split['train_size']:,}", f"{split['train_dates'][0][:10]} ~ {split['train_dates'][1][:10]}")
    scols[1].metric("Val", f"{split['val_size']:,}", f"{split['val_dates'][0][:10]} ~ {split['val_dates'][1][:10]}")
    scols[2].metric("Test", f"{split['test_size']:,}", f"{split['test_dates'][0][:10]} ~ {split['test_dates'][1][:10]}")

st.divider()

# ── PR Curves ─────────────────────────────────────────────

st.subheader("Precision-Recall Curves")
pr_col1, pr_col2 = st.columns(2)

for col, target_label, label_name in [(pr_col1, 1, "Peak"), (pr_col2, 2, "Trough")]:
    with col:
        pr_data = get_pr_curve_data(market, target_label)
        if pr_data:
            st.plotly_chart(make_pr_curve(pr_data, label_name), use_container_width=True)
        else:
            st.info(f"No PR data for {label_name}")

# ── Threshold metrics ─────────────────────────────────────

st.subheader("Threshold Metrics")
for target, label_name in [("peak", "Peak"), ("trough", "Trough")]:
    pr_metrics = eval_result.get(target, {}).get("pr_metrics", {})
    if pr_metrics:
        st.markdown(f"**{label_name}** — PR-AUC: **{pr_metrics['pr_auc']:.4f}** | Positive rate: {pr_metrics['positive_rate']:.4f}")
        tm = pr_metrics.get("threshold_metrics", [])
        if tm:
            st.dataframe(pd.DataFrame(tm), use_container_width=True, hide_index=True)

st.divider()

# ── Time error ────────────────────────────────────────────

st.subheader("Time Error Analysis")
te_col1, te_col2 = st.columns(2)

for col, target, label_name in [(te_col1, "peak", "Peak"), (te_col2, "trough", "Trough")]:
    with col:
        te = eval_result.get(target, {}).get("time_error", {})
        if "error" in te:
            st.info(f"{label_name}: {te['error']}")
        elif te:
            st.markdown(f"**{label_name}** — MAE: {te['mae_bars']:.2f} bars | On-time: {te['on_time_pct']:.1%}")
            st.plotly_chart(make_time_error_chart(te), use_container_width=True)

st.divider()

# ── Backtest ──────────────────────────────────────────────

st.subheader("Backtest Results")
bt = eval_result.get("backtest", {})
backtest_summary(bt)

st.divider()

# ── Feature importance ────────────────────────────────────

st.subheader("Feature Importance (Top 20)")
fi_col1, fi_col2 = st.columns(2)

for col, target_label, label_name in [(fi_col1, 1, "Peak"), (fi_col2, 2, "Trough")]:
    with col:
        imp_df = get_feature_importance(market, target_label)
        if not imp_df.empty:
            st.plotly_chart(make_feature_importance_bar(imp_df), use_container_width=True)
        else:
            st.info(f"No importance data for {label_name}")
