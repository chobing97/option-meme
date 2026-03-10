"""Phase 2: Feature distribution, correlation, and violin plots."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.charts import (
    make_correlation_heatmap,
    make_feature_boxplot,
    make_feature_histogram,
    make_violin_by_label,
)
from dashboard.components.filters import (
    feature_selector, label_config_selector, market_selector, model_config_selector, reload_button,
)
from dashboard.data_loader import get_feature_column_list, get_featured_summary, load_featured

st.set_page_config(page_title="Features", layout="wide")
st.title("Phase 2: Feature Analysis")

# ── Sidebar ───────────────────────────────────────────────

reload_button()
market = market_selector(key="feat_market")
lc = label_config_selector(key="feat_lc")
mc = model_config_selector(key="feat_mc")

summary = get_featured_summary(market, lc, mc)
if not summary.get("exists"):
    st.warning(f"No featured data for **{market.upper()}**. Run: `python run_pipeline.py features --market {market}`")
    st.stop()

st.sidebar.markdown(f"**Rows:** {summary['total_rows']:,}  |  **Features:** {summary['n_features']}  |  **Size:** {summary['file_size_mb']} MB")

all_features = get_feature_column_list(market, lc, mc)
selected_features = feature_selector(all_features, key="feat_sel")

if not selected_features:
    st.info("Select features from the sidebar.")
    st.stop()

# ── Load data ─────────────────────────────────────────────

df = load_featured(market, lc, mc, columns=selected_features)
if df.empty:
    st.warning("Failed to load featured data.")
    st.stop()

st.caption(f"Loaded {len(df):,} rows (sampled if > 50k)")

# ── Feature summary table ────────────────────────────────

st.subheader("Feature Summary")
valid_cols = [c for c in selected_features if c in df.columns]
if valid_cols:
    st.dataframe(df[valid_cols].describe().T, use_container_width=True)

# ── Histograms / Box plots ───────────────────────────────

st.subheader("Distribution")
tab_hist, tab_box = st.tabs(["Histograms", "Box Plots"])

with tab_hist:
    for feat in valid_cols:
        st.plotly_chart(make_feature_histogram(df[feat], feat), use_container_width=True)

with tab_box:
    if valid_cols:
        st.plotly_chart(make_feature_boxplot(df, valid_cols), use_container_width=True)

# ── Correlation heatmap ──────────────────────────────────

st.subheader("Correlation Heatmap")
if len(valid_cols) >= 2:
    st.plotly_chart(make_correlation_heatmap(df, valid_cols), use_container_width=True)
else:
    st.info("Select at least 2 features to see correlations.")

# ── Violin by label ──────────────────────────────────────

st.subheader("Feature by Label")
if "label" in df.columns and valid_cols:
    violin_feat = st.selectbox("Feature", valid_cols, key="violin_feat")
    st.plotly_chart(make_violin_by_label(df, violin_feat), use_container_width=True)
