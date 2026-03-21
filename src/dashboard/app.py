"""Option-meme Pipeline Dashboard — landing page."""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Option-Meme Dashboard", layout="wide")

from config.variants import get_label_configs, get_model_configs
from dashboard.components.filters import timeframe_selector
from dashboard.components.metrics import pipeline_status_card
from dashboard.data_loader import (
    get_featured_summary,
    get_labeled_summary,
    get_model_status,
    get_raw_summary,
)

st.title("Option-Meme Pipeline Dashboard")
st.markdown("4-stage pipeline overview: **Collector** -> **Labeler** -> **Features** -> **Model**")

st.divider()

timeframe = timeframe_selector(key="timeframe")

label_keys = sorted(get_label_configs(timeframe).keys())
model_keys = sorted(get_model_configs(timeframe).keys())

for market in ["us", "kr"]:
    st.subheader(f"Market: {market.upper()} [{timeframe}]")
    raw = get_raw_summary(market, timeframe)

    for lc in label_keys:
        for mc in model_keys:
            st.markdown(f"**{lc} x {mc}**")
            labeled = get_labeled_summary(market, lc, timeframe)
            featured = get_featured_summary(market, lc, mc, timeframe)
            models = get_model_status(market, lc, mc, timeframe)

            pipeline_status_card(raw, labeled, featured, models)
    st.divider()

st.caption("Use the sidebar pages to explore each pipeline stage in detail.")
