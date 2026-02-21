"""Option-meme Pipeline Dashboard — landing page."""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Option-Meme Dashboard", layout="wide")

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

for market in ["kr", "us"]:
    st.subheader(f"Market: {market.upper()}")
    raw = get_raw_summary(market)
    labeled = get_labeled_summary(market)
    featured = get_featured_summary(market)
    models = get_model_status(market)

    pipeline_status_card(raw, labeled, featured, models)
    st.divider()

st.caption("Use the sidebar pages to explore each pipeline stage in detail.")
