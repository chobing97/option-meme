"""Sidebar filter widgets for the dashboard."""

from datetime import date, timedelta

import streamlit as st


def market_selector(key: str = "market") -> str:
    """Radio button to pick KR or US market."""
    return st.sidebar.radio("Market", ["kr", "us"], format_func=str.upper, key=key, horizontal=True)


def symbol_selector(symbols: list[str], key: str = "symbol") -> str | None:
    """Dropdown to pick a symbol."""
    if not symbols:
        st.sidebar.warning("No symbols available")
        return None
    return st.sidebar.selectbox("Symbol", symbols, key=key)


def date_range_selector(
    min_date: date | None = None,
    max_date: date | None = None,
    key: str = "dates",
) -> tuple[date, date]:
    """Start/end date pickers."""
    if min_date is None:
        min_date = date.today() - timedelta(days=365)
    if max_date is None:
        max_date = date.today()

    start = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key=f"{key}_start")
    end = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key=f"{key}_end")
    return start, end


def feature_selector(feature_cols: list[str], key: str = "features") -> list[str]:
    """Group features by prefix and let user multi-select groups + individual features."""
    prefix_map = {
        "pf_": "Price (pf_)",
        "vf_": "Volume (vf_)",
        "tf_": "Technical (tf_)",
        "tmf_": "Time (tmf_)",
        "mf_": "Market (mf_)",
    }

    groups: dict[str, list[str]] = {}
    for col in feature_cols:
        for prefix, label in prefix_map.items():
            if col.startswith(prefix):
                groups.setdefault(label, []).append(col)
                break

    st.sidebar.markdown("### Feature Groups")
    selected_groups = st.sidebar.multiselect(
        "Groups",
        options=list(groups.keys()),
        default=list(groups.keys())[:2] if groups else [],
        key=f"{key}_groups",
    )

    selected_cols = []
    for g in selected_groups:
        selected_cols.extend(groups[g])

    if selected_cols:
        selected_cols = st.sidebar.multiselect(
            "Individual features",
            options=selected_cols,
            default=selected_cols[:5],
            key=f"{key}_individual",
        )

    return selected_cols
