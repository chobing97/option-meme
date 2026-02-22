"""Sidebar filter widgets for the dashboard."""

from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def reload_button() -> None:
    """Sidebar button to clear all cached data and rerun."""
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()


def market_selector(key: str = "market") -> str:
    """Radio button to pick KR or US market."""
    return st.sidebar.radio("Market", ["kr", "us"], format_func=str.upper, key=key, horizontal=True)


def label_config_selector(key: str = "label_config") -> str:
    """L1/L2 radio selector. Loads keys from variants.py."""
    from config.variants import LABEL_CONFIGS
    options = sorted(LABEL_CONFIGS.keys())
    return st.sidebar.radio("Label Config", options, key=key, horizontal=True)


def model_config_selector(key: str = "model_config") -> str:
    """M1~M4 radio selector. Loads keys from variants.py."""
    from config.variants import MODEL_CONFIGS
    options = sorted(MODEL_CONFIGS.keys())
    return st.sidebar.radio("Model Config", options, key=key, horizontal=True)


def symbol_selector(
    symbols: list[str],
    key: str = "symbol",
    name_map: dict[str, str] | None = None,
) -> str | None:
    """Dropdown to pick a symbol. Shows 'symbol(name)' when name_map is provided."""
    if not symbols:
        st.sidebar.warning("No symbols available")
        return None
    fmt = (lambda s: f"{s}({name_map[s]})" if s in name_map else s) if name_map else None
    return st.sidebar.selectbox("Symbol", symbols, key=key, format_func=fmt)


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


_kb_nav_func = components.declare_component(
    "kb_nav",
    path=str(Path(__file__).parent / "kb_nav"),
)


def kb_nav_read() -> str | None:
    """Render keyboard nav component, return direction ('left'/'right'/'up'/'down') or None."""
    nav = _kb_nav_func(key="__kb_nav", default=None)
    if nav is None:
        return None
    if nav == st.session_state.get("__kb_nav_handled"):
        return None
    st.session_state["__kb_nav_handled"] = nav
    return nav.split("_")[0]


def kb_nav_apply_symbol(direction: str | None, symbols: list, symbol_key: str, date_key: str) -> None:
    """Handle ↑↓: change symbol, clear date. Must be called BEFORE symbol_selector."""
    if direction not in ("up", "down") or not symbols:
        return
    cur = st.session_state.get(symbol_key, symbols[0])
    try:
        idx = list(symbols).index(cur)
    except ValueError:
        idx = 0
    idx += -1 if direction == "up" else 1
    idx = max(0, min(idx, len(symbols) - 1))
    st.session_state[symbol_key] = symbols[idx]
    st.session_state.pop(date_key, None)


def kb_nav_apply_date(direction: str | None, dates: list, date_key: str) -> None:
    """Handle ←→: change date. Must be called BEFORE select_slider."""
    if direction not in ("left", "right") or not dates:
        return
    cur = st.session_state.get(date_key, dates[-1])
    try:
        idx = list(dates).index(cur)
    except ValueError:
        idx = len(dates) - 1
    idx += 1 if direction == "right" else -1
    idx = max(0, min(idx, len(dates) - 1))
    st.session_state[date_key] = dates[idx]
