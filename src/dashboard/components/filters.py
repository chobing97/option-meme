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
    return st.sidebar.radio("Market", ["us", "kr"], format_func=str.upper, key=key, horizontal=True)


def timeframe_selector(key: str = "timeframe") -> str:
    """Timeframe selector (1m/5m). Value auto-stored in session_state via widget key."""
    from config.settings import SUPPORTED_TIMEFRAMES
    return st.sidebar.selectbox("Timeframe", SUPPORTED_TIMEFRAMES, key=key)


def label_config_selector(key: str = "label_config", timeframe: str = "1m") -> str:
    """L1/L2/L3 radio selector. Loads keys from variants.py based on timeframe."""
    from config.variants import get_label_configs
    options = sorted(get_label_configs(timeframe).keys())
    return st.sidebar.radio("Label Config", options, key=key, horizontal=True)


def model_config_selector(key: str = "model_config", timeframe: str = "1m") -> str:
    """M1~M4 radio selector. Loads keys from variants.py based on timeframe."""
    from config.variants import get_model_configs
    options = sorted(get_model_configs(timeframe).keys())
    return st.sidebar.radio("Model Config", options, key=key, horizontal=True)


def model_type_selector(key: str = "model_type") -> str:
    """Radio button to pick model type (gbm/lstm/ensemble). Always shows all options."""
    all_types = ["gbm", "lstm", "ensemble"]
    labels = {"gbm": "GBM", "lstm": "LSTM", "ensemble": "Ensemble"}
    return st.sidebar.radio(
        "Model Type", all_types,
        format_func=lambda x: labels.get(x, x),
        key=key, horizontal=True,
    )


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


def load_from_query_params(key: str, default=None, cast=str):
    """Load a value from URL query params into session_state before widget creation.

    Only sets session_state if:
    - key is NOT already in session_state (don't override user interaction)
    - key IS in query_params (URL has the value)

    Args:
        key: session_state key (must match the widget key)
        default: not used directly, just for documentation
        cast: type conversion function (str, int, float)
    """
    if key not in st.session_state and key in st.query_params:
        raw = st.query_params[key]
        try:
            st.session_state[key] = cast(raw)
        except (ValueError, TypeError):
            pass

def sync_to_query_params(**kwargs):
    """Write current widget values to URL query params.

    Call at the end of the page after all widgets are rendered.
    Values are converted to strings for URL encoding.
    """
    for key, value in kwargs.items():
        if value is not None:
            st.query_params[key] = str(value)


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


def kb_nav_apply_selectbox(direction: str | None, options: list, key: str, axes: tuple[str, str] = ("up", "down")) -> None:
    """Handle keyboard nav for a sidebar selectbox. Must be called BEFORE the selectbox."""
    if direction not in axes or not options:
        return
    cur = st.session_state.get(key, options[0])
    try:
        idx = list(options).index(cur)
    except ValueError:
        idx = 0
    idx += -1 if direction == axes[0] else 1
    idx = max(0, min(idx, len(options) - 1))
    st.session_state[key] = options[idx]


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
