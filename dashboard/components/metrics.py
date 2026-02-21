"""Metric cards and summary tables for the dashboard."""

import streamlit as st


def pipeline_status_card(raw_summary: dict, labeled_summary: dict, featured_summary: dict, model_status: dict) -> None:
    """Display 4-column pipeline status overview."""
    cols = st.columns(4)

    with cols[0]:
        st.markdown("#### Phase 0: Raw")
        if raw_summary.get("exists"):
            st.metric("Symbols", raw_summary["n_symbols"])
            st.metric("Total Bars", f"{raw_summary['total_bars']:,}")
            if raw_summary.get("date_range"):
                st.caption(f"{raw_summary['date_range'][0]} ~ {raw_summary['date_range'][1]}")
        else:
            st.warning("No data")

    with cols[1]:
        st.markdown("#### Phase 1: Labels")
        if labeled_summary.get("exists"):
            st.metric("Total Bars", f"{labeled_summary['total_bars']:,}")
            st.metric("Symbols", labeled_summary["n_symbols"])
            lc = labeled_summary.get("label_counts", {})
            st.caption(f"Peaks: {lc.get(1, 0):,} | Troughs: {lc.get(2, 0):,}")
        else:
            st.warning("No data")

    with cols[2]:
        st.markdown("#### Phase 2: Features")
        if featured_summary.get("exists"):
            st.metric("Total Rows", f"{featured_summary['total_rows']:,}")
            st.metric("Features", featured_summary["n_features"])
            st.caption(f"File: {featured_summary['file_size_mb']} MB")
        else:
            st.warning("No data")

    with cols[3]:
        st.markdown("#### Phase 3: Models")
        n_found = sum(1 for v in model_status.values() if v)
        if n_found > 0:
            st.metric("Models", f"{n_found} / {len(model_status)}")
            for name, exists in model_status.items():
                icon = "OK" if exists else "--"
                st.caption(f"{name}: {icon}")
        else:
            st.warning("No models")


def backtest_summary(backtest: dict) -> None:
    """Display backtest result metrics."""
    if "error" in backtest:
        st.warning(f"Backtest: {backtest['error']}")
        return

    cols = st.columns(4)
    cols[0].metric("Trades", backtest.get("n_trades", 0))
    cols[1].metric("Total Return", f"{backtest.get('total_return', 0):.2%}")
    cols[2].metric("Buy & Hold", f"{backtest.get('buy_hold_return', 0):.2%}")
    cols[3].metric("Win Rate", f"{backtest.get('win_rate', 0):.1%}")

    if backtest.get("n_trades", 0) > 0:
        cols2 = st.columns(4)
        cols2[0].metric("Avg Win", f"{backtest.get('avg_win', 0):.2%}")
        cols2[1].metric("Avg Loss", f"{backtest.get('avg_loss', 0):.2%}")
        cols2[2].metric("Profit Factor", f"{backtest.get('profit_factor', 0):.2f}")
        cols2[3].metric("Max Drawdown", f"{backtest.get('max_drawdown', 0):.2%}")
