"""Phase 6: Backtest — run and analyze backtests using the new backtest system."""

import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.components.filters import (
    label_config_selector,
    model_config_selector,
    reload_button,
    timeframe_selector,
)
from dashboard.data_loader import (
    find_backtest_defaults,
    get_backtest_symbols,
    has_options_data,
    load_options_ohlcv_by_strike,
    load_prediction_for_backtest,
    load_raw_bars,
    run_dashboard_backtest,
)

st.set_page_config(page_title="Backtest", layout="wide")
st.title("Phase 6: Backtest")

# ── Auto-detect valid defaults ─────────────────────────────

defaults = find_backtest_defaults(market="us")

# ── Sidebar ────────────────────────────────────────────────

reload_button()

# Timeframe: pre-select from defaults
if defaults and "bt_timeframe_init" not in st.session_state:
    st.session_state["timeframe"] = defaults["timeframe"]
    st.session_state["bt_timeframe_init"] = True
timeframe = timeframe_selector(key="timeframe")

# Symbol selector (only symbols with options data)
symbols = get_backtest_symbols("us")
if not symbols:
    st.warning("No symbols with options data found.")
    st.stop()

# Pre-select symbol from defaults
default_sym_idx = 0
if defaults and defaults["symbol"] in symbols:
    default_sym_idx = symbols.index(defaults["symbol"])
symbol = st.sidebar.selectbox("Symbol", symbols, index=default_sym_idx, key="bt_symbol")

# Label/Model config: pre-select from defaults
if defaults and "bt_lc_init" not in st.session_state:
    st.session_state["bt_lc"] = defaults["label_config"]
    st.session_state["bt_mc"] = defaults["model_config"]
    st.session_state["bt_lc_init"] = True
lc = label_config_selector(key="bt_lc", timeframe=timeframe)
mc = model_config_selector(key="bt_mc", timeframe=timeframe)

# Date range: pre-populate from defaults (options data range)
default_start = defaults["date_min"] if defaults else None
default_end = defaults["date_max"] if defaults else None
st.sidebar.markdown("### Date Range")
start_date = st.sidebar.date_input("Start", value=default_start, key="bt_start_date")
end_date = st.sidebar.date_input("End", value=default_end, key="bt_end_date")

# Strategy parameters
st.sidebar.markdown("### Strategy")
threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.3, 0.05, key="bt_threshold")
tp_pct = st.sidebar.slider("Take Profit %", 0.01, 0.50, 0.10, 0.01, key="bt_tp")
sl_pct = st.sidebar.slider("Stop Loss %", -0.50, -0.01, -0.05, 0.01, key="bt_sl")

# Run button
run_clicked = st.sidebar.button("Run Backtest", type="primary")


# ── Run backtest ──────────────────────────────────────────

if run_clicked:
    pred_df = load_prediction_for_backtest(
        market="us", symbol=symbol, timeframe=timeframe,
        label_config=lc, model_config=mc, model_type="gbm",
    )
    if pred_df.empty:
        st.error(f"No prediction data found for {symbol} ({lc}/{mc}/{timeframe}). Run batch predict first.")
        st.stop()

    # Apply date filters (strip tz for comparison)
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
    dt_col = pred_df["datetime"]
    if dt_col.dt.tz is not None:
        dt_naive = dt_col.dt.tz_localize(None)
    else:
        dt_naive = dt_col
    if start_date is not None:
        pred_df = pred_df[dt_naive >= pd.Timestamp(start_date)]
        dt_naive = dt_naive[pred_df.index]
    if end_date is not None:
        pred_df = pred_df[dt_naive <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    if pred_df.empty:
        st.error("No prediction data in the selected date range.")
        st.stop()

    with st.spinner("Running backtest..."):
        results = run_dashboard_backtest(
            market="us", symbol=symbol, pred_df=pred_df,
            threshold=threshold, tp_pct=tp_pct, sl_pct=sl_pct,
        )
    st.session_state["bt_results"] = results
    st.session_state["bt_symbol_used"] = symbol
    st.session_state["bt_config_used"] = f"{lc}/{mc}"


# ── Display results ──────────────────────────────────────

if "bt_results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
    st.stop()

results = st.session_state["bt_results"]
metrics = results["metrics"]
trades_df = results["trades_df"]
equity_df = results["equity_df"]
bt_symbol = st.session_state.get("bt_symbol_used", symbol)
bt_config = st.session_state.get("bt_config_used", "")

st.caption(f"Symbol: **{bt_symbol}** | Config: **{bt_config}** | Threshold: {metrics.get('total_trades', 0)} trades")


# ── 1. Summary Metrics ──────────────────────────────────

st.subheader("Summary Metrics")

row1 = st.columns(4)
row1[0].metric("Total Return", f"{metrics['total_return']:+.2%}")
row1[1].metric("Win Rate", f"{metrics['win_rate']:.1%}")
row1[2].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2%}")
row1[3].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

row2 = st.columns(4)
row2[0].metric("Total Trades", f"{metrics['total_trades']}")
row2[1].metric("Avg Win", f"{metrics['avg_win_pct']:+.2%}")
row2[2].metric("Avg Loss", f"{metrics['avg_loss_pct']:+.2%}")
pf = metrics["profit_factor"]
pf_str = f"{pf:.2f}" if pf != float("inf") else "Inf"
row2[3].metric("Profit Factor", pf_str)


# ── 2. Equity Curve ─────────────────────────────────────

st.subheader("Equity Curve")

if equity_df.empty:
    st.info("No equity data available.")
else:
    eq = equity_df.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])

    # Compute running peak for drawdown fill
    eq["equity_peak"] = eq["equity"].cummax()

    fig_eq = go.Figure()

    # Drawdown area (between peak and equity)
    fig_eq.add_trace(go.Scatter(
        x=eq["timestamp"], y=eq["equity_peak"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_eq.add_trace(go.Scatter(
        x=eq["timestamp"], y=eq["equity"],
        mode="lines", line=dict(color="royalblue", width=1.5),
        fill="tonexty", fillcolor="rgba(255,100,100,0.15)",
        name="Equity",
    ))

    # BUY markers
    buys = eq[eq["action"] == "BUY"]
    if not buys.empty:
        fig_eq.add_trace(go.Scatter(
            x=buys["timestamp"], y=buys["equity"],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=10, color="green"),
        ))

    # SELL markers
    sells = eq[eq["action"] == "SELL"]
    if not sells.empty:
        fig_eq.add_trace(go.Scatter(
            x=sells["timestamp"], y=sells["equity"],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=10, color="red"),
        ))

    fig_eq.update_layout(
        height=400, margin=dict(l=40, r=20, t=30, b=30),
        xaxis_title="", yaxis_title="Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_eq, use_container_width=True)


# ── 3. Daily Chart ───────────────────────────────────────

st.subheader("Daily Chart")

if equity_df.empty or trades_df.empty:
    st.info("No trade data available for daily chart.")
else:
    eq = equity_df.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq["date"] = eq["timestamp"].dt.date
    available_dates = sorted(eq["date"].unique())

    if not available_dates:
        st.info("No dates available.")
    else:
        # Only show dates that have trades
        td = trades_df.copy()
        td["entry_time"] = pd.to_datetime(td["entry_time"])
        trade_dates = sorted(td["entry_time"].dt.date.unique())
        # Show all dates but default to first trade date
        default_idx = 0
        if trade_dates and trade_dates[0] in available_dates:
            default_idx = available_dates.index(trade_dates[0])

        selected_date = st.selectbox(
            "Select date", available_dates, index=default_idx, key="bt_daily_date"
        )

        col_left, col_right = st.columns(2)

        # Stock candlestick with prob overlay + BUY/SELL markers
        with col_left:
            next_day = selected_date + timedelta(days=1)
            stock_bars = load_raw_bars("us", bt_symbol, str(selected_date), str(next_day), timeframe)

            day_eq = eq[eq["date"] == selected_date]

            if stock_bars.empty:
                st.info(f"No stock data for {selected_date}")
            else:
                stock_bars["datetime"] = pd.to_datetime(stock_bars["datetime"])

                fig_stock = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=[f"{bt_symbol} Stock", "Probabilities"],
                )

                # Candlestick
                fig_stock.add_trace(go.Candlestick(
                    x=stock_bars["datetime"],
                    open=stock_bars["open"], high=stock_bars["high"],
                    low=stock_bars["low"], close=stock_bars["close"],
                    name="Stock", showlegend=False,
                ), row=1, col=1)

                # BUY/SELL markers on stock chart
                if not day_eq.empty:
                    day_buys = day_eq[day_eq["action"] == "BUY"]
                    day_sells = day_eq[day_eq["action"] == "SELL"]

                    if not day_buys.empty:
                        fig_stock.add_trace(go.Scatter(
                            x=day_buys["timestamp"], y=day_buys["underlying_close"],
                            mode="markers", name="BUY",
                            marker=dict(symbol="triangle-up", size=12, color="green"),
                        ), row=1, col=1)

                    if not day_sells.empty:
                        fig_stock.add_trace(go.Scatter(
                            x=day_sells["timestamp"], y=day_sells["underlying_close"],
                            mode="markers", name="SELL",
                            marker=dict(symbol="triangle-down", size=12, color="red"),
                        ), row=1, col=1)

                # Probability overlay
                if not day_eq.empty:
                    fig_stock.add_trace(go.Scatter(
                        x=day_eq["timestamp"], y=day_eq["peak_prob"],
                        mode="lines", name="Peak Prob",
                        line=dict(color="orange", width=1),
                    ), row=2, col=1)
                    fig_stock.add_trace(go.Scatter(
                        x=day_eq["timestamp"], y=day_eq["trough_prob"],
                        mode="lines", name="Trough Prob",
                        line=dict(color="blue", width=1),
                    ), row=2, col=1)
                    # Threshold line
                    fig_stock.add_hline(y=threshold, line_dash="dash", line_color="gray", row=2, col=1)

                fig_stock.update_layout(
                    height=500, margin=dict(l=40, r=20, t=30, b=30),
                    xaxis2_rangeslider_visible=False,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_stock, use_container_width=True)

        # Option candlestick
        with col_right:
            # Find trades for this date to get strike
            day_trades = td[td["entry_time"].dt.date == selected_date]

            if day_trades.empty:
                st.info(f"No trades on {selected_date}")
            else:
                trade_strike = day_trades["entry_strike"].iloc[0]
                option_ohlcv = load_options_ohlcv_by_strike("us", bt_symbol, str(selected_date), trade_strike)

                if option_ohlcv.empty:
                    st.info(f"No option data for strike {trade_strike:.0f}")
                else:
                    option_ohlcv["datetime"] = pd.to_datetime(option_ohlcv["datetime"])
                    contract_info = option_ohlcv.attrs.get("contract_info", f"Put K={trade_strike:.0f}")

                    fig_opt = go.Figure()
                    fig_opt.add_trace(go.Candlestick(
                        x=option_ohlcv["datetime"],
                        open=option_ohlcv["open"], high=option_ohlcv["high"],
                        low=option_ohlcv["low"], close=option_ohlcv["close"],
                        name="Option", showlegend=False,
                    ))

                    # Entry/exit markers from trades
                    for _, trade in day_trades.iterrows():
                        fig_opt.add_trace(go.Scatter(
                            x=[trade["entry_time"]], y=[trade["entry_price"]],
                            mode="markers", name="Entry",
                            marker=dict(symbol="triangle-up", size=12, color="green"),
                            showlegend=False,
                        ))
                        if pd.notna(trade.get("exit_time")):
                            fig_opt.add_trace(go.Scatter(
                                x=[trade["exit_time"]], y=[trade["exit_price"]],
                                mode="markers", name="Exit",
                                marker=dict(symbol="triangle-down", size=12, color="red"),
                                showlegend=False,
                            ))

                    fig_opt.update_layout(
                        title=contract_info,
                        height=500, margin=dict(l=40, r=20, t=40, b=30),
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_opt, use_container_width=True)


# ── 4. Trade History Table ───────────────────────────────

st.subheader("Trade History")

if trades_df.empty:
    st.info("No trades executed.")
else:
    display = trades_df.copy()
    display["#"] = range(1, len(display) + 1)

    # Format columns
    display_cols = ["#", "entry_time", "entry_price", "entry_strike", "exit_time", "exit_price", "pnl_pct", "exit_reason", "holding_minutes"]
    available_cols = [c for c in display_cols if c in display.columns]
    display = display[available_cols].copy()

    # Rename for readability
    col_rename = {
        "entry_time": "Entry Time",
        "entry_price": "Entry Price",
        "entry_strike": "Strike",
        "exit_time": "Exit Time",
        "exit_price": "Exit Price",
        "pnl_pct": "PnL%",
        "exit_reason": "Reason",
        "holding_minutes": "Holding (min)",
    }
    display = display.rename(columns=col_rename)

    if "PnL%" in display.columns:
        display["PnL%"] = display["PnL%"].map("{:+.2%}".format)

    st.dataframe(display, use_container_width=True, hide_index=True)


# ── 5. Distribution Analysis ─────────────────────────────

st.subheader("Distribution Analysis")

if trades_df.empty:
    st.info("No trades for distribution analysis.")
else:
    dist_left, dist_right = st.columns(2)

    with dist_left:
        fig_pnl = go.Figure()
        pnl_vals = trades_df["pnl_pct"].dropna() * 100  # convert to percentage
        colors = ["green" if v >= 0 else "red" for v in pnl_vals]
        fig_pnl.add_trace(go.Histogram(
            x=pnl_vals, nbinsx=30, name="PnL%",
            marker_color="rgba(100,149,237,0.7)",
        ))
        fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_pnl.update_layout(
            title="PnL% Distribution",
            xaxis_title="PnL (%)", yaxis_title="Count",
            height=350, margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    with dist_right:
        fig_hold = go.Figure()
        hold_vals = trades_df["holding_minutes"].dropna()
        fig_hold.add_trace(go.Histogram(
            x=hold_vals, nbinsx=30, name="Holding",
            marker_color="rgba(255,165,0,0.7)",
        ))
        fig_hold.update_layout(
            title="Holding Time Distribution",
            xaxis_title="Minutes", yaxis_title="Count",
            height=350, margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_hold, use_container_width=True)


# ── 6. Monthly Returns ──────────────────────────────────

st.subheader("Monthly Returns")

monthly = metrics.get("monthly_returns", {})
if not monthly:
    st.info("No monthly return data.")
else:
    months = sorted(monthly.keys())
    values = [monthly[m] for m in months]
    colors = ["green" if v >= 0 else "red" for v in values]

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=months, y=values,
        marker_color=colors, name="Monthly PnL",
    ))
    fig_monthly.update_layout(
        xaxis_title="Month", yaxis_title="PnL ($)",
        height=350, margin=dict(l=40, r=20, t=30, b=30),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)


# ── 7. Exit Reason Breakdown ────────────────────────────

st.subheader("Exit Reason Breakdown")

exit_reasons = metrics.get("exit_reasons", {})
if not exit_reasons:
    st.info("No exit reason data.")
else:
    labels = list(exit_reasons.keys())
    values = list(exit_reasons.values())

    fig_pie = go.Figure()
    fig_pie.add_trace(go.Pie(
        labels=labels, values=values,
        textinfo="label+percent+value",
        marker_colors=["#26a69a", "#ef5350", "#ff9800", "#42a5f5", "#ab47bc"],
    ))
    fig_pie.update_layout(
        height=350, margin=dict(l=20, r=20, t=30, b=30),
    )
    st.plotly_chart(fig_pie, use_container_width=True)
