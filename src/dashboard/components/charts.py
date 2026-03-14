"""Plotly chart builder functions for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _trading_rangebreaks(df: pd.DataFrame) -> list[dict]:
    """Detect actual gaps in datetime and return rangebreaks to hide them."""
    if "datetime" not in df.columns or len(df) < 2:
        return []
    dt = df["datetime"].sort_values().reset_index(drop=True)
    diffs_sec = dt.diff().dt.total_seconds()
    median_sec = diffs_sec.median()

    if median_sec is None or median_sec <= 0:
        return []
    gap_indices = diffs_sec[diffs_sec > median_sec * 2].index
    offset = pd.Timedelta(seconds=median_sec)
    breaks = []
    for i in gap_indices:
        gap_start = dt.iloc[i - 1] + offset
        gap_ms = (diffs_sec.iloc[i] - median_sec) * 1000
        if gap_ms > 0:
            breaks.append(dict(values=[str(gap_start)], dvalue=gap_ms))
    return breaks


def _vol_outlier_cap(df: pd.DataFrame) -> tuple[float, bool]:
    """Return (cap, has_outliers). Outlier if top values average > 2× the 98th percentile."""
    vol = df["volume"]
    cap = vol.quantile(0.95)
    above = vol[vol > cap]
    has_outliers = not above.empty and above.mean() > cap * 2
    return cap, has_outliers


def _vol_colors(df: pd.DataFrame) -> list[str]:
    """Red for down bars, green for up bars."""
    return ["#ef5350" if c < o else "#26a69a" for c, o in zip(df["close"], df["open"])]


def make_candlestick(df: pd.DataFrame, title: str = "OHLCV") -> go.Figure:
    """Create candlestick chart with volume subplot (broken axis for outliers)."""
    vol = df["volume"]
    cap, has_outliers = _vol_outlier_cap(df)
    colors = _vol_colors(df)

    if has_outliers:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0, row_heights=[0.66, 0.10, 0.20],
        )
        # Glue vol rows (spacing=0), then add gap only between price and vol
        price_d = list(fig.layout.yaxis.domain)
        fig.update_layout(yaxis=dict(domain=[price_d[0] + 0.03, price_d[1]]))
        # Row 2: outlier range (log scale, bottom = cap)
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol, marker_color=colors, opacity=0.6, showlegend=False),
            row=2, col=1,
        )
        fig.update_yaxes(
            type="log", range=[np.log10(cap), np.log10(vol.max() * 1.1)],
            dtick=1, minor=dict(dtick="D1"),
            row=2, col=1,
        )
        # Row 3: normal range (top = cap)
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol.clip(upper=cap), marker_color=colors, name="Volume", opacity=0.6),
            row=3, col=1,
        )
        fig.update_yaxes(range=[0, cap * 1.05], title_text="Volume", row=3, col=1)
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol, marker_color=colors, name="Volume", opacity=0.6),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Price always row 1
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
    )
    fig.update_xaxes(rangebreaks=_trading_rangebreaks(df))

    # Fix x-axis range to full regular trading hours
    if "datetime" in df.columns and not df.empty:
        trade_date = pd.Timestamp(df["datetime"].iloc[0]).normalize()
        first_time = df["datetime"].min()
        # Detect market: KR starts ~09:00, US starts ~09:30
        start_hour = pd.Timestamp(first_time).hour
        if start_hour < 9 or (start_hour == 9 and pd.Timestamp(first_time).minute < 15):
            # KR market: 09:00 ~ 15:30
            market_open = trade_date + pd.Timedelta(hours=9)
            market_close = trade_date + pd.Timedelta(hours=15, minutes=30)
        else:
            # US market: 09:30 ~ 16:00
            market_open = trade_date + pd.Timedelta(hours=9, minutes=30)
            market_close = trade_date + pd.Timedelta(hours=16)
        fig.update_xaxes(range=[market_open, market_close])

    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig


def make_candlestick_with_labels(df: pd.DataFrame, title: str = "Labels") -> go.Figure:
    """Candlestick with peak (red triangle down) and trough (green triangle up) markers."""
    fig = make_candlestick(df, title)

    peaks = df[df["label"] == 1]
    troughs = df[df["label"] == 2]

    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["datetime"], y=peaks["high"] * 1.002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                name="Peak",
            ),
            row=1, col=1,
        )

    if not troughs.empty:
        fig.add_trace(
            go.Scatter(
                x=troughs["datetime"], y=troughs["low"] * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                name="Trough",
            ),
            row=1, col=1,
        )

    fig.update_layout(showlegend=True)
    return fig


def make_editable_candlestick(df: pd.DataFrame, title: str = "Labels (click to edit)") -> go.Figure:
    """Candlestick with clickable scatter for interactive label editing.

    Every bar gets a small, semi-transparent circle at `close` price.
    Each point carries `customdata = [datetime_str]` so the selection
    callback can identify which bar was clicked.
    Peak/trough markers are shown identically to the read-only chart.
    """
    fig = make_candlestick(df, title)

    # Clickable scatter on every bar (semi-transparent, small circles)
    dt_strings = df["datetime"].astype(str).tolist()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["close"],
            mode="markers",
            marker=dict(size=8, color="rgba(100,100,100,0.15)", symbol="circle"),
            customdata=dt_strings,
            name="Click to edit",
            hovertemplate="%{x}<br>close: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Peak markers
    peaks = df[df["label"] == 1]
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["datetime"], y=peaks["high"] * 1.002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                name="Peak",
            ),
            row=1, col=1,
        )

    # Trough markers
    troughs = df[df["label"] == 2]
    if not troughs.empty:
        fig.add_trace(
            go.Scatter(
                x=troughs["datetime"], y=troughs["low"] * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                name="Trough",
            ),
            row=1, col=1,
        )

    fig.update_layout(showlegend=True)
    return fig


def make_candlestick_with_probs(df: pd.DataFrame, title: str = "Predictions") -> go.Figure:
    """Candlestick with peak/trough markers and probability subplot."""
    vol = df["volume"]
    cap, has_outliers = _vol_outlier_cap(df)
    colors = _vol_colors(df)

    if has_outliers:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0,
            row_heights=[0.55, 0.07, 0.18, 0.10],
        )
        vol_upper, vol_lower, prob_row = 2, 3, 4
        # Glue vol rows (spacing=0), then add gaps around vol block
        price_d = list(fig.layout.yaxis.domain)
        prob_d = list(fig.layout.yaxis4.domain)
        fig.update_layout(
            yaxis=dict(domain=[price_d[0] + 0.03, price_d[1]]),
            yaxis4=dict(domain=[prob_d[0], prob_d[1] - 0.03]),
        )
    else:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.60, 0.25, 0.10],
        )
        vol_upper, vol_lower, prob_row = None, 2, 3

    # Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    peaks = df[df["label"] == 1]
    troughs = df[df["label"] == 2]
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["datetime"], y=peaks["high"] * 1.002,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                name="Peak",
            ),
            row=1, col=1,
        )
    if not troughs.empty:
        fig.add_trace(
            go.Scatter(
                x=troughs["datetime"], y=troughs["low"] * 0.998,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                name="Trough",
            ),
            row=1, col=1,
        )

    # Volume
    if has_outliers:
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol, marker_color=colors, opacity=0.6, showlegend=False),
            row=vol_upper, col=1,
        )
        fig.update_yaxes(
            type="log", range=[np.log10(cap), np.log10(vol.max() * 1.1)],
            dtick=1, minor=dict(dtick="D1"),
            row=vol_upper, col=1,
        )
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol.clip(upper=cap), marker_color=colors, name="Volume", opacity=0.6),
            row=vol_lower, col=1,
        )
        fig.update_yaxes(range=[0, cap * 1.05], title_text="Volume", row=vol_lower, col=1)
    else:
        fig.add_trace(
            go.Bar(x=df["datetime"], y=vol, marker_color=colors, name="Volume", opacity=0.6),
            row=vol_lower, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=vol_lower, col=1)

    # Probabilities (bar chart) — width in ms for datetime axis (1min = 60000ms, use ~30s)
    prob_bar_width = 30_000
    if "peak_prob" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["datetime"], y=df["peak_prob"],
                name="Peak prob",
                marker_color="rgba(239,83,80,0.6)",
                width=prob_bar_width,
            ),
            row=prob_row, col=1,
        )
    if "trough_prob" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["datetime"], y=df["trough_prob"],
                name="Trough prob",
                marker_color="rgba(38,166,154,0.6)",
                width=prob_bar_width,
            ),
            row=prob_row, col=1,
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#bdbdbd", row=prob_row, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        barmode="overlay",
        height=650,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=True,
    )
    fig.update_xaxes(rangebreaks=_trading_rangebreaks(df))

    # Fix x-axis range to full regular trading hours
    if "datetime" in df.columns and not df.empty:
        trade_date = pd.Timestamp(df["datetime"].iloc[0]).normalize()
        first_time = df["datetime"].min()
        start_hour = pd.Timestamp(first_time).hour
        if start_hour < 9 or (start_hour == 9 and pd.Timestamp(first_time).minute < 15):
            market_open = trade_date + pd.Timedelta(hours=9)
            market_close = trade_date + pd.Timedelta(hours=15, minutes=30)
        else:
            market_open = trade_date + pd.Timedelta(hours=9, minutes=30)
            market_close = trade_date + pd.Timedelta(hours=16)
        fig.update_xaxes(range=[market_open, market_close])

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Prob", range=[0, 1], row=prob_row, col=1)
    return fig


def make_option_candlestick(
    option_df: pd.DataFrame, title: str = "Option OHLCV",
    stock_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Standalone option candlestick chart with volume subplot.

    If stock_df is provided, x-axis range and rangebreaks are matched to the stock chart.
    """
    vol = option_df["volume"] if "volume" in option_df.columns else None
    has_vol = vol is not None and vol.sum() > 0

    if has_vol:
        colors = _vol_colors(option_df)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Bar(x=option_df["datetime"], y=vol, marker_color=colors, name="Volume", opacity=0.6),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=option_df["datetime"],
            open=option_df["open"], high=option_df["high"],
            low=option_df["low"], close=option_df["close"],
            name="Option OHLC",
        ),
        row=1, col=1,
    )

    contract_info = option_df.attrs.get("contract_info", "")
    y_label = contract_info if contract_info else "Option $"
    fig.update_yaxes(title_text=y_label, row=1, col=1)

    # Match x-axis to stock chart if provided
    ref_df = stock_df if stock_df is not None else option_df
    breaks = _trading_rangebreaks(ref_df)
    x_range = [ref_df["datetime"].min(), ref_df["datetime"].max()] if stock_df is not None else None

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=350,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
    )
    fig.update_xaxes(rangebreaks=breaks)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    return fig


def make_backtest_chart(
    df: pd.DataFrame,
    title: str = "Backtest",
    stock_ohlcv: pd.DataFrame | None = None,
    option_ohlcv: pd.DataFrame | None = None,
) -> go.Figure:
    """4-row subplot: price+trades, option price, model probs, equity+drawdown.

    All rows share x-axis for synchronized zoom/pan.
    Uses string x-labels (HH:MM) to avoid Plotly datetime axis gaps.

    Args:
        stock_ohlcv: Real stock OHLCV data (with datetime, open, high, low, close).
            If provided, Row 1 shows a candlestick chart instead of a line.
        option_ohlcv: Real option OHLCV data (with datetime, open, high, low, close).
            If provided, Row 3 shows a candlestick chart instead of position mark price.
    """
    # Build full regular-hours x-axis (09:30~15:59, 390 slots) for even spacing
    raw_ts = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]
    all_minutes = pd.date_range("2000-01-01 09:30", "2000-01-01 15:59", freq="1min")
    full_time_labels = all_minutes.strftime("%H:%M").tolist()
    time_to_idx = {t: i for i, t in enumerate(full_time_labels)}

    # Map backtest rows to full-session x positions
    df_time_keys = raw_ts.dt.strftime("%H:%M")
    ts = df_time_keys.map(time_to_idx)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.30, 0.30, 0.10, 0.30],
    )

    # Helper: map OHLCV data to full-session integer x-axis via HH:MM time key
    def _align_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame | None:
        aligned = ohlcv.copy()
        aligned["datetime"] = pd.to_datetime(aligned["datetime"])
        if aligned["datetime"].dt.tz is not None:
            aligned["datetime"] = aligned["datetime"].dt.tz_localize(None)
        aligned["time_key"] = aligned["datetime"].dt.strftime("%H:%M")
        aligned["x"] = aligned["time_key"].map(time_to_idx)
        return aligned.dropna(subset=["x"])

    # ── Row 1: Underlying price ──
    if stock_ohlcv is not None and not stock_ohlcv.empty:
        stk = _align_ohlcv(stock_ohlcv)
        if stk is not None and not stk.empty:
            fig.add_trace(
                go.Candlestick(
                    x=stk["x"], open=stk["open"], high=stk["high"],
                    low=stk["low"], close=stk["close"],
                    name="Underlying",
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=ts, y=df["underlying_close"], mode="lines", name="Underlying",
                           line=dict(color="#78909c", width=1.5)),
                row=1, col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(x=ts, y=df["underlying_close"], mode="lines", name="Underlying",
                       line=dict(color="#78909c", width=1.5)),
            row=1, col=1,
        )

    # Signal markers (small, semi-transparent)
    peaks = df[df["signal"] == "PEAK"]
    troughs = df[df["signal"] == "TROUGH"]
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=ts[peaks.index], y=peaks["underlying_close"],
                mode="markers", name="Peak Signal",
                marker=dict(symbol="circle", size=6, color="rgba(239,83,80,0.4)"),
                hovertemplate="PEAK<br>%{x}<br>price: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if not troughs.empty:
        fig.add_trace(
            go.Scatter(
                x=ts[troughs.index], y=troughs["underlying_close"],
                mode="markers", name="Trough Signal",
                marker=dict(symbol="circle", size=6, color="rgba(38,166,154,0.4)"),
                hovertemplate="TROUGH<br>%{x}<br>price: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Trade markers — use candle high/low like labeling chart (high*1.002, low*0.998)
    buys = df[df["action"] == "BUY_PUT"]
    sells = df[df["action"] == "SELL_PUT"]

    # Build high/low lookup from stock_ohlcv aligned to backtest index
    _stk_high = None
    _stk_low = None
    if stock_ohlcv is not None and not stock_ohlcv.empty:
        stk_aligned = _align_ohlcv(stock_ohlcv)
        if stk_aligned is not None and not stk_aligned.empty:
            _stk_high = stk_aligned.set_index("x")["high"]
            _stk_low = stk_aligned.set_index("x")["low"]

    if not buys.empty:
        if _stk_high is not None:
            buy_y = pd.Series(ts[buys.index], index=buys.index).map(_stk_high) * 1.002
            buy_y = buy_y.fillna(buys["underlying_close"])
        else:
            buy_y = buys["underlying_close"] * 1.002
        fig.add_trace(
            go.Scatter(
                x=ts[buys.index], y=buy_y,
                mode="markers", name="BUY PUT",
                marker=dict(symbol="triangle-down", size=12, color="#26a69a", line=dict(width=1, color="white")),
                customdata=np.column_stack([buys["strike"], buys["fill_price"], buys["reason"]]),
                hovertemplate="BUY PUT<br>%{x}<br>price: %{y:,.2f}<br>strike: %{customdata[0]}<br>fill: %{customdata[1]:.4f}<br>reason: %{customdata[2]}<extra></extra>",
            ),
            row=1, col=1,
        )
    if not sells.empty:
        if _stk_low is not None:
            sell_y = pd.Series(ts[sells.index], index=sells.index).map(_stk_low) * 0.998
            sell_y = sell_y.fillna(sells["underlying_close"])
        else:
            sell_y = sells["underlying_close"] * 0.998
        fig.add_trace(
            go.Scatter(
                x=ts[sells.index], y=sell_y,
                mode="markers", name="SELL PUT",
                marker=dict(symbol="triangle-up", size=12, color="#ef5350", line=dict(width=1, color="white")),
                customdata=np.column_stack([sells["strike"], sells["fill_price"], sells["reason"]]),
                hovertemplate="SELL PUT<br>%{x}<br>price: %{y:,.2f}<br>strike: %{customdata[0]}<br>fill: %{customdata[1]:.4f}<br>reason: %{customdata[2]}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Row 2: Option price ──
    if option_ohlcv is not None and not option_ohlcv.empty:
        opt = _align_ohlcv(option_ohlcv)
        if opt is not None and not opt.empty:
            fig.add_trace(
                go.Candlestick(
                    x=opt["x"], open=opt["open"], high=opt["high"],
                    low=opt["low"], close=opt["close"],
                    name="Option",
                    increasing_line_color="#5d8a80", decreasing_line_color="#8a5d5d",
                ),
                row=2, col=1,
            )
        else:
            opt_price = df["position_mark_price"].where(df["position_qty"] > 0)
            fig.add_trace(
                go.Scatter(x=ts, y=opt_price, mode="lines", name="Option Price",
                           line=dict(color="#7a6a8a", width=1.2), connectgaps=False),
                row=2, col=1,
            )
    else:
        opt_price = df["position_mark_price"].where(df["position_qty"] > 0)
        fig.add_trace(
            go.Scatter(x=ts, y=opt_price, mode="lines", name="Option Price",
                       line=dict(color="#7a6a8a", width=1.2), connectgaps=False),
            row=2, col=1,
        )
    # Entry→Exit connecting lines on option chart (draw first so dots overlay)
    if not buys.empty and not sells.empty:
        buy_list = buys.sort_index()
        sell_list = sells.sort_index()
        for _, buy_row in buy_list.iterrows():
            later_sells = sell_list[sell_list.index > buy_row.name]
            if later_sells.empty:
                continue
            sell_row = later_sells.iloc[0]
            pnl = sell_row["fill_price"] - buy_row["fill_price"]
            line_color = "#e040fb" if pnl >= 0 else "#ffea00"
            fig.add_trace(
                go.Scatter(
                    x=[ts[buy_row.name], ts[sell_row.name]],
                    y=[buy_row["fill_price"], sell_row["fill_price"]],
                    mode="lines", showlegend=False,
                    line=dict(color=line_color, width=3),
                    hoverinfo="skip",
                ),
                row=2, col=1,
            )

    # Entry/exit dots on option chart (drawn after lines so dots are on top)
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=ts[buys.index], y=buys["fill_price"],
                mode="markers", name="Entry",
                marker=dict(symbol="circle", size=11, color="#ffea00", line=dict(width=1.5, color="#333")),
                hovertemplate="Entry: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=ts[sells.index], y=sells["fill_price"],
                mode="markers", name="Exit",
                marker=dict(symbol="circle", size=11, color="#e040fb", line=dict(width=1.5, color="#333")),
                hovertemplate="Exit: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ── Row 3: Model probabilities (bar chart) ──
    fig.add_trace(
        go.Bar(
            x=ts, y=df["peak_prob"],
            name="Peak Prob",
            marker_color="rgba(239,83,80,0.6)",
            width=0.3,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ts, y=df["trough_prob"],
            name="Trough Prob",
            marker_color="rgba(38,166,154,0.6)",
            width=0.3,
        ),
        row=3, col=1,
    )
    fig.update_layout(barmode="overlay")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#bdbdbd", row=3, col=1)

    # ── Row 4: Equity curve + drawdown ──
    fig.add_trace(
        go.Scatter(
            x=ts, y=df["equity"],
            mode="lines", name="Equity",
            line=dict(color="#42a5f5", width=1.5),
        ),
        row=4, col=1,
    )
    dd_pct = df["drawdown_pct"] * 100
    if dd_pct.abs().max() > 0:
        fig.add_trace(
            go.Scatter(
                x=ts, y=dd_pct,
                mode="lines", name="Drawdown %",
                line=dict(color="#ef5350", width=1),
                fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
            ),
            row=4, col=1,
        )

    # ── Layout ──
    fig.update_layout(
        title=title,
        height=1000,
        margin=dict(l=50, r=20, t=60, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Disable rangeslider on all x-axes (Candlestick adds one by default)
    for i in range(1, 5):
        axis_key = f"xaxis{i}" if i > 1 else "xaxis"
        fig.update_layout(**{f"{axis_key}_rangeslider_visible": False})

    # Set tick labels to show time (HH:MM) on the bottom x-axis — every 30 min
    n_slots = len(full_time_labels)
    tickvals = list(range(0, n_slots, 30))
    ticktext = [full_time_labels[i] for i in tickvals]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, range=[0, n_slots - 1], row=4, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)

    if option_ohlcv is not None and not option_ohlcv.empty:
        opt_min = option_ohlcv["low"].min()
        opt_max = option_ohlcv["high"].max()
    else:
        opt_vals = df["position_mark_price"].dropna()
        opt_vals = opt_vals[opt_vals > 0]
        opt_min = opt_vals.min() if not opt_vals.empty else 0
        opt_max = opt_vals.max() if not opt_vals.empty else 0.01
    opt_margin = (opt_max - opt_min) * 0.05
    opt_y_title = "Option $"
    if option_ohlcv is not None and option_ohlcv.attrs.get("contract_info"):
        opt_y_title = f"Option $<br>{option_ohlcv.attrs['contract_info']}"
    fig.update_yaxes(title_text=opt_y_title, range=[opt_min - opt_margin, opt_max + opt_margin], row=2, col=1)
    fig.update_yaxes(title_text="Prob", range=[0, 1], row=3, col=1)

    eq_min, eq_max = df["equity"].min(), df["equity"].max()
    margin = max(abs(eq_max - eq_min) * 0.05, 1)
    fig.update_yaxes(title_text="Equity", range=[eq_min - margin, eq_max + margin], row=4, col=1)

    return fig


def make_label_distribution(label_counts: dict) -> go.Figure:
    """Pie + bar chart for label distribution."""
    label_map = {0: "Neither", 1: "Peak", 2: "Trough"}
    colors_map = {0: "#78909c", 1: "#ef5350", 2: "#26a69a"}

    labels = [label_map.get(k, str(k)) for k in sorted(label_counts.keys())]
    values = [label_counts[k] for k in sorted(label_counts.keys())]
    colors = [colors_map.get(k, "#999") for k in sorted(label_counts.keys())]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["Distribution", "Count"],
    )

    fig.add_trace(
        go.Pie(labels=labels, values=values, marker_colors=colors, textinfo="label+percent"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=values, marker_color=colors),
        row=1, col=2,
    )

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig


def make_feature_histogram(series: pd.Series, name: str, nbins: int = 50) -> go.Figure:
    """Histogram for a single feature."""
    fig = go.Figure(
        go.Histogram(x=series.dropna(), nbinsx=nbins, marker_color="#42a5f5", opacity=0.8)
    )
    fig.update_layout(
        title=f"Distribution: {name}",
        xaxis_title=name, yaxis_title="Count",
        height=350, margin=dict(l=40, r=20, t=40, b=20),
    )
    return fig


def make_feature_boxplot(df: pd.DataFrame, features: list[str]) -> go.Figure:
    """Box plots for multiple features."""
    fig = go.Figure()
    for feat in features:
        if feat in df.columns:
            fig.add_trace(go.Box(y=df[feat].dropna(), name=feat))
    fig.update_layout(
        title="Feature Box Plots",
        height=400, margin=dict(l=40, r=20, t=40, b=20),
    )
    return fig


def make_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> go.Figure:
    """Correlation heatmap for selected features."""
    valid = [f for f in features if f in df.columns]
    if not valid:
        return go.Figure()
    corr = df[valid].corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=9,
        )
    )
    fig.update_layout(
        title="Feature Correlation",
        height=max(400, 30 * len(valid)),
        margin=dict(l=100, r=20, t=40, b=100),
    )
    return fig


def make_violin_by_label(df: pd.DataFrame, feature: str) -> go.Figure:
    """Violin plot of a feature split by label."""
    label_map = {0: "Neither", 1: "Peak", 2: "Trough"}
    colors = {0: "#78909c", 1: "#ef5350", 2: "#26a69a"}

    fig = go.Figure()
    for label_val in sorted(df["label"].unique()):
        subset = df[df["label"] == label_val][feature].dropna()
        fig.add_trace(go.Violin(
            y=subset,
            name=label_map.get(label_val, str(label_val)),
            marker_color=colors.get(label_val, "#999"),
            box_visible=True,
            meanline_visible=True,
        ))

    fig.update_layout(
        title=f"{feature} by Label",
        height=400, margin=dict(l=40, r=20, t=40, b=20),
    )
    return fig


def make_pr_curve(pr_data: dict, label_name: str) -> go.Figure:
    """Precision-Recall curve from pr_data dict."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pr_data["recall"], y=pr_data["precision"],
        mode="lines", name=f"PR Curve",
        line=dict(color="#42a5f5", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 0],
        mode="lines", name="Baseline",
        line=dict(color="#bdbdbd", dash="dash"),
    ))
    fig.update_layout(
        title=f"Precision-Recall Curve ({label_name})",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
        height=400, margin=dict(l=40, r=20, t=40, b=20),
    )
    return fig


def make_time_error_chart(time_error: dict) -> go.Figure:
    """Bar chart for time error distribution (within_1..within_5)."""
    dist = time_error.get("error_distribution", {})
    if not dist:
        return go.Figure()

    labels = ["Within 1 bar", "Within 2 bars", "Within 3 bars", "Within 5 bars"]
    values = [dist.get("within_1", 0), dist.get("within_2", 0), dist.get("within_3", 0), dist.get("within_5", 0)]

    fig = go.Figure(go.Bar(
        x=labels, y=[v * 100 for v in values],
        marker_color=["#66bb6a", "#42a5f5", "#ab47bc", "#ffa726"],
        text=[f"{v:.1f}%" for v in [v * 100 for v in values]],
        textposition="auto",
    ))
    fig.update_layout(
        title="Time Error Distribution",
        yaxis_title="Percentage (%)",
        height=350, margin=dict(l=40, r=20, t=40, b=20),
    )
    return fig


def make_feature_importance_bar(imp_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for feature importance."""
    if imp_df.empty:
        return go.Figure()

    imp_df = imp_df.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
        marker_color="#42a5f5",
    ))
    fig.update_layout(
        title="Feature Importance (Gain)",
        xaxis_title="Importance",
        height=max(350, 25 * len(imp_df)),
        margin=dict(l=180, r=20, t=40, b=20),
    )
    return fig
