"""Plotly chart builder functions for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_candlestick(df: pd.DataFrame, title: str = "OHLCV") -> go.Figure:
    """Create candlestick chart with volume subplot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    colors = ["#ef5350" if c < o else "#26a69a" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df["datetime"], y=df["volume"], marker_color=colors, name="Volume", opacity=0.6),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
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
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
    )

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

    # Row 2: Volume
    colors = ["#ef5350" if c < o else "#26a69a" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df["datetime"], y=df["volume"], marker_color=colors, name="Volume", opacity=0.6),
        row=2, col=1,
    )

    # Row 3: Probabilities
    if "peak_prob" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["peak_prob"],
                mode="lines", name="Peak prob",
                line=dict(color="#ef5350", width=1.5),
            ),
            row=3, col=1,
        )
    if "trough_prob" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df["trough_prob"],
                mode="lines", name="Trough prob",
                line=dict(color="#26a69a", width=1.5),
            ),
            row=3, col=1,
        )
    # Threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="#bdbdbd", row=3, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=True,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Prob", row=3, col=1)
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
