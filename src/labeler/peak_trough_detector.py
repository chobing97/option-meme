"""Peak and trough detection using scipy.signal.find_peaks.

Peak: local maximum (price starts declining after) → potential sell signal
Trough: local minimum (price starts rising after) → potential buy signal
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from loguru import logger

from config.settings import (
    LABEL_NEITHER,
    LABEL_PEAK,
    LABEL_TROUGH,
    PEAK_DISTANCE,
    PEAK_PROMINENCE_PCT,
    PEAK_WIDTH,
)


@dataclass
class DetectionResult:
    """Result of peak/trough detection for a single day."""

    date: str
    peak_indices: np.ndarray
    trough_indices: np.ndarray
    peak_prominences: np.ndarray
    trough_prominences: np.ndarray
    labels: np.ndarray  # 0=neither, 1=peak, 2=trough
    prices: np.ndarray
    n_bars: int

    @property
    def n_peaks(self) -> int:
        return len(self.peak_indices)

    @property
    def n_troughs(self) -> int:
        return len(self.trough_indices)


def detect_peaks_troughs(
    prices: np.ndarray,
    open_price: float,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_DISTANCE,
    width: int = PEAK_WIDTH,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Detect peaks and troughs in a price series.

    Args:
        prices: Array of close prices.
        open_price: Opening price (for prominence scaling).
        prominence_pct: Minimum prominence as fraction of open price.
        distance: Minimum distance between peaks/troughs (in bars).
        width: Minimum width of peak/trough.

    Returns:
        Tuple of (peak_indices, trough_indices, peak_properties, trough_properties).
    """
    prominence = open_price * prominence_pct

    # Detect peaks (local maxima → start of decline)
    peak_idx, peak_props = find_peaks(
        prices,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    # Detect troughs (local minima → start of rally) by inverting prices
    trough_idx, trough_props = find_peaks(
        -prices,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    return peak_idx, trough_idx, peak_props, trough_props


def label_day(
    day_df: pd.DataFrame,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_DISTANCE,
    width: int = PEAK_WIDTH,
) -> DetectionResult:
    """Label a single day's early session bars with peak/trough labels.

    Args:
        day_df: DataFrame for one day with 'close' and 'open' columns.
        prominence_pct: Minimum prominence as fraction of open price.
        distance: Minimum bars between peaks/troughs.
        width: Minimum width of peak/trough.

    Returns:
        DetectionResult with labels and detection metadata.
    """
    prices = day_df["close"].values.astype(float)
    open_price = day_df["open"].iloc[0]
    date_str = str(day_df["date"].iloc[0]) if "date" in day_df.columns else "unknown"
    n_bars = len(prices)

    peak_idx, trough_idx, peak_props, trough_props = detect_peaks_troughs(
        prices, open_price, prominence_pct, distance, width,
    )

    # Build labels array
    labels = np.full(n_bars, LABEL_NEITHER, dtype=int)
    labels[peak_idx] = LABEL_PEAK
    labels[trough_idx] = LABEL_TROUGH

    # Handle conflicts (same bar labeled as both peak and trough)
    conflict_mask = np.isin(peak_idx, trough_idx)
    if conflict_mask.any():
        conflict_bars = peak_idx[conflict_mask]
        logger.warning(f"Peak/trough conflict at bars {conflict_bars} on {date_str}")
        # Resolve by keeping the one with higher prominence
        for bar in conflict_bars:
            p_idx = np.where(peak_idx == bar)[0][0]
            t_idx = np.where(trough_idx == bar)[0][0]
            p_prom = peak_props["prominences"][p_idx]
            t_prom = trough_props["prominences"][t_idx]
            labels[bar] = LABEL_PEAK if p_prom >= t_prom else LABEL_TROUGH

    return DetectionResult(
        date=date_str,
        peak_indices=peak_idx,
        trough_indices=trough_idx,
        peak_prominences=peak_props.get("prominences", np.array([])),
        trough_prominences=trough_props.get("prominences", np.array([])),
        labels=labels,
        prices=prices,
        n_bars=n_bars,
    )


def grid_search_params(
    day_df: pd.DataFrame,
    prominence_range: Optional[list[float]] = None,
    distance_range: Optional[list[int]] = None,
    width_range: Optional[list[int]] = None,
) -> list[dict]:
    """Search over parameter grid to find optimal detection settings.

    Args:
        day_df: Single day DataFrame.
        prominence_range: List of prominence_pct values to try.
        distance_range: List of distance values to try.
        width_range: List of width values to try.

    Returns:
        List of dicts with params and resulting peak/trough counts.
    """
    if prominence_range is None:
        prominence_range = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
    if distance_range is None:
        distance_range = [3, 5, 7, 10]
    if width_range is None:
        width_range = [1, 2, 3, 5]

    results = []
    for prom in prominence_range:
        for dist in distance_range:
            for w in width_range:
                try:
                    result = label_day(day_df, prominence_pct=prom, distance=dist, width=w)
                    results.append({
                        "prominence_pct": prom,
                        "distance": dist,
                        "width": w,
                        "n_peaks": result.n_peaks,
                        "n_troughs": result.n_troughs,
                        "total_labels": result.n_peaks + result.n_troughs,
                        "label_ratio": (result.n_peaks + result.n_troughs) / result.n_bars,
                    })
                except Exception as e:
                    logger.warning(f"Grid search error (prom={prom}, dist={dist}, w={w}): {e}")

    return results
