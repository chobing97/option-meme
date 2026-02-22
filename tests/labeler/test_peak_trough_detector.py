"""Tests for src.labeler.peak_trough_detector."""

import numpy as np
import pandas as pd
import pytest

from config.settings import LABEL_NEITHER, LABEL_PEAK, LABEL_TROUGH
from src.labeler.peak_trough_detector import (
    DetectionResult,
    detect_peaks_troughs,
    grid_search_params,
    label_day,
)


# ── DetectionResult ────────────────────────────────────


class TestDetectionResult:
    def test_n_peaks_n_troughs_properties(self):
        result = DetectionResult(
            date="2025-01-02",
            peak_indices=np.array([10, 30]),
            trough_indices=np.array([20]),
            peak_prominences=np.array([100.0, 80.0]),
            trough_prominences=np.array([90.0]),
            labels=np.zeros(60, dtype=int),
            prices=np.ones(60),
            n_bars=60,
        )

        assert result.n_peaks == 2
        assert result.n_troughs == 1


# ── detect_peaks_troughs ──────────────────────────────


class TestDetectPeaksTroughs:
    def test_sine_wave_detects_peaks_and_troughs(self):
        n = 60
        base = 50000.0
        t = np.linspace(0, 4 * np.pi, n)
        prices = base + base * 0.005 * np.sin(t)

        peak_idx, trough_idx, _, _ = detect_peaks_troughs(
            prices, open_price=base, prominence_pct=0.003, distance=5, width=3,
        )

        assert len(peak_idx) > 0
        assert len(trough_idx) > 0

    def test_flat_signal_no_detections(self):
        prices = np.full(60, 50000.0)

        peak_idx, trough_idx, _, _ = detect_peaks_troughs(
            prices, open_price=50000.0, prominence_pct=0.003, distance=5, width=3,
        )

        assert len(peak_idx) == 0
        assert len(trough_idx) == 0


# ── label_day ─────────────────────────────────────────


class TestLabelDay:
    def test_basic_labels(self, day_df_with_peaks):
        result = label_day(day_df_with_peaks)

        assert LABEL_PEAK in result.labels
        assert LABEL_TROUGH in result.labels
        assert LABEL_NEITHER in result.labels

    def test_flat_signal_all_neither(self):
        n = 60
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-02 09:00", periods=n, freq="min"),
                "open": np.full(n, 50000.0),
                "close": np.full(n, 50000.0),
                "high": np.full(n, 50000.0),
                "low": np.full(n, 50000.0),
                "volume": np.full(n, 10000),
                "date": "2025-01-02",
            }
        )

        result = label_day(df)
        assert np.all(result.labels == LABEL_NEITHER)

    def test_labels_length_matches_bars(self, day_df_with_peaks):
        result = label_day(day_df_with_peaks)

        assert len(result.labels) == len(day_df_with_peaks)
        assert result.n_bars == len(day_df_with_peaks)

    def test_labels_shifted_to_next_bar(self, day_df_with_peaks):
        """Peak/trough labels should appear at detection_index + 1, not at the detection index itself."""
        result = label_day(day_df_with_peaks)

        # Original detection indices should NOT carry labels (shifted away)
        for idx in result.peak_indices:
            if idx + 1 < result.n_bars:
                assert result.labels[idx] != LABEL_PEAK or idx in result.trough_indices + 1, (
                    f"Peak label should not be at original detection index {idx}"
                )
                assert result.labels[idx + 1] == LABEL_PEAK, (
                    f"Peak label missing at shifted index {idx + 1}"
                )

        for idx in result.trough_indices:
            if idx + 1 < result.n_bars:
                assert result.labels[idx] != LABEL_TROUGH or idx in result.peak_indices + 1, (
                    f"Trough label should not be at original detection index {idx}"
                )
                assert result.labels[idx + 1] == LABEL_TROUGH, (
                    f"Trough label missing at shifted index {idx + 1}"
                )

    def test_last_bar_peak_dropped(self):
        """A peak detected at the last bar should be excluded after shifting (out of bounds)."""
        n = 60
        base = 50000.0
        # Create a signal that peaks at the very last bar
        t = np.linspace(0, np.pi, n)  # half-sine: rises to peak at the end
        close = base + base * 0.005 * np.sin(t)
        # Force the last bar to be the clear maximum
        close[-1] = close.max() + base * 0.005

        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-02 09:00", periods=n, freq="min"),
                "open": np.full(n, base),
                "high": close + 10,
                "low": close - 10,
                "close": close,
                "volume": np.full(n, 10000),
                "date": "2025-01-02",
            }
        )
        result = label_day(df)

        # If the last bar (index n-1) is detected as a peak, shifted label would be at n → out of bounds
        if n - 1 in result.peak_indices:
            assert result.labels[n - 1] != LABEL_PEAK, (
                "Last-bar peak should be dropped after shift (no room for +1)"
            )


# ── grid_search_params ────────────────────────────────


class TestGridSearchParams:
    def test_returns_results(self, day_df_with_peaks):
        results = grid_search_params(day_df_with_peaks)

        assert len(results) > 0
        required_keys = {"prominence_pct", "distance", "width", "n_peaks", "n_troughs", "total_labels", "label_ratio"}
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_custom_ranges(self, day_df_with_peaks):
        prom = [0.002, 0.003]
        dist = [5, 7]
        wid = [2, 3]

        results = grid_search_params(
            day_df_with_peaks,
            prominence_range=prom,
            distance_range=dist,
            width_range=wid,
        )

        assert len(results) == len(prom) * len(dist) * len(wid)
