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
