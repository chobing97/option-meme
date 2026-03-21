"""Tests for predict_all() batch prediction and run_batch_predict() wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_MOD = "src.inference.predict"


# ── Helpers ──────────────────────────────────────────────────


def _mock_model(proba: np.ndarray) -> MagicMock:
    """Create a mock LightGBM Booster that returns fixed probabilities."""
    model = MagicMock()
    model.predict.return_value = proba
    return model


def _run_predict_all(
    featured_df: pd.DataFrame,
    peak_proba: np.ndarray,
    trough_proba: np.ndarray,
    tmp_path,
    market: str = "kr",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run predict_all with mocked dependencies."""
    from src.inference.predict import predict_all

    featured_path = tmp_path / "processed" / "featured"
    featured_path.mkdir(parents=True, exist_ok=True)
    featured_df.to_parquet(featured_path / f"{market}_featured.parquet", index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"lgb_{market}_peak.txt").touch()
    (models_dir / f"lgb_{market}_trough.txt").touch()

    pred_dir = tmp_path / "predictions" / "labeled"

    peak_model = _mock_model(peak_proba)
    trough_model = _mock_model(trough_proba)

    with (
        patch(f"{_MOD}.PROCESSED_DIR", tmp_path / "processed"),
        patch(f"{_MOD}.DATA_DIR", tmp_path),
        patch(f"{_MOD}.PREDICTIONS_DIR", pred_dir),
        patch("src.model.train_gbm.load_model", side_effect=[peak_model, trough_model]),
    ):
        return predict_all(
            market=market,
            model_type="gbm",
            threshold=threshold,
        )


# ── Happy Path ───────────────────────────────────────────────


class TestPredictAllHappyPath:
    def test_returns_dataframe(self, featured_df, tmp_path):
        n = len(featured_df)
        result = _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.3),
            trough_proba=np.full(n, 0.3),
            tmp_path=tmp_path,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == n

    def test_output_columns_labeled_format(self, featured_df, tmp_path):
        n = len(featured_df)
        result = _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.1),
            trough_proba=np.full(n, 0.1),
            tmp_path=tmp_path,
        )
        expected = {
            "datetime", "open", "high", "low", "close", "volume",
            "date", "minutes_from_open", "label", "symbol", "market",
            "peak_prob", "trough_prob",
        }
        assert expected.issubset(set(result.columns))
        # Feature columns should NOT be in output
        for col in result.columns:
            assert not col.startswith(("pf_", "vf_", "tf_", "tmf_"))

    def test_proba_columns_always_present(self, featured_df, tmp_path):
        n = len(featured_df)
        result = _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.5),
            trough_proba=np.full(n, 0.5),
            tmp_path=tmp_path,
        )
        assert "peak_prob" in result.columns
        assert "trough_prob" in result.columns

    def test_proba_values_match_model_output(self, featured_df, tmp_path):
        n = len(featured_df)
        peak_p = np.linspace(0.0, 1.0, n, dtype=np.float32)
        trough_p = np.linspace(1.0, 0.0, n, dtype=np.float32)
        result = _run_predict_all(
            featured_df,
            peak_proba=peak_p,
            trough_proba=trough_p,
            tmp_path=tmp_path,
        )
        np.testing.assert_array_almost_equal(
            result["peak_prob"].values, np.round(peak_p, 4), decimal=4
        )
        np.testing.assert_array_almost_equal(
            result["trough_prob"].values, np.round(trough_p, 4), decimal=4
        )

    def test_parquet_saved(self, featured_df, tmp_path):
        n = len(featured_df)
        _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.1),
            trough_proba=np.full(n, 0.1),
            tmp_path=tmp_path,
        )
        saved_path = tmp_path / "predictions" / "labeled" / "kr" / "kr_predicted.parquet"
        assert saved_path.exists()

    def test_parquet_roundtrip(self, featured_df, tmp_path):
        n = len(featured_df)
        result = _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.6),
            trough_proba=np.full(n, 0.3),
            tmp_path=tmp_path,
        )
        saved_path = tmp_path / "predictions" / "labeled" / "kr" / "kr_predicted.parquet"
        loaded = pd.read_parquet(saved_path)
        pd.testing.assert_frame_equal(result, loaded)


# ── Label Assignment ─────────────────────────────────────────


class TestLabelAssignment:
    def _single_label(self, peak_p, trough_p, threshold, featured_df, tmp_path):
        """Helper: predict on single-row df, return the label."""
        row_df = featured_df.iloc[:1].copy()
        result = _run_predict_all(
            row_df,
            peak_proba=np.array([peak_p]),
            trough_proba=np.array([trough_p]),
            tmp_path=tmp_path,
            threshold=threshold,
        )
        return int(result["label"].iloc[0])

    def test_peak_detected(self, featured_df, tmp_path):
        label = self._single_label(0.8, 0.3, 0.5, featured_df, tmp_path)
        assert label == 1

    def test_trough_detected(self, featured_df, tmp_path):
        label = self._single_label(0.3, 0.8, 0.5, featured_df, tmp_path)
        assert label == 2

    def test_neither_below_threshold(self, featured_df, tmp_path):
        label = self._single_label(0.3, 0.4, 0.5, featured_df, tmp_path)
        assert label == 0

    def test_equal_proba_above_threshold(self, featured_df, tmp_path):
        # Both >= threshold but neither strictly > the other → label=0
        label = self._single_label(0.7, 0.7, 0.5, featured_df, tmp_path)
        assert label == 0

    def test_threshold_boundary_peak(self, featured_df, tmp_path):
        # Exactly at threshold with peak > trough → label=1
        label = self._single_label(0.5, 0.3, 0.5, featured_df, tmp_path)
        assert label == 1

    def test_threshold_boundary_trough(self, featured_df, tmp_path):
        # Exactly at threshold with trough > peak → label=2
        label = self._single_label(0.3, 0.5, 0.5, featured_df, tmp_path)
        assert label == 2

    def test_mixed_labels_vectorized(self, featured_df, tmp_path):
        df = featured_df.iloc[:3].copy()
        result = _run_predict_all(
            df,
            peak_proba=np.array([0.9, 0.1, 0.2]),
            trough_proba=np.array([0.1, 0.9, 0.2]),
            tmp_path=tmp_path,
            threshold=0.5,
        )
        assert list(result["label"]) == [1, 2, 0]

    def test_low_threshold_more_signals(self, featured_df, tmp_path):
        n = len(featured_df)
        rng = np.random.RandomState(42)
        peak_p = rng.rand(n).astype(np.float32)
        trough_p = rng.rand(n).astype(np.float32)

        result_low = _run_predict_all(
            featured_df, peak_p, trough_p, tmp_path, threshold=0.1
        )
        # Use a separate tmp_path subfolder for the second run
        high_path = tmp_path / "high"
        high_path.mkdir()
        result_high = _run_predict_all(
            featured_df, peak_p, trough_p, high_path, threshold=0.9
        )

        signals_low = (result_low["label"] != 0).sum()
        signals_high = (result_high["label"] != 0).sum()
        assert signals_low >= signals_high


# ── Error Cases ──────────────────────────────────────────────


class TestPredictAllErrors:
    def test_featured_not_found(self, tmp_path):
        from src.inference.predict import predict_all

        with (
            patch(f"{_MOD}.PROCESSED_DIR", tmp_path / "processed"),
            pytest.raises(FileNotFoundError, match="Run features first"),
        ):
            predict_all(market="kr")

    def test_model_not_found(self, featured_df, tmp_path):
        from src.inference.predict import predict_all

        # Write featured parquet but don't create model files
        featured_path = tmp_path / "processed" / "featured"
        featured_path.mkdir(parents=True, exist_ok=True)
        featured_df.to_parquet(featured_path / "kr_featured.parquet", index=False)

        with (
            patch(f"{_MOD}.PROCESSED_DIR", tmp_path / "processed"),
            patch(f"{_MOD}.DATA_DIR", tmp_path),
            pytest.raises(FileNotFoundError, match="Train first"),
        ):
            predict_all(market="kr")

    def test_no_feature_columns(self, no_feature_df, tmp_path):
        from src.inference.predict import predict_all

        featured_path = tmp_path / "processed" / "featured"
        featured_path.mkdir(parents=True, exist_ok=True)
        no_feature_df.to_parquet(featured_path / "kr_featured.parquet", index=False)

        with (
            patch(f"{_MOD}.PROCESSED_DIR", tmp_path / "processed"),
            pytest.raises(ValueError, match="No feature columns"),
        ):
            predict_all(market="kr")


# ── NaN / Inf Handling ───────────────────────────────────────


class TestNanHandling:
    def test_nan_in_features(self, featured_df_with_nans, tmp_path):
        n = len(featured_df_with_nans)
        result = _run_predict_all(
            featured_df_with_nans,
            peak_proba=np.full(n, 0.6),
            trough_proba=np.full(n, 0.3),
            tmp_path=tmp_path,
        )
        assert len(result) == n
        assert result["label"].isin([0, 1, 2]).all()

    def test_inf_handled_same_as_nan(self, featured_df_with_nans, tmp_path):
        """Verify inf/-inf don't cause errors or produce invalid labels."""
        n = len(featured_df_with_nans)
        result = _run_predict_all(
            featured_df_with_nans,
            peak_proba=np.full(n, 0.8),
            trough_proba=np.full(n, 0.1),
            tmp_path=tmp_path,
        )
        assert not result["label"].isna().any()
        assert set(result["label"].unique()).issubset({0, 1, 2})


# ── run_batch_predict Wrapper ────────────────────────────────


_PREDICT_ALL = f"{_MOD}.predict_all"


class TestRunBatchPredict:
    def test_calls_predict_all_per_market(self):
        from run_pipeline import run_batch_predict

        with patch(_PREDICT_ALL) as mock_predict:
            mock_predict.return_value = pd.DataFrame({"label": [0, 1, 2]})
            run_batch_predict(["kr", "us"], model_type="gbm", threshold=0.5)

            assert mock_predict.call_count == 2
            mock_predict.assert_any_call(
                market="kr", model_type="gbm", threshold=0.5,
                label_config=None, model_config=None, timeframe="1m",
            )
            mock_predict.assert_any_call(
                market="us", model_type="gbm", threshold=0.5,
                label_config=None, model_config=None, timeframe="1m",
            )

    def test_error_logged_not_raised(self):
        from run_pipeline import run_batch_predict

        with patch(_PREDICT_ALL, side_effect=FileNotFoundError("no data")):
            # Should not raise — errors are caught and logged
            run_batch_predict(["kr"], model_type="gbm", threshold=0.5)

    def test_threshold_passed_through(self):
        from run_pipeline import run_batch_predict

        with patch(_PREDICT_ALL) as mock_predict:
            mock_predict.return_value = pd.DataFrame({"label": [0]})
            run_batch_predict(["kr"], model_type="gbm", threshold=0.3)

            mock_predict.assert_called_once_with(
                market="kr", model_type="gbm", threshold=0.3,
                label_config=None, model_config=None, timeframe="1m",
            )

    def test_value_error_also_caught(self):
        from run_pipeline import run_batch_predict

        with patch(_PREDICT_ALL, side_effect=ValueError("bad")):
            run_batch_predict(["us"], model_type="gbm", threshold=0.5)

    def test_continues_after_first_market_fails(self):
        from run_pipeline import run_batch_predict

        with patch(_PREDICT_ALL) as mock_predict:
            mock_predict.side_effect = [
                FileNotFoundError("kr failed"),
                pd.DataFrame({"label": [0, 1]}),
            ]
            run_batch_predict(["kr", "us"], model_type="gbm", threshold=0.5)
            assert mock_predict.call_count == 2


# ── Timeframe Parameter ────────────────────────────────────────


class TestTimeframeParameter:
    """Tests for predict_all timeframe parameter."""

    def test_predict_all_accepts_timeframe(self, featured_df, tmp_path):
        """predict_all should accept timeframe parameter without error."""
        n = len(featured_df)
        result = _run_predict_all(
            featured_df,
            peak_proba=np.full(n, 0.3),
            trough_proba=np.full(n, 0.3),
            tmp_path=tmp_path,
        )
        assert isinstance(result, pd.DataFrame)

    def test_predict_all_timeframe_path_resolution(self, featured_df, tmp_path):
        """When label_config and model_config are set, predictions saved partitioned."""
        from src.inference.predict import predict_all

        # Create featured parquet at timeframe-aware partitioned path (symbol/year)
        sym = featured_df["symbol"].iloc[0]
        year = pd.to_datetime(featured_df["datetime"].iloc[0]).year
        featured_part_dir = tmp_path / "processed" / "featured" / "1m" / "L1" / "M1" / "kr" / sym
        featured_part_dir.mkdir(parents=True, exist_ok=True)
        featured_df[featured_df["symbol"] == sym].to_parquet(
            featured_part_dir / f"{year}.parquet", index=False
        )
        sym2 = featured_df["symbol"].unique()[1] if featured_df["symbol"].nunique() > 1 else sym
        if sym2 != sym:
            featured_part_dir2 = tmp_path / "processed" / "featured" / "1m" / "L1" / "M1" / "kr" / sym2
            featured_part_dir2.mkdir(parents=True, exist_ok=True)
            featured_df[featured_df["symbol"] == sym2].to_parquet(
                featured_part_dir2 / f"{year}.parquet", index=False
            )

        models_dir = tmp_path / "models" / "1m" / "L1" / "M1"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "lgb_kr_peak.txt").touch()
        (models_dir / "lgb_kr_trough.txt").touch()

        # Mock models that handle variable-length input
        def _make_dynamic_model(val=0.3):
            m = MagicMock()
            m.predict.side_effect = lambda X: np.full(len(X), val)
            return m

        peak_model = _make_dynamic_model(0.3)
        trough_model = _make_dynamic_model(0.3)

        pred_dir = tmp_path / "predictions" / "labeled"

        with (
            patch(f"{_MOD}.PROCESSED_DIR", tmp_path / "processed"),
            patch("src.features.feature_pipeline.PROCESSED_DIR", tmp_path / "processed"),
            patch(f"{_MOD}.DATA_DIR", tmp_path),
            patch(f"{_MOD}.PREDICTIONS_DIR", pred_dir),
            patch("src.model.train_gbm.load_model", side_effect=[peak_model, trough_model]),
        ):
            result = predict_all(
                market="kr",
                model_type="gbm",
                threshold=0.5,
                label_config="L1",
                model_config="M1",
                timeframe="1m",
            )
        assert isinstance(result, pd.DataFrame)
        # Partitioned: result is last symbol's partition, but predictions saved for all
        assert len(result) > 0
        # Verify partitioned files exist
        n_symbols = featured_df["symbol"].nunique()
        pred_files = list((pred_dir / "gbm" / "1m" / "L1" / "M1" / "kr").rglob("*.parquet"))
        assert len(pred_files) == n_symbols
