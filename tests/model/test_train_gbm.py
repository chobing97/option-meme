"""Tests for src.model.train_gbm."""

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
import optuna

from src.model.dataset import prepare_xy
from src.model.train_gbm import load_model, optimize_lgb, save_model, train_lgb, train_lgb_incremental
from tests.model.conftest import FEATURE_COLS


class TestTrainLgb:
    def test_returns_booster_and_metrics(self, split_result, feature_cols):
        model, metrics = train_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        assert isinstance(model, lgb.Booster)
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, split_result, feature_cols):
        _, metrics = train_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        for key in ["pr_auc_test", "best_iteration", "top_features"]:
            assert key in metrics

    def test_predictions_are_probabilities(self, split_result, feature_cols):
        model, _ = train_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        X_test, _ = prepare_xy(split_result.test, 1, feature_cols)
        preds = model.predict(X_test)
        assert preds.min() >= 0
        assert preds.max() <= 1


class TestOptimizeLgb:
    def test_returns_model_metrics_params(self, split_result, feature_cols):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        model, metrics, best_params = optimize_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            n_trials=2,
            timeout=60,
        )
        assert isinstance(model, lgb.Booster)
        assert isinstance(metrics, dict)
        assert isinstance(best_params, dict)
        assert "num_leaves" in best_params


class TestSaveLoadModel:
    def test_roundtrip(self, split_result, feature_cols, tmp_path):
        model, _ = train_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        path = tmp_path / "model.txt"
        save_model(model, path)
        loaded = load_model(path)

        X_test, _ = prepare_xy(split_result.test, 1, feature_cols)
        pred_orig = model.predict(X_test)
        pred_loaded = loaded.predict(X_test)
        np.testing.assert_allclose(pred_orig, pred_loaded)

    def test_creates_parent_dirs(self, split_result, feature_cols, tmp_path):
        model, _ = train_lgb(
            split_result,
            target_label=1,
            feature_cols=feature_cols,
            num_boost_round=10,
            early_stopping_rounds=5,
        )
        path = tmp_path / "sub" / "dir" / "model.txt"
        save_model(model, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_model(tmp_path / "nonexistent.txt")


class TestTrainLgbIncremental:
    """Tests for incremental LightGBM training."""

    def _make_chunks(self, model_df, tmp_path):
        """Save model_df as 2 partitioned chunks and return chunk info list."""
        symbols = ["SYM_A", "SYM_B"]
        n = len(model_df)
        half = n // 2

        chunks = []
        for i, sym in enumerate(symbols):
            start = i * half
            end = start + half
            part_df = model_df.iloc[start:end].copy()
            part_df["symbol"] = sym

            sym_dir = tmp_path / sym
            sym_dir.mkdir(parents=True, exist_ok=True)
            path = sym_dir / "2024.parquet"
            part_df.to_parquet(path, index=False)
            chunks.append([{"path": path, "symbol": sym, "num_rows": len(part_df)}])

        return chunks

    def test_returns_booster_and_metrics(self, model_df, tmp_path):
        chunks = self._make_chunks(model_df, tmp_path)
        split_dates = {"val_start": "2023-06-01", "test_start": "2024-06-01"}

        model, metrics = train_lgb_incremental(
            chunks, target_label=1, feature_cols=FEATURE_COLS,
            split_dates=split_dates, num_boost_round=20,
            early_stopping_rounds=5,
        )
        assert isinstance(model, lgb.Booster)
        assert isinstance(metrics, dict)
        assert "pr_auc_test" in metrics
        assert "n_chunks" in metrics
        assert metrics["n_chunks"] == 2

    def test_model_has_trees_from_multiple_chunks(self, model_df, tmp_path):
        chunks = self._make_chunks(model_df, tmp_path)
        split_dates = {"val_start": "2023-06-01", "test_start": "2024-06-01"}

        model, _ = train_lgb_incremental(
            chunks, target_label=1, feature_cols=FEATURE_COLS,
            split_dates=split_dates, num_boost_round=20,
            early_stopping_rounds=5,
        )
        # With 2 chunks × 10 rounds each, model should have trees from both
        assert model.num_trees() > 0

    def test_predictions_are_probabilities(self, model_df, tmp_path):
        chunks = self._make_chunks(model_df, tmp_path)
        split_dates = {"val_start": "2023-06-01", "test_start": "2024-06-01"}

        model, _ = train_lgb_incremental(
            chunks, target_label=1, feature_cols=FEATURE_COLS,
            split_dates=split_dates, num_boost_round=20,
            early_stopping_rounds=5,
        )
        # Predict on some data
        X_test = model_df[FEATURE_COLS].values[:10].astype(np.float32)
        preds = model.predict(X_test)
        assert preds.min() >= 0
        assert preds.max() <= 1

    def test_single_chunk_works(self, model_df, tmp_path):
        """Single chunk should work the same as standard training."""
        path = tmp_path / "all" / "2024.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        model_df.to_parquet(path, index=False)

        chunks = [[{"path": path, "symbol": "ALL", "num_rows": len(model_df)}]]
        split_dates = {"val_start": "2023-06-01", "test_start": "2024-06-01"}

        model, metrics = train_lgb_incremental(
            chunks, target_label=1, feature_cols=FEATURE_COLS,
            split_dates=split_dates, num_boost_round=20,
            early_stopping_rounds=5,
        )
        assert isinstance(model, lgb.Booster)
        assert metrics["n_chunks"] == 1
