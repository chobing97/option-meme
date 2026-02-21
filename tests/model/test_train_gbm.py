"""Tests for src.model.train_gbm."""

import numpy as np
import pytest
import lightgbm as lgb
import optuna

from src.model.dataset import prepare_xy
from src.model.train_gbm import load_model, optimize_lgb, save_model, train_lgb
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
