"""Tests for src.model.train_lstm."""

import numpy as np
import torch

from src.model.dataset import TimeSeriesDataset
from src.model.train_lstm import (
    FocalLoss,
    PeakTroughLSTM,
    load_model,
    predict,
    save_model,
    train_lstm,
)
from tests.model.conftest import FEATURE_COLS

# Shared minimal training kwargs
_TRAIN_KWARGS = dict(
    target_label=1,
    feature_cols=FEATURE_COLS,
    lookback=3,
    hidden_size=16,
    num_layers=1,
    batch_size=32,
    epochs=2,
    device="cpu",
)


class TestPeakTroughLSTM:
    def test_forward_output_shape(self):
        model = PeakTroughLSTM(n_features=5, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(4, 3, 5)
        output = model(x)
        assert output.shape == (4,)

    def test_single_sample(self):
        model = PeakTroughLSTM(n_features=5, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(1, 3, 5)
        output = model(x)
        assert output.shape == (1,)

    def test_gradient_flows(self):
        model = PeakTroughLSTM(n_features=5, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(4, 3, 5)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestFocalLoss:
    def test_output_is_scalar_positive(self):
        loss_fn = FocalLoss()
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = FocalLoss()
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        perfect_logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
        wrong_logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])
        perfect_loss = loss_fn(perfect_logits, targets)
        wrong_loss = loss_fn(wrong_logits, targets)
        assert perfect_loss < wrong_loss


class TestTrainLstm:
    def test_returns_model_and_metrics(self, split_result):
        model, metrics = train_lstm(split_result, **_TRAIN_KWARGS)
        assert isinstance(model, PeakTroughLSTM)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, split_result):
        _, metrics = train_lstm(split_result, **_TRAIN_KWARGS)
        for key in ["pr_auc_val", "pr_auc_test", "best_epoch", "device"]:
            assert key in metrics

    def test_predict_shape(self, split_result):
        model, _ = train_lstm(split_result, **_TRAIN_KWARGS)
        ds = TimeSeriesDataset(
            split_result.test, target_label=1, lookback=3, feature_cols=FEATURE_COLS,
        )
        probs = predict(model, ds, device="cpu", batch_size=32)
        assert len(probs) == len(ds)


class TestLstmSaveLoad:
    def test_roundtrip_predictions_match(self, split_result, tmp_path):
        model, _ = train_lstm(split_result, **_TRAIN_KWARGS)
        ds = TimeSeriesDataset(
            split_result.test, target_label=1, lookback=3, feature_cols=FEATURE_COLS,
        )
        pred_orig = predict(model, ds, device="cpu")

        path = tmp_path / "lstm.pt"
        save_model(model, path)
        loaded = load_model(path, n_features=5, hidden_size=16, num_layers=1)
        pred_loaded = predict(loaded, ds, device="cpu")
        np.testing.assert_allclose(pred_orig, pred_loaded)

    def test_creates_parent_dirs(self, split_result, tmp_path):
        model, _ = train_lstm(split_result, **_TRAIN_KWARGS)
        path = tmp_path / "sub" / "dir" / "lstm.pt"
        save_model(model, path)
        assert path.exists()

    def test_load_requires_n_features(self, split_result, tmp_path):
        model, _ = train_lstm(split_result, **_TRAIN_KWARGS)
        path = tmp_path / "lstm.pt"
        save_model(model, path)
        loaded = load_model(path, n_features=5, hidden_size=16, num_layers=1)
        assert loaded.lstm.input_size == 5
