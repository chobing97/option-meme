"""LSTM model training for sequential peak/trough detection."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.metrics import average_precision_score

from config.settings import (
    LOOKBACK_WINDOW,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    RANDOM_SEED,
)
from src.model.dataset import SplitResult, TimeSeriesDataset


class PeakTroughLSTM(nn.Module):
    """Bidirectional LSTM for peak/trough binary classification."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, 1] logits
        """
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, hidden*2]

        return self.classifier(context).squeeze(-1)  # [batch]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


def train_lstm(
    split: SplitResult,
    target_label: int,
    feature_cols: Optional[list[str]] = None,
    lookback: int = LOOKBACK_WINDOW,
    hidden_size: int = LSTM_HIDDEN_SIZE,
    num_layers: int = LSTM_NUM_LAYERS,
    dropout: float = LSTM_DROPOUT,
    lr: float = LSTM_LR,
    batch_size: int = LSTM_BATCH_SIZE,
    epochs: int = LSTM_EPOCHS,
    use_focal_loss: bool = True,
    device: Optional[str] = None,
) -> tuple[PeakTroughLSTM, dict]:
    """Train LSTM model for peak or trough detection.

    Args:
        split: Time-based data split.
        target_label: 1 for peak, 2 for trough.
        feature_cols: Feature columns.
        lookback: Sequence length.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        lr: Learning rate.
        batch_size: Batch size.
        epochs: Max training epochs.
        use_focal_loss: Use Focal Loss instead of BCE.
        device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    logger.info(f"Training LSTM on {device} for label={target_label}")

    # Build datasets
    train_ds = TimeSeriesDataset(split.train, target_label, lookback, feature_cols)
    val_ds = TimeSeriesDataset(split.val, target_label, lookback, feature_cols)
    test_ds = TimeSeriesDataset(split.test, target_label, lookback, feature_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        f"Datasets: train={len(train_ds)} (pos={train_ds.positive_rate:.4f}), "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )

    # Build model
    model = PeakTroughLSTM(
        n_features=train_ds.n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        pos_weight = torch.tensor([(1 - train_ds.positive_rate) / max(train_ds.positive_rate, 1e-6)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Training loop
    best_val_prauc = 0
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        val_prauc = _evaluate(model, val_loader, device)
        scheduler.step(val_prauc)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
                f"val_PR-AUC={val_prauc:.4f}"
            )

        # Early stopping
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    test_prauc = _evaluate(model, test_loader, device)

    metrics = {
        "target_label": target_label,
        "pr_auc_val": best_val_prauc,
        "pr_auc_test": test_prauc,
        "best_epoch": epochs - patience_counter,
        "n_features": train_ds.n_features,
        "lookback": lookback,
        "device": device,
    }

    logger.info(f"LSTM test PR-AUC: {test_prauc:.4f}")

    return model, metrics


def _evaluate(
    model: PeakTroughLSTM,
    loader: DataLoader,
    device: str,
) -> float:
    """Evaluate model and return PR-AUC."""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())

    if len(all_targets) == 0 or sum(all_targets) == 0:
        return 0.0

    return average_precision_score(all_targets, all_probs)


def predict(
    model: PeakTroughLSTM,
    dataset: TimeSeriesDataset,
    device: str = "cpu",
    batch_size: int = LSTM_BATCH_SIZE,
) -> np.ndarray:
    """Get prediction probabilities from trained model."""
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def save_model(model: PeakTroughLSTM, path: Path) -> None:
    """Save PyTorch model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"LSTM model saved to {path}")


def load_model(path: Path, n_features: int, **kwargs) -> PeakTroughLSTM:
    """Load PyTorch model."""
    model = PeakTroughLSTM(n_features=n_features, **kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    logger.info(f"LSTM model loaded from {path}")
    return model
