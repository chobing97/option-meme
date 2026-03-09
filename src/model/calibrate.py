"""LSTM probability calibration using Isotonic Regression.

캘리브레이션 흐름:
  1. LSTM val predictions → fit_calibrator() → IsotonicRegression
  2. save_calibrator() → models_dir/lstm_{market}_{label}_calibrator.joblib
  3. 추론 시: load_calibrator() → apply_calibration(probs)
"""

from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression on LSTM validation predictions.

    Args:
        probs: Raw LSTM probabilities from validation set. Shape: (N,)
        labels: True binary labels (0/1). Shape: (N,)

    Returns:
        Fitted IsotonicRegression calibrator.
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, labels.astype(float))
    pos_rate = float(labels.mean()) if len(labels) > 0 else 0.0
    logger.info(
        f"Calibrator fitted: n={len(probs)}, pos_rate={pos_rate:.4f}, "
        f"pred_range=[{probs.min():.4f}, {probs.max():.4f}]"
    )
    return cal


def apply_calibration(cal: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    """Apply isotonic calibration to raw probabilities.

    Args:
        cal: Fitted IsotonicRegression calibrator.
        probs: Raw probabilities to calibrate.

    Returns:
        Calibrated probabilities, same shape as input.
    """
    return cal.predict(probs).astype(np.float32)


def save_calibrator(cal: IsotonicRegression, path: Path) -> None:
    """Persist calibrator to disk via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, path)
    logger.info(f"Calibrator saved to {path}")


def load_calibrator(path: Path) -> IsotonicRegression:
    """Load calibrator from disk."""
    cal = joblib.load(path)
    logger.info(f"Calibrator loaded from {path}")
    return cal
