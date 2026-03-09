"""Weighted ensemble of GBM and calibrated LSTM for US market.

앙상블 흐름:
  1. val set에서 GBM + calibrated LSTM 예측값 수집
  2. find_optimal_weight()로 PR-AUC 최대화하는 w_gbm 탐색 (per label)
  3. ensemble_predict()로 가중 평균 확률 생성
  4. 결과를 save_weights() → models_dir/ensemble_{market}_weights.json

추론 시:
  - load_weights() → w_gbm 로드
  - ensemble_predict(gbm_proba, lstm_cal_proba, w_gbm)

Note:
  - 앙상블은 주로 US 시장을 위해 설계됨.
    KR LSTM은 데이터 부족(~4,680 bars)으로 PR-AUC 0.13~0.33 수준이므로
    KR에 적용 시 GBM 단독보다 성능이 하락할 수 있음.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score


def find_optimal_weight(
    gbm_proba: np.ndarray,
    lstm_cal_proba: np.ndarray,
    labels: np.ndarray,
    w_candidates: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Grid search for optimal GBM weight that maximizes val PR-AUC.

    Args:
        gbm_proba: GBM probabilities on validation set. Shape: (N,)
        lstm_cal_proba: Calibrated LSTM probabilities on validation set. Shape: (N,)
        labels: True binary labels (0/1). Shape: (N,)
        w_candidates: GBM weight candidates. Default: 0.50..1.00 step 0.05.

    Returns:
        Tuple of (best_w_gbm, best_val_pr_auc).
    """
    if w_candidates is None:
        w_candidates = np.arange(0.50, 1.01, 0.05)

    if labels.sum() == 0:
        logger.warning("No positive labels in validation set — defaulting to w_gbm=0.70")
        return 0.70, 0.0

    # Baseline: GBM alone
    gbm_only_auc = average_precision_score(labels, gbm_proba)
    logger.info(f"GBM-only val PR-AUC: {gbm_only_auc:.4f}")

    best_w, best_auc = 1.0, gbm_only_auc  # start from GBM-only as baseline
    for w in w_candidates:
        ens = ensemble_predict(gbm_proba, lstm_cal_proba, w_gbm=float(w))
        auc = average_precision_score(labels, ens)
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)

    improvement = best_auc - gbm_only_auc
    logger.info(
        f"Optimal w_gbm={best_w:.2f}, val PR-AUC={best_auc:.4f} "
        f"(improvement over GBM-only: {improvement:+.4f})"
    )
    return best_w, best_auc


def ensemble_predict(
    gbm_proba: np.ndarray,
    lstm_cal_proba: np.ndarray,
    w_gbm: float = 0.70,
) -> np.ndarray:
    """Weighted average ensemble: w_gbm * GBM + (1 - w_gbm) * LSTM_cal.

    Args:
        gbm_proba: GBM probabilities.
        lstm_cal_proba: Calibrated LSTM probabilities (same length as gbm_proba).
        w_gbm: Weight for GBM (0~1). Weight for LSTM is (1 - w_gbm).

    Returns:
        Ensemble probabilities, same shape as inputs.
    """
    return (w_gbm * gbm_proba + (1.0 - w_gbm) * lstm_cal_proba).astype(np.float32)


def save_weights(weights: dict, path: Path) -> None:
    """Save ensemble weights to JSON.

    Expected format:
        {
            "peak": {"w_gbm": 0.75, "val_pr_auc": 0.7123},
            "trough": {"w_gbm": 0.70, "val_pr_auc": 0.7201}
        }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    logger.info(f"Ensemble weights saved to {path}")


def load_weights(path: Path) -> dict:
    """Load ensemble weights from JSON."""
    weights = json.loads(path.read_text())
    logger.info(f"Ensemble weights loaded: {weights}")
    return weights
