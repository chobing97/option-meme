"""Variant registry: Labeling x Model configuration combinations.

1분봉: L1/L2/L3 x M1~M4 = 12개 조합
5분봉: L1/L2/L3 x M1~M4 = 12개 조합
총 24개 조합.

Labeling (1m):
    L1: prominence=0.003, width=3, distance=5, shift=1 (strict, 확인 후 진입)
    L2: prominence=0.002, width=1, distance=5, shift=1 (sensitive, 확인 후 진입)
    L3: prominence=0.002, width=1, distance=5, shift=0 (sensitive, 즉시 진입)

Labeling (5m):
    L1: prominence=0.004, width=1, distance=3, shift=1 (strict, 확인 후 진입)
    L2: prominence=0.0025, width=1, distance=2, shift=1 (sensitive, 확인 후 진입)
    L3: prominence=0.0025, width=1, distance=2, shift=0 (sensitive, 즉시 진입)

Model (1m):
    M1: GBM lookback=10, LSTM lookback=10, drop early bars
    M2: GBM lookback=10, LSTM lookback=10, 0-fill early bars
    M3: GBM lookback=5,  LSTM lookback=5,  0-fill early bars
    M4: GBM lookback=0 (base features only), LSTM lookback=10 with padding

Model (5m):
    M1: GBM lookback=6, LSTM lookback=6, drop early bars
    M2: GBM lookback=6, LSTM lookback=6, 0-fill early bars
    M3: GBM lookback=3, LSTM lookback=3, 0-fill early bars
    M4: GBM lookback=0 (base features only), LSTM lookback=6 with padding
"""

# ── 1분봉 Label Configs ──────────────────────────────
LABEL_CONFIGS = {
    "L1": {"prominence_pct": 0.003, "width": 3, "distance": 5, "shift": 1},
    "L2": {"prominence_pct": 0.002, "width": 1, "distance": 5, "shift": 1},
    "L3": {"prominence_pct": 0.002, "width": 1, "distance": 5, "shift": 0},
}

# ── 5분봉 Label Configs ──────────────────────────────
LABEL_CONFIGS_5M = {
    "L1": {"prominence_pct": 0.004, "width": 1, "distance": 3, "shift": 1},
    "L2": {"prominence_pct": 0.0025, "width": 1, "distance": 2, "shift": 1},
    "L3": {"prominence_pct": 0.0025, "width": 1, "distance": 2, "shift": 0},
}

# ── 1분봉 Model Configs ──────────────────────────────
MODEL_CONFIGS = {
    "M1": {"gbm_lookback": 10, "lstm_lookback": 10, "fill_method": "drop"},
    "M2": {"gbm_lookback": 10, "lstm_lookback": 10, "fill_method": "0fill"},
    "M3": {"gbm_lookback": 5,  "lstm_lookback": 5,  "fill_method": "0fill"},
    "M4": {"gbm_lookback": 0,  "lstm_lookback": 10, "fill_method": "0fill"},
}

# ── 5분봉 Model Configs ──────────────────────────────
MODEL_CONFIGS_5M = {
    "M1": {"gbm_lookback": 6, "lstm_lookback": 6, "fill_method": "drop"},
    "M2": {"gbm_lookback": 6, "lstm_lookback": 6, "fill_method": "0fill"},
    "M3": {"gbm_lookback": 3, "lstm_lookback": 3, "fill_method": "0fill"},
    "M4": {"gbm_lookback": 0, "lstm_lookback": 6, "fill_method": "0fill"},
}


def get_label_configs(timeframe: str = "1m") -> dict:
    """Timeframe에 맞는 Label Config 반환."""
    from config.settings import SUPPORTED_TIMEFRAMES

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"지원하지 않는 timeframe: {timeframe!r}. 가능한 값: {SUPPORTED_TIMEFRAMES}")
    return LABEL_CONFIGS_5M if timeframe == "5m" else LABEL_CONFIGS


def get_model_configs(timeframe: str = "1m") -> dict:
    """Timeframe에 맞는 Model Config 반환."""
    from config.settings import SUPPORTED_TIMEFRAMES

    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"지원하지 않는 timeframe: {timeframe!r}. 가능한 값: {SUPPORTED_TIMEFRAMES}")
    return MODEL_CONFIGS_5M if timeframe == "5m" else MODEL_CONFIGS
