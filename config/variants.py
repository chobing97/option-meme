"""Variant registry: Labeling x Model configuration combinations.

8 combinations: L1/L2 x M1~M4.

Labeling:
    L1: prominence=0.003, width=3, distance=5
    L2: prominence=0.002, width=1, distance=5

Model:
    M1: GBM lookback=10, LSTM lookback=10, drop early bars
    M2: GBM lookback=10, LSTM lookback=10, 0-fill early bars
    M3: GBM lookback=5,  LSTM lookback=5,  0-fill early bars
    M4: GBM lookback=0 (base features only), LSTM lookback=10 with padding
"""

LABEL_CONFIGS = {
    "L1": {"prominence_pct": 0.003, "width": 3, "distance": 5},
    "L2": {"prominence_pct": 0.002, "width": 1, "distance": 5},
}

MODEL_CONFIGS = {
    "M1": {"gbm_lookback": 10, "lstm_lookback": 10, "fill_method": "drop"},
    "M2": {"gbm_lookback": 10, "lstm_lookback": 10, "fill_method": "0fill"},
    "M3": {"gbm_lookback": 5,  "lstm_lookback": 5,  "fill_method": "0fill"},
    "M4": {"gbm_lookback": 0,  "lstm_lookback": 10, "fill_method": "0fill"},
}
