"""Global settings for option-meme project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = PROCESSED_DIR / "labeled"
METADATA_DIR = DATA_DIR / "metadata"
CONFIG_DIR = PROJECT_ROOT / "config"
SYMBOLS_DIR = CONFIG_DIR / "symbols"

# ── Market Sessions ───────────────────────────────────
# Early session window (minutes from market open)
EARLY_SESSION_MINUTES = 60

# Korea: 09:00 ~ 10:00 KST
KR_MARKET_OPEN = "09:00"
KR_EARLY_END = "10:00"
KR_TIMEZONE = "Asia/Seoul"

# US: 09:30 ~ 10:30 ET
US_MARKET_OPEN = "09:30"
US_EARLY_END = "10:30"
US_TIMEZONE = "America/New_York"

# ── Data Collection ───────────────────────────────────
TV_RATE_LIMIT_SEC = 2.0
TV_MAX_RETRIES = 3
TV_MAX_BARS = 5000  # ~13 trading days of 1-min bars
TV_BACKOFF_BASE = 2  # exponential backoff base

# ── Labeling ──────────────────────────────────────────
# Peak/trough detection via scipy.signal.find_peaks
PEAK_PROMINENCE_PCT = 0.003   # 0.3% of open price
PEAK_DISTANCE = 5             # minimum 5 bars between peaks
PEAK_WIDTH = 3                # minimum width of peak

# Labels: 0=neither, 1=peak, 2=trough
LABEL_NEITHER = 0
LABEL_PEAK = 1
LABEL_TROUGH = 2

# ── Features ──────────────────────────────────────────
LOOKBACK_WINDOW = 10  # bars of history as input
RETURN_WINDOWS = [1, 2, 3, 5, 10]  # for multi-period returns

# Technical indicator periods
MA_PERIODS = [5, 10, 20]
RSI_PERIODS = [7, 14]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# ── Model ─────────────────────────────────────────────
# Time-based split
TRAIN_YEARS = 4
VAL_MONTHS = 6
TEST_MONTHS = 12

# Walk-forward
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 1

# LightGBM defaults
LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "is_unbalance": True,
    "verbosity": -1,
    "num_threads": -1,
    "seed": 42,
}

# LSTM defaults
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 256
LSTM_EPOCHS = 50

RANDOM_SEED = 42
