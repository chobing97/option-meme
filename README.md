# option-meme

정규장 초기(장 시작 후 60분) 주가 전환점(피크/트로프)을 머신러닝으로 탐지하는 파이프라인.

한국(KOSPI/KOSDAQ) 및 미국(NYSE/NASDAQ) 시장의 1분봉 데이터를 수집하고, scipy 기반 피크 탐지로 라벨링한 뒤, LightGBM과 LSTM+Attention 모델로 실시간 전환점을 예측합니다.

## 주요 기능

| 단계 | 모듈 | 설명 |
|------|------|------|
| **Phase 0** | `collector` | yfinance + tvDatafeed 멀티소스 1분봉 수집 |
| **Phase 1** | `labeler` | scipy.signal.find_peaks 기반 피크/트로프 라벨링 |
| **Phase 2** | `features` | 가격·거래량·기술적·시간 피처 + 10-bar lookback 윈도우 |
| **Phase 3** | `model` | LightGBM / Bidirectional LSTM+Attention 학습·평가 |

## 기술 스택

- **Python 3.12** (tvDatafeed 호환)
- **데이터 수집**: yfinance, tvDatafeed (tradingview-datafeed 2.1.1), pykrx
- **데이터 처리**: pandas, pyarrow (Parquet), scipy, ta
- **ML/DL**: scikit-learn, LightGBM, XGBoost, PyTorch, Optuna
- **유틸**: exchange-calendars, loguru, matplotlib, plotly

## 설치

```bash
# 1. 저장소 클론
git clone <repo-url> option-meme
cd option-meme

# 2. Python 3.12 가상환경 생성
python3.12 -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install -e ".[dev]"

# 4. tvDatafeed 별도 설치
pip install --upgrade tvDatafeed

# 5. macOS: LightGBM용 OpenMP 런타임
brew install libomp
```

> **참고**: 시스템 Python 3.14는 tvDatafeed와 호환되지 않습니다. 반드시 Python 3.12를 사용하세요.

## 사용법

```bash
# 전체 파이프라인 실행 (모든 시장)
./run.sh all

# 특정 단계만 실행
./run.sh collector --market kr      # 한국 시장 데이터 수집 (증분)
./run.sh collector --full           # 전체 재수집
./run.sh labeler --market us        # 미국 시장 라벨링
./run.sh features                   # 피처 엔지니어링 (전체 시장)
./run.sh model --model gbm         # LightGBM만 학습
./run.sh model --model lstm         # LSTM만 학습

# 한국 시장 전체 파이프라인
./run.sh all --market kr
```

### 테스트

```bash
pytest tests/ -v
```

## 프로젝트 구조

```
option-meme/
├── run.sh                    # CLI 진입점
├── run_pipeline.py           # 파이프라인 오케스트레이터
├── pyproject.toml            # 프로젝트 메타데이터 & 의존성
│
├── config/
│   ├── settings.py           # 중앙 설정 (경로, 시장 시간, 하이퍼파라미터)
│   └── symbols/              # 종목 리스트 (CSV)
│       ├── kr_symbols.csv
│       ├── us_stocks.csv
│       └── us_etf_index.csv
│
├── src/
│   ├── collector/            # Phase 0: 데이터 수집
│   │   ├── bar_fetcher.py    # 멀티소스 수집 오케스트레이션
│   │   ├── tv_client.py      # TradingView API 래퍼
│   │   ├── storage.py        # Parquet 저장 & 병합
│   │   └── collection_tracker.py  # SQLite 진행 추적
│   │
│   ├── labeler/              # Phase 1: 라벨링
│   │   ├── label_generator.py      # 라벨링 파이프라인
│   │   ├── peak_trough_detector.py # scipy 피크 탐지
│   │   └── session_extractor.py    # 초기 세션 추출
│   │
│   ├── features/             # Phase 2: 피처 엔지니어링
│   │   ├── feature_pipeline.py     # 피처 통합 오케스트레이터
│   │   ├── price_features.py       # 수익률, 캔들스틱
│   │   ├── technical_features.py   # MA, RSI, MACD, BB
│   │   ├── volume_features.py      # 거래량 분석
│   │   └── time_features.py        # 시간 패턴
│   │
│   └── model/                # Phase 3: 모델 학습 & 평가
│       ├── dataset.py        # 시간 기반 분할 & PyTorch Dataset
│       ├── train_gbm.py      # LightGBM + Optuna 하이퍼파라미터 최적화
│       ├── train_lstm.py     # Bidirectional LSTM+Attention
│       └── evaluate.py       # PR-AUC, 백테스팅
│
├── tests/                    # pytest 테스트 스위트
│
└── data/
    ├── raw/{kr,us}/          # 1분봉 OHLCV Parquet
    ├── processed/
    │   ├── labeled/          # 라벨링된 데이터
    │   └── featured/         # 피처 데이터
    ├── models/               # 학습된 모델 파일
    └── metadata/             # collection.db (SQLite)
```

## 데이터 흐름

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Phase 0    │     │  Phase 1    │     │  Phase 2    │     │  Phase 3    │
│  Collector  │────▶│  Labeler    │────▶│  Features   │────▶│  Model      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
  yfinance +          scipy.signal        100+ 피처           LightGBM
  tvDatafeed          find_peaks         (가격·거래량·       LSTM+Attention
       │                   │             기술적·시간)              │
       ▼                   ▼                   ▼                   ▼
  raw/*.parquet    labeled/*.parquet   featured/*.parquet    models/*.{txt,pt}
```

### 데이터 수집 전략

1. **yfinance** (기본): ~30-60일의 1분봉 데이터 (7일 윈도우로 분할 요청)
2. **tvDatafeed** (오버레이): 최신 ~13거래일 (5000 bars), yfinance 데이터에 덮어쓰기

## 모델

### LightGBM

- **방식**: 피크/트로프 각각 독립 이진 분류기
- **하이퍼파라미터**: Optuna 자동 튜닝
- **불균형 처리**: `is_unbalance=True`
- **평가 지표**: PR-AUC (Average Precision)

### LSTM+Attention

```
Input (batch, 10, n_features)
  │
  ▼
BiLSTM (hidden=128, layers=2, bidirectional)
  │
  ▼
Attention (softmax weighted sum)
  │
  ▼
Classifier (FC → ReLU → Dropout → FC → Sigmoid)
  │
  ▼
Output (batch, 1)  ← 피크/트로프 확률
```

- **입력**: 10-bar lookback 윈도우
- **불균형 처리**: `BCEWithLogitsLoss(pos_weight=...)`
- **학습**: Adam (lr=1e-3), batch=256, epochs=50 + early stopping

### 학습 설정

- **시간 기반 분할** (셔플 없음 → look-ahead bias 방지)
  - Train: 4년 / Validation: 6개월 / Test: 12개월
- **Walk-forward**: 6개월 학습 + 1개월 테스트 롤링 윈도우
- 피크(label=1)와 트로프(label=2) 각각 독립 모델 학습

## 주요 설정 (`config/settings.py`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `EARLY_SESSION_MINUTES` | 60 | 분석 대상 초기 세션 길이 |
| KR 세션 | 09:00~10:00 KST | 한국 시장 |
| US 세션 | 09:30~10:30 ET | 미국 시장 |
| `PEAK_PROMINENCE_PCT` | 0.3% | 피크 탐지 최소 prominence |
| `PEAK_DISTANCE` | 5 bars | 피크 간 최소 거리 |
| `LOOKBACK_WINDOW` | 10 bars | 피처 lookback 길이 |

## 라이선스

MIT
