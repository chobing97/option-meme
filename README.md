# option-meme

정규장 초기(장 시작 후 60분) 주가 전환점(피크/트로프)을 머신러닝으로 탐지하는 파이프라인.

한국(KOSPI/KOSDAQ) 및 미국(NYSE/NASDAQ) 시장의 1분봉 데이터를 수집하고, scipy 기반 피크 탐지로 라벨링한 뒤, LightGBM과 LSTM+Attention 모델로 실시간 전환점을 예측합니다.

## 주요 기능

| 단계 | 모듈 | 설명 |
|------|------|------|
| **Phase 0** | `collector` | yfinance + tvDatafeed 멀티소스 1분봉 수집 |
| **Phase 1** | `labeler` | scipy.signal.find_peaks 기반 피크/트로프 라벨링 |
| **Phase 2** | `features` | 가격·거래량·기술적·시간 피처 + lookback 윈도우 |
| **Phase 3** | `model` | LightGBM / Bidirectional LSTM+Attention 학습·평가 |
| **Phase 4** | `batch_predict` | 전 종목 배치 예측 & parquet 저장 |

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
# 전체 파이프라인 실행 (모든 시장, 모든 변형)
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

# 단일 변형 실행
./run.sh labeler --label-config L2
./run.sh features --label-config L2 --model-config M3
./run.sh model --label-config L2 --model-config M3
./run.sh batch_predict --label-config L2 --model-config M3

# 전체 변형 순회 (8조합)
./run.sh labeler --label-config all
./run.sh features --label-config all --model-config all
./run.sh model --label-config all --model-config all
./run.sh batch_predict --label-config all --model-config all
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
│   ├── variants.py           # Multi-Config 변형 레지스트리 (L1/L2 × M1~M4)
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
│   ├── model/                # Phase 3: 모델 학습 & 평가
│   │   ├── dataset.py        # 시간 기반 분할 & PyTorch Dataset
│   │   ├── train_gbm.py      # LightGBM + Optuna 하이퍼파라미터 최적화
│   │   ├── train_lstm.py     # Bidirectional LSTM+Attention
│   │   └── evaluate.py       # PR-AUC, 백테스팅
│   │
│   └── inference/            # Phase 4: 추론
│       └── predict.py        # 단일 종목 / 배치 예측
│
├── tests/                    # pytest 테스트 스위트
│
└── data/
    ├── raw/{kr,us}/                              # 1분봉 OHLCV Parquet
    ├── processed/
    │   ├── labeled/{L1,L2}/                      # 변형별 라벨링 데이터
    │   └── featured/{L1,L2}/{M1,M2,M3,M4}/      # 변형별 피처 데이터
    ├── models/{L1,L2}/{M1,M2,M3,M4}/            # 변형별 모델 파일
    ├── predictions/labeled/{L1,L2}/{M1,M2,M3,M4}/ # 변형별 예측 결과
    └── metadata/                                  # collection.db (SQLite)
```

## 데이터 흐름

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Phase 0  │    │ Phase 1  │    │ Phase 2  │    │ Phase 3  │    │  Phase 4     │
│Collector │───▶│ Labeler  │───▶│ Features │───▶│  Model   │───▶│batch_predict │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────────┘
      │              │               │               │                │
 yfinance +     L1/L2 변형      M1~M4 변형       GBM+LSTM         전 종목
 tvDatafeed    prominence/     lookback/fill    peak/trough       배치 예측
      │          width              │               │                │
      ▼              ▼               ▼               ▼                ▼
 raw/*.parquet  labeled/L*/   featured/L*/M*/  models/L*/M*/  predictions/L*/M*/
```

### 데이터 수집 전략

1. **yfinance** (기본): ~30-60일의 1분봉 데이터 (7일 윈도우로 분할 요청)
2. **tvDatafeed** (오버레이): 최신 ~13거래일 (5000 bars), yfinance 데이터에 덮어쓰기

## Multi-Config 파이프라인

Labeling 파라미터(prominence, width)와 Model 파라미터(lookback, 초반 바 처리)의 조합에 따라 성능이 크게 달라지므로,
8가지 조합(L1/L2 × M1~M4)을 모두 학습하고 비교할 수 있다.

### Labeling 변형

| ID | prominence | width | distance | 설명 |
|---|---|---|---|---|
| **L1** | 0.003 (0.3%) | 3 | 5 | 보수적: 큰 전환점만 탐지 |
| **L2** | 0.002 (0.2%) | 1 | 5 | 민감: 작은 전환점까지 탐지 |

### Model 변형

| ID | GBM lookback | LSTM lookback | 초반 바 처리 | Feature 수 |
|---|---|---|---|---|
| **M1** | 10 | 10 | drop (제거) | 495 (45 base + 450 lag) |
| **M2** | 10 | 10 | 0-fill (패딩) | 495 (45 base + 450 lag) |
| **M3** | 5 | 5 | 0-fill (패딩) | 270 (45 base + 225 lag) |
| **M4** | 0 (없음) | 10 (패딩) | 0-fill (패딩) | 45 (base only) |

### 실행 결과 (2026-02-22, threshold=0.5)

#### Labeling 통계

| Config | Market | 전체 바 | Peaks | Troughs | Peak% | Trough% |
|---|---|---|---|---|---|---|
| L1 | KR | 26,998 | 920 | 987 | 3.4% | 3.7% |
| L1 | US | 64,516 | 1,764 | 1,686 | 2.7% | 2.6% |
| L2 | KR | 26,998 | 2,021 | 2,065 | 7.5% | 7.6% |
| L2 | US | 64,516 | 3,846 | 3,744 | 6.0% | 5.8% |

#### GBM Test PR-AUC

| Variant | KR Peak | KR Trough | US Peak | US Trough |
|---|---|---|---|---|
| L1/M1 | 0.4417 | 0.3025 | 0.3382 | 0.5047 |
| L1/M2 | 0.4323 | 0.3352 | 0.3652 | 0.4720 |
| L1/M3 | **0.4678** | 0.3127 | **0.3778** | 0.4915 |
| L1/M4 | 0.3738 | 0.2795 | 0.3289 | 0.3864 |
| L2/M1 | 0.6351 | 0.6664 | 0.6447 | 0.6814 |
| L2/M2 | 0.6405 | 0.6529 | 0.6576 | 0.6903 |
| L2/M3 | 0.6438 | **0.6672** | **0.6632** | 0.6887 |
| L2/M4 | 0.6061 | 0.6143 | 0.6061 | 0.6560 |

#### Batch Predict 결과 (threshold=0.5)

| Variant | KR Rows | KR Peaks | KR Troughs | KR Peak% | KR Trough% | US Rows | US Peaks | US Troughs | US Peak% | US Trough% |
|---|---|---|---|---|---|---|---|---|---|---|
| L1/M1 | 26,986 | 2,297 | 2,659 | 8.5% | 9.9% | 64,504 | 5,093 | 4,548 | 7.9% | 7.1% |
| L1/M2 | 26,998 | 2,338 | 2,462 | 8.7% | 9.1% | 64,516 | 5,129 | 5,004 | 7.9% | 7.8% |
| L1/M3 | 26,998 | 2,481 | 2,203 | 9.2% | 8.2% | 64,516 | 5,204 | 4,779 | 8.1% | 7.4% |
| L1/M4 | 26,998 | 2,850 | 2,396 | 10.6% | 8.9% | 64,516 | 6,712 | 6,335 | 10.4% | 9.8% |
| L2/M1 | 26,986 | 3,561 | 3,675 | 13.2% | 13.6% | 64,504 | 7,441 | 6,894 | 11.5% | 10.7% |
| L2/M2 | 26,998 | 3,519 | 3,683 | 13.0% | 13.6% | 64,516 | 7,472 | 7,212 | 11.6% | 11.2% |
| L2/M3 | 26,998 | 3,828 | 4,015 | 14.2% | 14.9% | 64,516 | 7,526 | 7,064 | 11.7% | 10.9% |
| L2/M4 | 26,998 | 3,876 | 3,724 | 14.4% | 13.8% | 64,516 | 8,322 | 7,652 | 12.9% | 11.9% |

#### 관찰

- **L2 >> L1 (PR-AUC)**: L2의 PR-AUC가 L1 대비 현저히 높음 (KR peak: 0.44 → 0.64). prominence를 낮추면 positive sample이 2배 이상 늘어나 모델 학습에 유리
- **M3이 최적**: lookback=5로도 M1/M2(lookback=10) 대비 PR-AUC 유지 또는 개선되며, feature 수는 55% 절감 (495 → 270)
- **M4는 성능 하락**: lag feature를 완전히 제거하면 PR-AUC가 확실히 낮아짐 (L1 peak KR: 0.47 → 0.37)
- **M1 vs M2**: drop vs 0-fill 차이는 미미하나, M2가 전체 바를 보존하므로 데이터 손실 없음
- **최적 조합**: L2/M3 (prominence=0.002, lookback=5, 0-fill) — 가장 높은 PR-AUC + 적은 feature 수

## 모델

### LightGBM

- **방식**: 피크/트로프 각각 독립 이진 분류기
- **하이퍼파라미터**: Optuna 자동 튜닝
- **불균형 처리**: `is_unbalance=True`
- **평가 지표**: PR-AUC (Average Precision)

### LSTM+Attention

```
Input (batch, lookback, n_features)
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

- **입력**: lookback-bar 윈도우 (M1/M2/M4=10, M3=5)
- **초반 바 처리**: `fill_method=drop` (M1) 또는 `fill_method=0fill` (M2/M3/M4: zero-padding)
- **불균형 처리**: Focal Loss (alpha=0.25, gamma=2.0)
- **학습**: AdamW (lr=1e-3), batch=256, epochs=50 + early stopping

### 학습 설정

- **시간 기반 분할** (셔플 없음 → look-ahead bias 방지)
  - Train: 4년 / Validation: 6개월 / Test: 12개월
- **Walk-forward**: 6개월 학습 + 1개월 테스트 롤링 윈도우
- 피크(label=1)와 트로프(label=2) 각각 독립 모델 학습

## 주요 설정

### 기본 설정 (`config/settings.py`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `EARLY_SESSION_MINUTES` | 60 | 분석 대상 초기 세션 길이 |
| KR 세션 | 09:00~10:00 KST | 한국 시장 |
| US 세션 | 09:30~10:30 ET | 미국 시장 |
| `PEAK_PROMINENCE_PCT` | 0.2% | 피크 탐지 기본 prominence |
| `PEAK_DISTANCE` | 5 bars | 피크 간 최소 거리 |
| `LOOKBACK_WINDOW` | 5 bars | 피처 기본 lookback 길이 |

### 변형 설정 (`config/variants.py`)

Labeling과 Model 파라미터의 조합은 `config/variants.py`의 `LABEL_CONFIGS`와 `MODEL_CONFIGS`에서 관리된다.
`--label-config`와 `--model-config` CLI 인자로 선택하며, `all`을 지정하면 전체 조합을 순회한다.

## 라이선스

MIT
