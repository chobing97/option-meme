# option-meme

정규장 초기(장 시작 후 60분) 주가 전환점(피크/트로프)을 머신러닝으로 탐지하고, 풋옵션 매매 시그널을 생성하는 엔드투엔드 파이프라인.

한국(KOSPI/KOSDAQ) 및 미국(NYSE/NASDAQ) 시장의 1분봉 데이터를 멀티소스(Databento, yfinance, tvDatafeed)로 수집하고, scipy 기반 피크 탐지로 라벨링한 뒤, LightGBM과 Bidirectional LSTM+Attention 모델로 실시간 전환점을 예측합니다. 예측 결과는 풋옵션 매매 엔진으로 연결됩니다.

## 주요 기능

| 단계 | 모듈 | 설명 |
|------|------|------|
| **Phase 0** | `collector` | Databento + yfinance + tvDatafeed 멀티소스 1분봉 수집 |
| **Phase 1** | `labeler` | scipy.signal.find_peaks 기반 피크/트로프 라벨링 |
| **Phase 2** | `features` | 가격·거래량·기술적·시간 피처 + lookback 윈도우 |
| **Phase 3** | `model` | LightGBM / Bidirectional LSTM+Attention 학습·평가 |
| **Phase 3.5** | `ensemble` | LSTM 캘리브레이션 + GBM/LSTM 가중 앙상블 (US) |
| **Phase 4** | `batch_predict` | 전 종목 배치 예측 & parquet 저장 |
| **Phase 5** | `trading` | 풋옵션 시그널 생성 & 매매 엔진 (Mock/실전) |

## 기술 스택

- **Python 3.12** (tvDatafeed 호환)
- **데이터 수집**: Databento (주식 OHLCV + OPRA 옵션), yfinance, tvDatafeed, pykrx
- **데이터 처리**: pandas, pyarrow (Parquet), scipy, ta
- **ML/DL**: scikit-learn, LightGBM, XGBoost, PyTorch (MPS/CUDA), Optuna
- **실험 추적**: MLflow (파라미터/메트릭/아티팩트 중앙 관리)
- **트레이딩**: Black-Scholes 풋옵션 프라이싱, 한국투자증권 OpenAPI
- **유틸**: exchange-calendars, loguru, matplotlib, plotly, Streamlit (대시보드)

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

# 6. (선택) Databento API 키 설정
export DATABENTO_API_KEY="your-api-key"
```

> **참고**: 시스템 Python 3.14는 tvDatafeed와 호환되지 않습니다. 반드시 Python 3.12를 사용하세요.

## 사용법

### 전체 파이프라인

```bash
# 전체 파이프라인 (모든 시장, 모든 변형)
./optionmeme all

# 미국 시장만 전체 파이프라인
./optionmeme all --market us

# 한국 시장만 전체 파이프라인
./optionmeme all --market kr
```

### 단계별 실행

```bash
# Phase 0: 데이터 수집
./optionmeme collector --market us      # 미국 시장 증분 수집
./optionmeme collector --full           # 전체 재수집

# Phase 1: 라벨링
./optionmeme labeler --market us
./optionmeme labeler --label-config L2  # L2 변형만

# Phase 2: 피처 엔지니어링
./optionmeme features --label-config all --model-config all

# Phase 3: 모델 학습
./optionmeme model --model gbm          # LightGBM만
./optionmeme model --model lstm          # LSTM만
./optionmeme model --label-config L2 --model-config M1  # 특정 조합

# Phase 4: 배치 예측
./optionmeme batch_predict --label-config all --model-config all

# Phase 5: 트레이딩 (모의)
./optionmeme trade --market us
```

### 미국 주식 Databento 수집 (고품질 데이터)

```bash
# 1. DBN 원본 다운로드 (20종목 × 3거래소)
.venv/bin/python scripts/download_stock_ohlcv.py --dry-run  # 비용 확인
.venv/bin/python scripts/download_stock_ohlcv.py            # 실제 다운로드

# 2. 통합 ETL (Databento → Parquet 변환 + 나머지 종목 yfinance/TV 수집)
.venv/bin/python scripts/build_us_ohlcv.py --yes            # 비대화형 실행
.venv/bin/python scripts/build_us_ohlcv.py --databento-only # Databento만
.venv/bin/python scripts/build_us_ohlcv.py --legacy-only    # yfinance+TV만
.venv/bin/python scripts/build_us_ohlcv.py --dry-run        # 미리보기
```

### 대시보드

```bash
./dashboard.sh   # Streamlit 대시보드 실행 (http://localhost:8501)
```

### 테스트

```bash
pytest tests/ -v
```

## 프로젝트 구조

```
option-meme/
├── optionmeme                # CLI 진입점 (venv 자동 활성화 + 로그)
├── run_pipeline.py           # 파이프라인 오케스트레이터
├── pyproject.toml            # 프로젝트 메타데이터 & 의존성
├── dashboard.sh              # Streamlit 대시보드 런처
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
│   │   ├── bar_fetcher.py    # 멀티소스 수집 오케스트레이션 (yfinance → tvDatafeed)
│   │   ├── tv_client.py      # TradingView API 래퍼 (재시도, 지수 백오프)
│   │   ├── storage.py        # Parquet 저장 & 병합 (연도별 분할, source 추적)
│   │   ├── collection_tracker.py  # SQLite 수집 진행 추적
│   │   └── stock_info_db.py  # 종목 기본정보 SQLite DB (섹터, 시총 등)
│   │
│   ├── labeler/              # Phase 1: 라벨링
│   │   ├── label_generator.py      # 라벨링 파이프라인
│   │   ├── peak_trough_detector.py # scipy.signal.find_peaks 기반 탐지
│   │   └── session_extractor.py    # 초기 세션(장 시작 60분) 추출
│   │
│   ├── features/             # Phase 2: 피처 엔지니어링
│   │   ├── feature_pipeline.py     # 피처 통합 오케스트레이터
│   │   ├── price_features.py       # 수익률, 모멘텀, 가속도, 캔들스틱
│   │   ├── technical_features.py   # MA, RSI, MACD, Bollinger Bands
│   │   ├── volume_features.py      # 거래량 비율, OBV
│   │   └── time_features.py        # 경과시간, 분/요일 인코딩
│   │
│   ├── model/                # Phase 3: 모델 학습 & 평가
│   │   ├── dataset.py        # 시간 기반 분할 & PyTorch Dataset
│   │   ├── train_gbm.py      # LightGBM + Optuna 하이퍼파라미터 최적화
│   │   ├── train_lstm.py     # Bidirectional LSTM + Attention
│   │   └── evaluate.py       # PR-AUC, 정밀도-재현율 평가
│   │
│   ├── inference/            # Phase 4: 추론
│   │   └── predict.py        # 단일 종목 / 배치 예측
│   │
│   └── trading/              # Phase 5: 트레이딩
│       ├── engine.py         # 멀티 심볼 트레이딩 루프 (시그널 → 주문)
│       ├── signal_detector.py # 바 누적 + 피처 → 모델 추론 → 시그널
│       ├── option_pricer.py  # Black-Scholes 풋옵션 가격 계산
│       ├── trade_db.py       # 거래 이력 SQLite DB
│       ├── broker/           # 주문 실행 (mock_broker / 실전 연동)
│       │   ├── base.py       # Broker ABC, Order, Signal 타입
│       │   └── mock_broker.py # 백테스팅용 모의 브로커
│       ├── datafeed/         # 실시간 데이터 피드
│       │   ├── base.py       # DataFeed ABC
│       │   └── mock_feed.py  # 히스토리 데이터 기반 모의 피드
│       └── notifier/         # 매매 알림
│           ├── base.py       # Notifier ABC, TradeEvent
│           └── console.py    # 콘솔 출력 노티파이어
│
├── scripts/                  # 유틸리티 & 테스트 스크립트
│   ├── build_us_ohlcv.py     # US 통합 ETL (Databento + yfinance + TV)
│   ├── download_stock_ohlcv.py # Databento DBN 일괄 다운로드
│   ├── collect_opra_puts.py  # OPRA 풋옵션 심볼 탐색 & 비용 산정
│   ├── kis_auth.py           # 한국투자증권 OAuth2 인증 모듈
│   ├── kis_to_occ.py         # 한투 ↔ OCC 종목코드 변환
│   └── test_kis_*.py         # 한투 OpenAPI 테스트 스크립트 (~15종)
│
├── dashboard/                # Streamlit 대시보드
│   ├── app.py                # 앱 진입점
│   ├── data_loader.py        # 데이터 로더
│   ├── components/           # 공용 컴포넌트 (charts, filters, metrics)
│   └── pages/                # 5개 페이지
│       ├── 1_Raw_Data.py     # 원시 1분봉 데이터 탐색
│       ├── 2_Labeling.py     # 피크/트로프 라벨링 시각화
│       ├── 3_Features.py     # 피처 분포 & 상관관계
│       ├── 4_Model_Performance.py  # 모델 성능 비교
│       └── 5_Predictions.py  # 예측 결과 시각화
│
├── tests/                    # pytest 테스트 스위트
│   ├── collector/            # 수집 모듈 테스트
│   ├── labeler/              # 라벨링 테스트
│   ├── features/             # 피처 테스트
│   ├── model/                # 모델 테스트
│   ├── inference/            # 추론 테스트
│   └── trading/              # 트레이딩 테스트
│
└── data/
    ├── raw/
    │   ├── {kr,us}/                              # 1분봉 OHLCV Parquet (연도별 분할)
    │   ├── databento/us/{SYMBOL}/                # Databento DBN 원본 (.dbn.zst)
    │   └── stock_info.db                         # 종목 기본정보 SQLite
    ├── processed/
    │   ├── labeled/{L1,L2}/                      # 변형별 라벨링 데이터
    │   └── featured/{L1,L2}/{M1,M2,M3,M4}/      # 변형별 피처 데이터
    ├── models/{L1,L2}/{M1,M2,M3,M4}/            # 변형별 모델 파일
    ├── predictions/labeled/{L1,L2}/{M1,M2,M3,M4}/ # 변형별 예측 결과
    ├── mlruns/                                   # MLflow 실험 추적 데이터
    └── metadata/
        ├── collection.db                         # 수집 진행 추적 SQLite
        └── logs/                                 # 파이프라인 실행 로그
```

## 데이터 흐름

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Phase 0  │    │ Phase 1  │    │ Phase 2  │    │ Phase 3  │    │  Phase 4     │    │ Phase 5  │
│Collector │───▶│ Labeler  │───▶│ Features │───▶│  Model   │───▶│batch_predict │───▶│ Trading  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────────┘    └──────────┘
      │              │               │               │                │                  │
 Databento +     L1/L2 변형      M1~M4 변형       GBM+LSTM         전 종목           풋옵션
 yfinance +     prominence/     lookback/fill    peak/trough       배치 예측        시그널+매매
 tvDatafeed      width              │               │                │                  │
      ▼              ▼               ▼               ▼                ▼                  ▼
 raw/*.parquet  labeled/L*/   featured/L*/M*/  models/L*/M*/  predictions/L*/M*/    trade_db
```

### 데이터 수집 전략 (3-tier)

| Tier | 소스 | 커버리지 | 품질 | 비용 |
|------|------|----------|------|------|
| **Tier 1** | Databento | 2020~현재, 20종목, 3거래소(XNAS/XNYS/ARCX) 합산 | 최고 (거래소 직접) | 유료 (~$34) |
| **Tier 2** | tvDatafeed | 최근 ~13거래일, 전 종목 | 높음 (TradingView) | 무료 |
| **Tier 3** | yfinance | 최근 ~60일, 전 종목 | 보통 (지연/누락 가능) | 무료 |

#### Databento 종목 (20종목)

```
주식 (13): TSLA, AAPL, NVDA, AMZN, AMD, META, MSFT, GOOGL, PLTR, MARA, AMC, BAC, BABA
ETF  (7) : SPY, QQQ, IWM, TQQQ, SQQQ, TLT, SLV
```

- 3개 거래소(XNAS.ITCH, XNYS.PILLAR, ARCX.PILLAR)에서 분봉 데이터를 개별 다운로드
- 동일 분봉의 3거래소 데이터를 OHLCV 합산 (open=가중평균, high=max, low=min, close=가중평균, volume=sum)
- 결과: 2020-01-01 ~ 현재까지 약 6년치 1분봉

#### 수집 우선순위

1. Databento 데이터가 있으면 해당 종목은 Databento만 사용
2. Databento 미지원 종목(54종)은 yfinance(base) → tvDatafeed(overlay) 2단계 수집
3. Parquet 파일에 `source` 컬럼으로 데이터 출처 추적 (`databento`, `yfinance`, `tvdatafeed`)
4. 사용자 허가 시 yfinance/tvDatafeed 데이터를 Databento 데이터로 교체 가능

### Parquet 저장 구조

```
data/raw/us/TSLA/
├── 2020.parquet    # 2020년 1분봉
├── 2021.parquet
├── ...
└── 2026.parquet
```

- 연도별 분할 저장, 증분 병합 (`keep="last"`로 중복 제거)
- 스키마: `datetime, open, high, low, close, volume, source`
- `source` 컬럼으로 데이터 출처 식별

### SQLite 메타데이터

| DB | 경로 | 용도 |
|---|---|---|
| `collection.db` | `data/metadata/` | 수집 진행 상태 (심볼별 소스/날짜/바 수/상태) |
| `stock_info.db` | `data/raw/` | 종목 기본정보 (섹터, 시총, 통화 등) |

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

### 실행 결과 (2026-03-05, threshold=0.5)

#### Labeling 통계

| Config | Market | 전체 바 | Peaks | Troughs | Peak% | Trough% |
|---|---|---|---|---|---|---|
| L1 | KR | 23,398 | 804 | 865 | 3.4% | 3.7% |
| L1 | US | 57,591 | 1,566 | 1,519 | 2.7% | 2.6% |
| L2 | KR | 23,398 | 1,771 | 1,806 | 7.6% | 7.7% |
| L2 | US | 57,591 | 3,502 | 3,419 | 6.1% | 5.9% |

#### GBM Test PR-AUC

| Variant | KR Peak | KR Trough | US Peak | US Trough | Features |
|---|---|---|---|---|---|
| L1/M1 | 0.3739 | 0.2742 | 0.4496 | 0.4470 | 495 |
| L1/M2 | 0.3710 | 0.3248 | 0.4449 | 0.4467 | 495 |
| L1/M3 | **0.3991** | 0.3102 | 0.4508 | 0.4511 | 270 |
| L1/M4 | 0.3267 | 0.3017 | 0.3931 | 0.3976 | 45 |
| L2/M1 | 0.6202 | 0.6474 | 0.6988 | 0.7017 | 495 |
| L2/M2 | 0.6172 | 0.6398 | 0.7002 | 0.7024 | 495 |
| L2/M3 | 0.6084 | **0.6621** | **0.7011** | **0.7041** | 270 |
| L2/M4 | 0.5843 | 0.6054 | 0.6578 | 0.6629 | 45 |

#### LSTM Test PR-AUC

| Variant | KR Peak | KR Trough | US Peak | US Trough | Features |
|---|---|---|---|---|---|
| L1/M1 | 0.1260 | 0.1376 | 0.3347 | 0.1570 | 45 |
| L1/M2 | 0.1651 | 0.1192 | 0.3193 | 0.3386 | 45 |
| L1/M3 | 0.1655 | 0.1070 | 0.3421 | 0.2506 | 45 |
| L1/M4 | 0.1651 | 0.1192 | 0.3193 | 0.3386 | 45 |
| L2/M1 | 0.2953 | 0.3079 | **0.5867** | 0.4095 | 45 |
| L2/M2 | **0.3304** | 0.2871 | 0.5610 | **0.5911** | 45 |
| L2/M3 | 0.3266 | 0.2865 | 0.2214 | 0.5100 | 45 |
| L2/M4 | **0.3304** | 0.2871 | 0.5610 | **0.5911** | 45 |

#### PR-AUC 평균 비교 (GBM vs LSTM)

| Variant | GBM KR | GBM US | LSTM KR | LSTM US |
|---|---|---|---|---|
| L1/M1 | 0.3240 | 0.4483 | 0.1318 | 0.2458 |
| L1/M2 | 0.3479 | 0.4458 | 0.1422 | 0.3290 |
| L1/M3 | 0.3547 | 0.4509 | 0.1362 | 0.2963 |
| L1/M4 | 0.3142 | 0.3953 | 0.1422 | 0.3290 |
| L2/M1 | 0.6338 | **0.7002** | 0.3016 | 0.4981 |
| L2/M2 | 0.6285 | 0.7013 | **0.3087** | **0.5760** |
| L2/M3 | **0.6353** | **0.7026** | 0.3065 | 0.3657 |
| L2/M4 | 0.5949 | 0.6604 | **0.3087** | **0.5760** |

#### 관찰

- **L2 >> L1 (PR-AUC)**: L2의 PR-AUC가 L1 대비 현저히 높음 (KR peak: 0.37 → 0.62). prominence를 낮추면 positive sample이 2배 이상 늘어나 모델 학습에 유리
- **L2/M3이 GBM US 최고**: US avg 0.7026으로 M1(0.7002), M2(0.7013)를 근소하게 상회하며, feature 수는 55% 절감 (495 → 270)
- **L2/M3이 GBM KR 최고 평균**: KR avg 0.6353으로 M1(0.6338)을 근소하게 상회
- **GBM threshold 0.7에서 F1 최고**: L2 변형에서 threshold를 높이면 precision이 상승하여 F1@0.7이 F1@0.3보다 높음
- **LSTM은 GBM 대비 전반적으로 낮음**: KR ~0.30 vs ~0.63, US ~0.58 vs ~0.70. calibration 문제로 F1@0.5 이상이 거의 0
- **LSTM L2/M2가 US 최고**: US avg 0.5760 (peak 0.56, trough 0.59). ensemble 후보이나 GBM과 격차 존재
- **LSTM KR은 사실상 미작동**: PR-AUC 0.13~0.33으로 실용성 낮음. 데이터 부족(KR 4,680 bars)이 주요 원인
- **M4 = M2 (LSTM)**: LSTM은 base features만 사용하므로 M2와 M4가 동일 결과 (lookback=10, fill=0fill)
- **최적 조합**: GBM은 L2/M3 (US+KR 최고 + 효율성), LSTM은 L2/M2 (US 최고)

## 앙상블 파이프라인 (US)

GBM의 통계적 강점과 LSTM의 시계열 패턴 학습 능력을 결합한 **2단계 앙상블**을 US 시장에 적용할 수 있다.

> **KR 시장 주의**: KR LSTM은 데이터 부족(~4,680 bars)으로 PR-AUC 0.13~0.33 수준이므로 앙상블 효과가 없거나 오히려 GBM 단독보다 저하될 수 있다.

### 앙상블 구성

```
GBM(L2/M3)  ─────────────────────────────────────┐
                                                   ├─▶  w_gbm * GBM + (1-w_gbm) * LSTM_cal
LSTM(L2/M3) ─▶ IsotonicRegression ─▶ LSTM_cal ───┘
                    (calibration)
```

### 왜 앙상블인가?

두 모델은 **완전히 다른 방식**으로 같은 문제를 본다.

| | GBM | LSTM |
|--|--|--|
| 입력 | 270개 hand-crafted 피처 (lag 포함) | 45개 raw 피처 시퀀스 (10 steps) |
| 학습 방식 | 통계적 패턴 (tree splits) | 시계열 흐름 (순서 기억) |
| 강점 | 전체적인 정확도, 안정성 | GBM이 놓치는 단기 패턴 |

→ 서로 다른 실수를 하므로, 잘 합치면 혼자보다 낫다.

### Phase 3.5: 앙상블 학습 (`ensemble` 스테이지)

| 단계 | 내용 |
|------|------|
| **1. Calibration** | LSTM val 예측값 → `IsotonicRegression` 피팅 → 확률 캘리브레이션 |
| **2. 가중치 탐색** | val set에서 w_gbm ∈ [0.50, 1.00] 그리드 탐색 → PR-AUC 최대화 |
| **3. Test 평가** | GBM-only vs Ensemble test PR-AUC 비교 → MLflow 기록 |

#### Step 1 — LSTM 캘리브레이션 (`src/model/calibrate.py`)

LSTM은 확률값이 실제 확률을 반영하지 않는 문제(calibration 불량)가 있다. 실제 피크인 바의 출력이 0.03이고 피크가 아닌 바도 0.02라면 사실상 구분이 불가능하다. `IsotonicRegression`으로 이 값들을 "실제 양성 비율"에 맞게 보정한다.

```
LSTM raw:  [0.01, 0.02, 0.03, 0.03, 0.04]  ← 다 비슷비슷, 구분 안 됨
LSTM cal:  [0.05, 0.12, 0.48, 0.51, 0.73]  ← 의미 있는 분포로 펼쳐짐
```

캘리브레이션은 **val set 예측값 vs 실제 레이블**로만 학습 → test set 누출 없음.

#### Step 2 — 최적 가중치 탐색 (`src/model/ensemble.py`)

val set에서 w_gbm을 0.50 ~ 1.00 사이로 바꿔가며 PR-AUC가 가장 높은 조합을 찾는다.

```
w_gbm=0.50 → PR-AUC 0.6991
w_gbm=0.70 → PR-AUC 0.7089  ← best
w_gbm=0.85 → PR-AUC 0.7061
w_gbm=1.00 → PR-AUC 0.7026  (GBM 단독)
```

LSTM이 GBM을 실제로 보완하면 w_gbm < 1.0, 도움이 안 되면 w_gbm = 1.0 (GBM 단독)으로 자동 선택된다.

#### Step 3 — Test set 비교 평가

캘리브레이션·가중치는 val에서만 결정하고, test set에서 최종 비교한다.

```
GBM 단독:  peak PR-AUC=0.7011, trough PR-AUC=0.7041
Ensemble:  peak PR-AUC=0.7089, trough PR-AUC=0.7118
Delta:     peak  +0.0078,       trough  +0.0077
```

결과는 MLflow `ensemble_evaluation` run에 자동 기록된다.

### 앙상블 아티팩트

```
data/models/{L}/{M}/
├── lstm_{market}_peak_calibrator.joblib    # peak IsotonicRegression
├── lstm_{market}_trough_calibrator.joblib  # trough IsotonicRegression
└── ensemble_{market}_weights.json          # 최적 가중치

# weights.json 예시
{
  "peak":   {"w_gbm": 0.75, "val_pr_auc": 0.7145},
  "trough": {"w_gbm": 0.70, "val_pr_auc": 0.7198}
}
```

### 실행

```bash
# 1. GBM + LSTM 모두 학습 필요
./optionmeme model --market us --model all --label-config L2 --model-config M3

# 2. 앙상블 (캘리브레이션 + 최적 가중치 탐색 + test 비교 평가)
./optionmeme ensemble --market us --label-config L2 --model-config M3

# 3. 앙상블 추론
./optionmeme predict --market us --symbol AAPL --model ensemble --label-config L2 --model-config M3
./optionmeme batch_predict --market us --model ensemble --label-config L2 --model-config M3
```

### 앙상블 MLflow 실험 구조

```
option-meme/us
└── ensemble_L2_M3                 ← parent run
    ├── ensemble_peak              ← nested: peak 캘리브레이션 + 가중치
    ├── ensemble_trough            ← nested: trough 캘리브레이션 + 가중치
    └── ensemble_evaluation        ← nested: GBM-only vs Ensemble test 비교
```

| Run | 기록 항목 |
|-----|----------|
| **ensemble_peak/trough** | best_w_gbm, ensemble_val_pr_auc, calibrator artifact |
| **ensemble_evaluation** | gbm_peak/trough_test_pr_auc, ensemble_peak/trough_test_pr_auc, delta |

## MLflow 실험 추적

모델 학습 시 파라미터·메트릭·아티팩트가 `data/mlruns/`에 자동 기록된다.
실험은 시장별(`option-meme/kr`, `option-meme/us`)로 분리되며, 런마다 Variant(`L1/L2 × M1~M4`) 조합이 parent run으로 기록되고 GBM·LSTM·평가가 nested run으로 기록된다.

### 실행 구조

```
option-meme/{market}                  ← experiment
└── {label_config}_{model_config}     ← parent run (예: L2_M3)
    ├── gbm_peak                      ← nested run
    ├── gbm_trough                    ← nested run
    ├── lstm_peak                     ← nested run
    ├── lstm_trough                   ← nested run
    └── evaluation                    ← nested run
```

### 기록 항목

| Run | Params | Metrics | Artifacts |
|-----|--------|---------|-----------|
| **parent** | market, label_config, model_config, prominence_pct, gbm_lookback, lstm_lookback, fill_method | — | — |
| **gbm_{label}** | LGB 하이퍼파라미터 (objective, seed 등) | pr_auc_test, positive_rate_train/test, best_iteration, train/val/test_size | 모델(.txt), params.json (Optuna 최적화 시) |
| **lstm_{label}** | hidden_size, num_layers, dropout, lr, batch_size, lookback, fill_method | pr_auc_val, pr_auc_test, best_epoch, n_features | 모델(.pt) |
| **evaluation** | — | peak/trough_pr_auc, backtest_win_rate, backtest_sharpe_approx 등 | — |

### UI 실행

```bash
mlflow ui --backend-store-uri data/mlruns
# → http://127.0.0.1:5000
```

## 모델

### LightGBM

- **방식**: 피크/트로프 각각 독립 이진 분류기
- **하이퍼파라미터**: Optuna 자동 튜닝 (num_leaves, learning_rate, feature_fraction 등)
- **불균형 처리**: `is_unbalance=True`
- **평가 지표**: PR-AUC (Average Precision)
- **출력**: `{label}_gbm.txt` (모델) + `{label}_gbm_metrics.json` (메트릭)

### LSTM+Attention

```
Input (batch, lookback, n_features)
  │
  ▼
BiLSTM (hidden=128, layers=2, bidirectional)
  │
  ▼
Attention (softmax weighted sum over timesteps)
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
- **학습**: AdamW (lr=1e-3), batch=256, epochs=50, early stopping (patience=3)
- **LR 스케줄링**: ReduceLROnPlateau (patience=2, factor=0.5)
- **디바이스**: MPS (Apple Silicon) / CUDA / CPU 자동 선택
- **출력**: `{label}_lstm.pt` (모델) + `{label}_lstm_metrics.json` (메트릭)

### 학습 설정

- **시간 기반 분할** (셔플 없음 → look-ahead bias 방지)
  - Train: 4년 / Validation: 6개월 / Test: 12개월
- **Walk-forward**: 6개월 학습 + 1개월 테스트 롤링 윈도우
- 피크(label=1)와 트로프(label=2) 각각 독립 모델 학습

### 피처 카테고리 (45 base features)

| 카테고리 | prefix | 피처 예시 | 수 |
|---|---|---|---|
| **가격** | `pf_` | return_1m~10m, momentum, acceleration, candle ratios | ~15 |
| **기술적** | `tf_` | MA(5/10/20), RSI(7/14), MACD, BB position/width | ~15 |
| **거래량** | `vf_` | volume_ratio, OBV_change, vol_ma_ratio | ~5 |
| **시간** | `tmf_` | elapsed_norm, minute_sin/cos, day_of_week_sin/cos | ~5 |
| **메타** | `mf_` | symbol_id (종목 인코딩) | ~5 |

Lookback 적용 시 각 base feature에 대해 lag1~lagN 파생 (M1/M2: 450 lag, M3: 225 lag, M4: 0 lag).

## 트레이딩 엔진

### 매매 전략

피크/트로프 예측 신호를 풋옵션 매매로 연결:

| 시그널 | 행동 | 근거 |
|---|---|---|
| **PEAK** 감지 + 미보유 | ATM 풋옵션 매수 | 주가 하락 예상 → 풋 가치 상승 |
| **TROUGH** 감지 + 보유 중 | 풋옵션 전량 매도 | 주가 반등 → 풋 가치 하락 전 청산 |

### 리스크 관리

| 규칙 | 조건 | 행동 |
|---|---|---|
| 이익실현 | PnL ≥ +10% | 풋 전량 매도 |
| 손절매 | PnL ≤ -5% | 풋 전량 매도 |
| 강제청산 | 장마감 120분 전 | 풋 전량 매도 |

### Black-Scholes 프라이싱

모의 브로커에서 풋옵션 가격 계산:
- 변동성: 25% (연환산)
- 무위험이자율: 3.5%
- 슬리피지: 0.5%
- 최소 잔존일: 7일

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
| `TRAIN_YEARS` | 4 | 학습 데이터 기간 |
| `VAL_MONTHS` / `TEST_MONTHS` | 6 / 12 | 검증/테스트 기간 |
| `LGB_PARAMS.is_unbalance` | True | 클래스 불균형 자동 처리 |
| `LSTM_EPOCHS` | 50 | 최대 에포크 (early stopping 적용) |
| `TRADE_PROFIT_TARGET_PCT` | +10% | 이익실현 기준 |
| `TRADE_STOP_LOSS_PCT` | -5% | 손절매 기준 |

### 변형 설정 (`config/variants.py`)

Labeling과 Model 파라미터의 조합은 `config/variants.py`의 `LABEL_CONFIGS`와 `MODEL_CONFIGS`에서 관리된다.
`--label-config`와 `--model-config` CLI 인자로 선택하며, `all`을 지정하면 전체 조합을 순회한다.

## 한국투자증권 API 스크립트

`scripts/` 디렉토리에는 한투 OpenAPI를 직접 호출해 응답 구조와 종목코드 체계를 확인하기 위한 테스트 스크립트가 포함되어 있다.
자세한 내용은 [`scripts/README.md`](scripts/README.md) 참조.

| 스크립트 | 기능 |
|---|---|
| `kis_auth.py` | OAuth2 인증 & 토큰 캐싱 |
| `kis_to_occ.py` | 한투 ↔ OCC 옵션 종목코드 변환 |
| `test_kis_kr_stock.py` | 국내주식 분봉조회 |
| `test_kis_option_board.py` | 옵션 전광판 (콜/풋 체인) |
| `test_kis_fuopt_codes.py` | 국내 선물옵션 마스터 파일 |
| `test_kis_fuopt_chart.py` | 국내 선물옵션 분봉 |
| `test_kis_overseas_chart.py` | 해외주식 분봉 |
| `test_kis_overseas_fuopt_codes.py` | 해외 선물옵션 마스터 파일 |
| `test_kis_overseas_fuopt_chart.py` | 해외선물 분봉 |
| `test_kis_overseas_fuopt_detail.py` | 해외옵션 종목상세 |
| `test_kis_overseas_fuopt_orderable.py` | 해외선물옵션 주문가능 조회 |
| `test_kis_overseas_fuopt_ws*.py` | 해외선물옵션 실시간 WebSocket (체결/호가/주문/체결통보) |
| `test_databento_opra.py` | Databento OPRA 옵션 데이터 조회 & 비용 확인 |
| `test_databento_stock.py` | Databento 주식 데이터 비용 확인 |
| `collect_opra_puts.py` | OPRA 풋옵션 심볼 탐색 & 비용 산정 |
| `download_stock_ohlcv.py` | Databento DBN 일괄 다운로드 (20종목 × 3거래소) |
| `build_us_ohlcv.py` | US 통합 ETL (Databento + yfinance + tvDatafeed) |

## 라이선스

MIT
