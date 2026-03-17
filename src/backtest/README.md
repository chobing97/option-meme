## 백테스트 전체 흐름 (상세)

---

### 1. CLI 진입

```
./optionmeme trade --market us --broker historical --symbol SPY --date-from 2025-03-11 --date-to 2026-02-27
```

**파싱 결과** ([run_pipeline.py](src/run_pipeline.py)):
- `market="us"`, `broker_type="historical"`, `symbols=["SPY"]`
- `model_type="gbm"` (기본값), `label_config="L2"`, `model_config="M3"`
- `threshold=0.5`, `quantity=1`, `capital=10,000,000`

---

### 2. 브로커 초기화 (HistoricalBroker)

**읽는 파일:**
```
data/raw/options/us/SPY/contracts.parquet   → 옵션 계약 메타 (symbol, expiry, strike, cp)
data/raw/options/us/SPY/2024.parquet        → 옵션 1분봉 OHLCV
data/raw/options/us/SPY/2025.parquet
data/raw/options/us/SPY/2026.parquet
```

```python
broker = HistoricalBroker(market="us", capital=10_000_000, slippage_pct=0.005)
broker.load_symbols(["SPY"])
# → _contracts["SPY"] = contracts DataFrame
# → _ohlcv["SPY"] = 전체 연도 OHLCV concat (symbol, datetime, OHLCV, volume)
```

---

### 3. 날짜 결정 & DataFeed 생성

**읽는 파일:**
```
data/raw/stock/us/SPY/2025.parquet   → 주식 1분봉 OHLCV
data/raw/stock/us/SPY/2026.parquet
```

```python
# Multi-day: 모든 거래일의 교집합 계산
early_df = extract_early_session(load_bars("us", "SPY"), "us")
# → 09:30~10:30 ET 60분만 추출, minutes_from_open 컬럼 추가
replay_dates = ["2025-03-11", "2025-03-12", ..., "2026-02-27"]
```

---

### 4. 일별 루프 시작

**각 거래일마다:**

```python
feed = HistoricalDataFeed(market="us", symbol="SPY", date="2025-03-11", early_df=early_df)
feed.connect()
# → _history = 직전 5거래일 × 60바 = ~300바 (BarAccumulator 초기화용)
# → _queue = 당일 60바 (deque, FIFO로 하나씩 pop)
```

---

### 5. 바별 메인 루프 (60회/일)

**각 바마다 순서대로 실행:**

#### 5-1. 바 가져오기
```python
bar = feed.get_latest_bar()  # deque.popleft()
# bar: datetime, open, high, low, close, volume, minutes_from_open
```

#### 5-2. BarAccumulator에 추가
```python
accumulators["SPY"].add_bar(bar)
# 내부: _today_bars 리스트에 append
```

#### 5-3. 브로커 시간 & 가격 갱신 + Mark-to-Market
```python
broker.update_underlying_price("SPY", close_price=280.50, dt=09:31)
```
내부에서:
```python
self._current_time = 09:31
self._mark_positions("SPY")
  └─ 보유 포지션이 있으면:
     quote = get_option_quote(pos.contract)
       └─ _ohlcv["SPY"]에서 해당 옵션 심볼 + datetime <= 09:31 로 asof join
       └─ mid = bar["close"], spread = (high - low) / 2
       └─ OptionQuote(bid=mid-spread, ask=mid+spread, last=mid, volume=bar["volume"])
     pos.update_mark(quote.last)
       └─ current_price = quote.last
       └─ unrealized_pnl_pct = (current_price - avg_entry_price) / avg_entry_price
```

#### 5-4. 시그널 감지 (Feature Pipeline + 모델 추론)

**읽는 파일 (최초 1회, lazy load):**
```
data/models/L2/M3/lgb_us_peak.txt      → LightGBM peak 모델
data/models/L2/M3/lgb_us_trough.txt    → LightGBM trough 모델
```

```python
signal = detector.detect(accumulators["SPY"])
```

내부:
```python
# 1) 피처 DataFrame 구성
df = concat([_history(300바), _today_bars(현재까지)])  # ~301바

# 2) 45개 base feature 계산
build_features(df)
# pf_*: return_1m~10m, momentum, acceleration, candle ratios (~15)
# tf_*: MA(5/10/20), RSI(7/14), MACD, BB position/width (~15)
# vf_*: volume_ratio, OBV_change, vol_ma_ratio (~5)
# tmf_*: elapsed_norm, minute_sin/cos, day_of_week_sin/cos (~5)
# mf_*: symbol_id (~5)

# 3) lookback lag feature 생성 (M3: lookback=5)
build_lookback_features(df)
# 각 base feature × lag1~lag5 = 45 × 5 = 225 lag features
# 총: 45 + 225 = 270 features

# 4) 마지막 행만 추론
X = df.iloc[[-1]][feature_cols].values  # shape (1, 270)
peak_prob = peak_model.predict(X)[0]      # e.g., 0.72
trough_prob = trough_model.predict(X)[0]  # e.g., 0.15

# 5) 시그널 결정 (threshold=0.5)
if peak_prob >= 0.5 and peak_prob > trough_prob → PEAK
elif trough_prob >= 0.5 and trough_prob > peak_prob → TROUGH
else → NONE
```

#### 5-5. 매매 규칙 (우선순위 순)

```
1순위: 장마감 120분 전 (14:00 ET) → 보유 중이면 강제 청산
2순위: 손절매 (unrealized_pnl_pct <= -5%) → SELL_PUT
3순위: 익절매 (unrealized_pnl_pct >= +10%) → SELL_PUT
4순위: TROUGH 시그널 + 풋 보유 중 → SELL_PUT
5순위: PEAK 시그널 + 풋 미보유 → BUY_PUT
```

#### 5-6. BUY_PUT 실행 (PEAK 시그널)

```python
# ATM 풋 계약 선택
chain = broker.get_option_chain("SPY", "put")
# → contracts에서 period_start <= now < expiry 필터
atm = min(chain, key=lambda c: abs(c.strike - 280.50))
# → strike=280 선택

# 주문 제출
order = Order(side=BUY, contract=atm, quantity=1)
filled = broker.submit_order(order)
```

내부:
```python
quote = get_option_quote(atm)
# → _ohlcv["SPY"]에서 해당 옵션 OCC 심볼로 asof join
# → mid=3.50, spread=0.20 → ask=3.60

if quote.volume == 0 → REJECTED (유동성 없음)

fill_price = 3.60 × 1.005 = 3.618  # ask + 0.5% slippage
total_cost = 3.618 × 1 × 100 = 361.80  # 1계약 × 100주

cash -= 361.80  # 10,000,000 → 9,999,638.20
_positions.append(Position(contract=atm, qty=1, avg_entry=3.618, current_price=3.50))
```

#### 5-7. SELL_PUT 실행 (TROUGH/손절/익절/강제청산)

```python
quote = get_option_quote(pos.contract)
# → mid=4.20, spread=0.15 → bid=4.05

fill_price = 4.05 × 0.995 = 4.030  # bid - 0.5% slippage
proceeds = 4.030 × 1 × 100 = 403.00

cash += 403.00  # 9,999,638.20 → 10,000,041.20
_positions.remove(pos)
```

#### 5-8. 바 스냅샷 기록

```python
tracker.record_bar(
    timestamp=09:31, symbol="SPY",
    underlying_close=280.50,
    signal="PEAK", peak_prob=0.72, trough_prob=0.15,
    action="BUY_PUT", reason="PEAK_SIGNAL",
    strike=280, fill_price=3.618,
    position_qty=1, position_avg_entry=3.618,
    position_mark_price=3.50,  # 브로커의 현재 mark 가격
    cash=9_999_638.20,
)
```

내부 계산:
```python
position_value = 1 × 3.50 × 100 = 350.00
equity = 9_999_638.20 + 350.00 = 9_999,988.20
equity_high = max(이전_고점, 9_999_988.20)
drawdown_pct = (equity - equity_high) / equity_high
```

---

### 6. 세션 종료 & 저장

**출력 파일:**
```
data/trading/backtests/2025-03-11_2026-02-27.parquet
```

**스키마 (매 분봉 1행):**

| 컬럼 | 타입 | 설명 |
|---|---|---|
| timestamp | datetime | 바 시각 |
| symbol | str | 종목 |
| underlying_close | float | 주가 종가 |
| signal | str | PEAK / TROUGH / NONE |
| peak_prob | float | 피크 확률 |
| trough_prob | float | 트로프 확률 |
| action | str | BUY_PUT / SELL_PUT / "" |
| reason | str | PEAK_SIGNAL / TROUGH_SIGNAL / STOP_LOSS / PROFIT_TARGET / FORCE_CLOSE / "" |
| strike | float | 행사가 |
| fill_price | float | 체결가 |
| position_qty | int | 보유 계약 수 |
| position_avg_entry | float | 평균 진입가 |
| position_mark_price | float | 현재 옵션 시장가 (asof join) |
| position_value | float | qty × mark_price × 100 |
| cash | float | 현금 잔고 |
| equity | float | cash + position_value |
| equity_high | float | 에쿼티 고점 (러닝 맥스) |
| drawdown_pct | float | (equity - equity_high) / equity_high |

---

### 전체 데이터 흐름 요약

```
data/raw/stock/us/SPY/{year}.parquet ─────┐
                                           ├─→ DataFeed → 바 60개/일 → BarAccumulator
                                           │                              │
data/models/L2/M3/lgb_us_{peak,trough}.txt ┤                    build_features()
                                           │                    build_lookback_features()
                                           │                    predict() → Signal
                                           │                              │
data/raw/options/us/SPY/contracts.parquet ──┤                    Trading Rules
data/raw/options/us/SPY/{year}.parquet ─────┤                    (PEAK→BUY, TROUGH→SELL)
                                           │                              │
                                           ├─→ HistoricalBroker ← submit_order()
                                           │   get_option_quote() ← asof join on OHLCV
                                           │   mark_positions() ← 매 바마다
                                           │                              │
                                           └─→ TradeTracker.record_bar()
                                               equity = cash + qty × mark × 100
                                                              │
                                               data/trading/backtests/{dates}.parquet
```
