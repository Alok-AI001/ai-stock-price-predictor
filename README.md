# AI Stock Price Predictor

An end-to-end stock forecasting and backtesting project written in **pure Python (3.8+)**.
It downloads daily prices, engineers stationary technical-indicator features, trains
Gradient Boosted Trees, a Random Forest and Ridge Regression, blends them into an ensemble,
runs an out-of-sample backtest against buy & hold, and projects future prices with a
95% confidence cone. Results are available from a terminal, a web dashboard, or a
standalone HTML report.

No `pip install` is needed: every algorithm (trees, boosting, ridge, indicators,
charts, web server) is implemented with the standard library.

> **Disclaimer:** educational project, not financial advice. Daily stock returns are
> extremely noisy. Expect directional accuracy near 50% and compare every model with the
> naive baseline ("tomorrow = today") that the tool prints for you.

## Quick start()

```bash
cd ai-stock-price-predictor

# 1. Web dashboard -> open http://127.0.0.1:8000
python app.py

# 2. Command line + HTML report (written to reports/prediction_report.html)
python main.py --ticker AAPL --period 1y --model ensemble --forecast 7

# 3. Tests
python test_predictor.py
```

## Project structure

```
ai-stock-price-predictor/
├── README.md
├── requirements.txt          # intentionally empty (stdlib only)
├── .gitignore
├── main.py                   # CLI entry point
├── app.py                    # web dashboard + JSON API (http.server)
├── test_predictor.py         # unit and end-to-end tests
├── static/
│   └── index.html            # dashboard front-end (form + report viewer)
├── stockpredictor/
│   ├── __init__.py
│   ├── data.py               # Yahoo Finance v8 fetch, disk cache, GBM fallback
│   ├── features.py           # SMA/EMA, RSI, MACD, Bollinger, ATR, momentum, lags
│   ├── models.py             # GBDT, Random Forest, Ridge, blended Ensemble
│   ├── metrics.py            # MAE, RMSE, MAPE, directional accuracy, Sharpe, MDD
│   ├── backtest.py           # long/cash strategy vs buy & hold ($10k)
│   ├── forecast.py           # autoregressive multi-step forecast + 95% cone
│   ├── pipeline.py           # orchestration (run_pipeline)
│   └── report.py             # standalone HTML report with inline SVG charts
├── data_cache/               # cached downloads (*.json), created automatically
└── reports/                  # generated HTML reports
```

## How it works

```
Yahoo Finance -> disk cache -> OHLCV bars
   -> 21 stationary features (ratios, oscillators, volatility, momentum, lags)
   -> target r(t+1) = (P(t+1) - P(t)) / P(t)
   -> chronological 80/20 split (no shuffling)
   -> GBDT | Random Forest | Ridge | Ensemble (65% trees + 35% ridge)
   -> price reconstruction  P_hat(t+1) = P(t) * (1 + r_hat)
   -> metrics, backtest, 7-30 day forecast
   -> terminal / dashboard / HTML report
```

**Why returns instead of prices?** Raw prices are non-stationary (a random walk, I(1)).
Modelling the next-day return keeps the target on a stable scale, and the price is
rebuilt analytically from today's close, so no future information leaks in.

**Features** (all computed only from data available at the close of day *t*):
price/SMA ratios (10, 20, 50), price/EMA ratios (12, 26), RSI(14), MACD line, signal and
histogram (normalised by price), Bollinger %B and bandwidth, ATR(14) as % of price,
1/3/5/10-day momentum, three lagged returns, volume ratio, and the daily high-low range.

**Models**

| Model | Notes |
| :--- | :--- |
| Gradient boosting | 120 depth-3 trees, learning rate 0.07, 80% row subsampling, histogram split search |
| Random forest | 60 depth-5 bootstrap trees, random feature subsets at each split |
| Ridge | z-scored features, L2 penalty lambda = 0.5, solved in closed form |
| Ensemble | 0.65 x mean(GBDT, RF) + 0.35 x Ridge |

**Backtest:** start with $10,000. Go long for a day when the predicted return exceeds
0.1% (10 bps); otherwise stay in cash. The benchmark holds the asset over the same
out-of-sample dates. Reported: total return, alpha, annualised Sharpe, max drawdown,
exposure and win rate. No transaction costs or slippage are modelled.

**Forecast:** each predicted close is appended as a synthetic bar and fed back into the
feature builder. The cone is `P_hat * (1 +/- 1.96 * sigma * sqrt(h))`, where sigma is the
standard deviation of out-of-sample return residuals. Each daily step is clamped to +/-5%.

## Command-line options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--ticker` | `AAPL` | Yahoo Finance symbol (e.g. `MSFT`, `TCS.NS`, `RELIANCE.NS`) |
| `--period` | `1y` | `6mo`, `1y`, `2y`, `5y` |
| `--model` | `ensemble` | `gradient_boosting`, `random_forest`, `ridge`, `ensemble` |
| `--forecast` | `7` | Days to forecast (1-30) |
| `--output` | `reports/prediction_report.html` | HTML report path |
| `--no-html` | off | Skip the HTML report |
| `--offline` | off | Never use the network |
| `--refresh` | off | Ignore cached data and re-download |

`python app.py --port 9000 --host 0.0.0.0` changes the dashboard address.

## Web API

| Endpoint | Returns |
| :--- | :--- |
| `GET /` | Dashboard |
| `GET /report?ticker=AAPL&period=1y&model=ensemble&forecast=7` | HTML report |
| `GET /api/predict?...` (same parameters) | Full JSON result |
| `GET /api/health` | `{"status": "ok"}` |

Use the API from Python:

```python
from stockpredictor import run_pipeline
result = run_pipeline("MSFT", period="2y", model="gradient_boosting", forecast_days=10)
print(result["next_day"], result["backtest"]["stats"]["alpha_pct"])
```

## Data sources and offline behaviour

1. A fresh cache file in `data_cache/` (under 12 hours old) is used first.
2. Otherwise data is requested from the Yahoo Finance v8 chart endpoint.
3. If the request fails, a stale cache is used when available.
4. As a last resort a deterministic **Geometric Brownian Motion** series is generated so
   the project always runs. Reports show a notice when this happens: synthetic data is
   for demonstration only.

## Extending the project

- Add a model: implement `fit(X, y)` and `predict(X)` in `models.py`, then register it in
  `train_all` and `MODEL_NAMES`.
- Add a feature: append to `FEATURE_NAMES` and the row in `features.build_features`.
  Keep it causal (use only data up to day *t*); `test_no_lookahead` guards this.
- Swap in `scikit-learn` or `pandas` for speed if you scale up; the pipeline only needs
  models exposing `fit` and `predict`.

## Known limitations

- A single chronological split is used, not rolling re-training.
- No transaction costs, slippage, taxes or position sizing in the backtest.
- Features are price/volume only (no fundamentals or news).
- Pure-Python models favour clarity over speed; a 5-year run takes about a second.
