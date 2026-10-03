"""End-to-end orchestration: data -> features -> models -> evaluation -> forecast."""
from datetime import datetime

from . import metrics as M
from .backtest import SIGNAL_THRESHOLD, run_backtest
from .data import load_bars
from .features import build_features
from .forecast import forecast_future
from .models import MODEL_NAMES, train_all

TRAIN_FRACTION = 0.8
HISTORY_TAIL = 120


def run_pipeline(ticker="AAPL", period="1y", model="ensemble", forecast_days=7,
                 refresh=False, offline=False, seed=42):
    ticker = ticker.upper().strip()
    if model not in MODEL_NAMES:
        raise ValueError("model must be one of %s" % ", ".join(MODEL_NAMES))
    if not 1 <= forecast_days <= 30:
        raise ValueError("forecast_days must be between 1 and 30")

    bars, source = load_bars(ticker, period, refresh=refresh, offline=offline)
    closes = [b["close"] for b in bars]
    dates = [b["date"] for b in bars]
    n = len(bars)

    feats = build_features(bars)
    X_all, idx_all = feats["X"], feats["indices"]

    # Labelled rows: features at t, target = r_{t+1} = (P_{t+1} - P_t) / P_t
    labelled = [(x, i) for x, i in zip(X_all, idx_all) if i < n - 1]
    X = [x for x, _ in labelled]
    I = [i for _, i in labelled]
    y = [closes[i + 1] / closes[i] - 1.0 for i in I]

    split = int(len(X) * TRAIN_FRACTION)               # chronological, no shuffling
    X_tr, y_tr, X_te, y_te, I_te = X[:split], y[:split], X[split:], y[split:], I[split:]

    models = train_all(X_tr, y_tr, seed=seed)

    p_now = [closes[i] for i in I_te]
    p_next = [closes[i + 1] for i in I_te]
    test_dates = [dates[i + 1] for i in I_te]
    start_date = dates[I_te[0]]

    comparison, preds, backtests = [], {}, {}
    for name in MODEL_NAMES:
        r_hat = models[name].predict(X_te)
        p_hat = [p * (1 + r) for p, r in zip(p_now, r_hat)]     # P_hat = P_t (1 + r_hat)
        bt = run_backtest(start_date, test_dates, r_hat, y_te)
        preds[name], backtests[name] = p_hat, bt
        comparison.append({
            "model": name,
            "mae": M.mae(p_next, p_hat), "rmse": M.rmse(p_next, p_hat),
            "mape": M.mape(p_next, p_hat),
            "directional_accuracy": M.directional_accuracy(y_te, r_hat),
            **bt["stats"],
        })
    baseline = {"mae": M.mae(p_next, p_now), "rmse": M.rmse(p_next, p_now),
                "mape": M.mape(p_next, p_now)}

    chosen = models[model]
    r_hat = chosen.predict(X_te)
    sigma = M.std([a - p for a, p in zip(y_te, r_hat)])
    fc = forecast_future(bars, chosen, forecast_days, sigma)

    next_r = chosen.predict([X_all[-1]])[0]
    signal = "LONG" if next_r > SIGNAL_THRESHOLD else "CASH"

    tail = min(n, HISTORY_TAIL)
    return _round({
        "ticker": ticker, "period": period, "model": model, "data_source": source,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bars": n, "train_rows": len(X_tr), "test_rows": len(X_te),
        "last_close": closes[-1], "last_date": dates[-1],
        "next_day": {"predicted_return_pct": 100 * next_r,
                     "predicted_close": closes[-1] * (1 + next_r), "signal": signal},
        "residual_sigma": sigma,
        "history": {"dates": dates[-tail:], "close": closes[-tail:]},
        "test": {"dates": test_dates, "actual": p_next, "predicted": preds[model]},
        "comparison": comparison, "baseline": baseline,
        "backtest": backtests[model],
        "forecast": fc,
    })


def _round(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj
