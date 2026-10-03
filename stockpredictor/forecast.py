"""Autoregressive multi-step forecasting with a sqrt(h) 95% confidence cone."""
import math
from datetime import date, timedelta

from .features import build_features

Z95 = 1.96
MAX_DAILY_MOVE = 0.05  # clamp each predicted step to +/-5%


def next_business_days(last_date, count):
    d = date.fromisoformat(last_date)
    out = []
    while len(out) < count:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return out


def forecast_future(bars, model, horizon, sigma):
    """Predict `horizon` closes ahead; each step feeds back into the feature builder."""
    work = list(bars)
    days = next_business_days(bars[-1]["date"], horizon)
    out = {"dates": days, "price": [], "upper": [], "lower": []}
    for h, day in enumerate(days, start=1):
        row = build_features(work)["X"][-1]
        r = max(-MAX_DAILY_MOVE, min(MAX_DAILY_MOVE, model.predict([row])[0]))
        last = work[-1]["close"]
        price = last * (1 + r)
        band = Z95 * sigma * math.sqrt(h)
        out["price"].append(price)
        out["upper"].append(price * (1 + band))
        out["lower"].append(price * (1 - band))
        work.append({"date": day, "open": last, "high": max(last, price),
                     "low": min(last, price), "close": price,
                     "volume": work[-1]["volume"]})
    return out
