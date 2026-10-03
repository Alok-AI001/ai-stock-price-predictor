"""Market data ingestion: Yahoo Finance v8 -> disk cache -> GBM fallback."""
import json
import math
import os
import random
import time
import zlib
import urllib.request
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data_cache")
CACHE_TTL_SECONDS = 12 * 3600
PERIOD_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
MIN_BARS = 100
YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}"
             "?range={r}&interval=1d")


def _cache_path(ticker, period):
    safe = "".join(ch for ch in ticker.upper() if ch.isalnum() or ch in "-_.")
    return os.path.join(CACHE_DIR, "%s_%s.json" % (safe, period))


def fetch_yahoo(ticker, period, timeout=10):
    """Download daily OHLCV bars straight from the Yahoo Finance v8 endpoint."""
    req = urllib.request.Request(YAHOO_URL.format(t=ticker, r=period),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    bars = []
    for i, ts in enumerate(result["timestamp"]):
        o, h, l, c = (quote[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, l, c):
            continue
        bars.append({
            "date": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"),
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(quote["volume"][i] or 0),
        })
    if len(bars) < MIN_BARS:
        raise ValueError("Yahoo returned only %d bars" % len(bars))
    return bars


def synthetic_bars(ticker, n):
    """Deterministic Geometric Brownian Motion series (per-ticker seed)."""
    rng = random.Random(zlib.crc32(ticker.upper().encode("utf-8")))
    price = 50.0 + rng.random() * 150.0
    mu, sigma = 0.0004, 0.012 + rng.random() * 0.01
    days, cur = [], date.today()
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    days.reverse()
    bars = []
    for d in days:
        r = rng.gauss(mu - 0.5 * sigma ** 2, sigma)
        o, c = price, price * math.exp(r)
        h = max(o, c) * (1 + abs(rng.gauss(0, sigma / 2)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sigma / 2)))
        v = 1e6 * (1 + rng.random()) * (1 + abs(r) * 20)
        bars.append({"date": d.isoformat(), "open": o, "high": h, "low": l,
                     "close": c, "volume": v})
        price = c
    return bars


def _read_cache(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["bars"]


def load_bars(ticker, period="1y", refresh=False, offline=False):
    """Return (bars, source): yahoo | cache | cache (stale) | synthetic."""
    if period not in PERIOD_DAYS:
        raise ValueError("period must be one of %s" % ", ".join(PERIOD_DAYS))
    path = _cache_path(ticker, period)
    cached = os.path.exists(path)
    fresh = cached and (time.time() - os.path.getmtime(path)) < CACHE_TTL_SECONDS
    if cached and fresh and not refresh:
        return _read_cache(path), "cache"
    if not offline:
        try:
            bars = fetch_yahoo(ticker, period)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"ticker": ticker.upper(), "bars": bars}, fh)
            return bars, "yahoo"
        except Exception:
            pass
    if cached:
        return _read_cache(path), "cache (stale)"
    return synthetic_bars(ticker, PERIOD_DAYS[period]), "synthetic"
