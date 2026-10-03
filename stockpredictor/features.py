"""Stationary technical-indicator feature matrix (all indicators are causal)."""
import math

WARMUP = 50

FEATURE_NAMES = [
    "sma_ratio_10", "sma_ratio_20", "sma_ratio_50", "ema_ratio_12", "ema_ratio_26",
    "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct_b", "bb_width", "atr_pct",
    "mom_1", "mom_3", "mom_5", "mom_10", "lag_ret_1", "lag_ret_2", "lag_ret_3",
    "volume_ratio", "hl_range",
]


def sma(x, n):
    out, s = [None] * len(x), 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(x, n):
    a = 2.0 / (n + 1)
    out = [x[0]]
    for v in x[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def rolling_std(x, n):
    out = [None] * len(x)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((v - m) ** 2 for v in w) / n)
    return out


def rsi(close, n=14):
    gains = [0.0] + [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [0.0] + [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    out = []
    for g, l in zip(ema(gains, n), ema(losses, n)):
        if l == 0:
            out.append(100.0 if g > 0 else 50.0)
        else:
            out.append(100.0 - 100.0 / (1.0 + g / l))
    return out


def build_features(bars):
    """Return {"names", "X", "indices"}: one row per bar index >= WARMUP.

    Row i only uses information available at the close of bar i.
    """
    c = [b["close"] for b in bars]
    h = [b["high"] for b in bars]
    l = [b["low"] for b in bars]
    v = [b["volume"] for b in bars]
    n = len(c)
    if n <= WARMUP + 1:
        raise ValueError("need more than %d bars" % (WARMUP + 1))

    s10, s20, s50 = sma(c, 10), sma(c, 20), sma(c, 50)
    e12, e26 = ema(c, 12), ema(c, 26)
    macd = [a - b for a, b in zip(e12, e26)]
    sig = ema(macd, 9)
    sd20 = rolling_std(c, 20)
    rs = rsi(c, 14)
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
                          for i in range(1, n)]
    atr = ema(tr, 14)
    vs20 = sma(v, 20)
    ret1 = [0.0] + [c[i] / c[i - 1] - 1.0 for i in range(1, n)]

    X, idx = [], []
    for i in range(WARMUP, n):
        upper, lower = s20[i] + 2 * sd20[i], s20[i] - 2 * sd20[i]
        band = upper - lower
        X.append([
            c[i] / s10[i] - 1, c[i] / s20[i] - 1, c[i] / s50[i] - 1,
            c[i] / e12[i] - 1, c[i] / e26[i] - 1,
            rs[i] / 100.0,
            macd[i] / c[i], sig[i] / c[i], (macd[i] - sig[i]) / c[i],
            (c[i] - lower) / band if band > 0 else 0.5,
            band / s20[i],
            atr[i] / c[i],
            c[i] / c[i - 1] - 1, c[i] / c[i - 3] - 1, c[i] / c[i - 5] - 1, c[i] / c[i - 10] - 1,
            ret1[i - 1], ret1[i - 2], ret1[i - 3],
            (v[i] / vs20[i] - 1.0) if vs20[i] else 0.0,
            (h[i] - l[i]) / c[i],
        ])
        idx.append(i)
    return {"names": FEATURE_NAMES, "X": X, "indices": idx}
