"""Forecast accuracy and trading performance metrics."""
import math


def mae(actual, pred):
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)


def rmse(actual, pred):
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual))


def mape(actual, pred):
    return 100.0 * sum(abs((a - p) / a) for a, p in zip(actual, pred)) / len(actual)


def directional_accuracy(actual_ret, pred_ret):
    """% of days where the predicted return had the same sign as the realised one."""
    hits = sum(1 for a, p in zip(actual_ret, pred_ret) if (a > 0) == (p > 0))
    return 100.0 * hits / len(actual_ret)


def std(x):
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def sharpe(daily_returns):
    s = std(daily_returns)
    return math.sqrt(252) * (sum(daily_returns) / len(daily_returns)) / s if s > 0 else 0.0


def max_drawdown(equity):
    peak, worst = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return 100.0 * worst
