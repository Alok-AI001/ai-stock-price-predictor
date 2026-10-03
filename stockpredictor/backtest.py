"""Walk-forward long/cash strategy simulation vs. passive buy & hold."""
from .metrics import max_drawdown, sharpe

INITIAL_CASH = 10_000.0
SIGNAL_THRESHOLD = 0.001  # 10 bps


def run_backtest(start_date, dates, pred_returns, actual_returns,
                 initial=INITIAL_CASH, threshold=SIGNAL_THRESHOLD):
    """Go long for a day when the predicted return exceeds the threshold, else hold cash."""
    strat, bench, daily = [initial], [initial], []
    longs = wins = 0
    for p, a in zip(pred_returns, actual_returns):
        r = a if p > threshold else 0.0
        if p > threshold:
            longs += 1
            wins += a > 0
        daily.append(r)
        strat.append(strat[-1] * (1 + r))
        bench.append(bench[-1] * (1 + a))
    s_ret = 100.0 * (strat[-1] / initial - 1)
    b_ret = 100.0 * (bench[-1] / initial - 1)
    return {
        "dates": [start_date] + list(dates),
        "strategy": strat,
        "benchmark": bench,
        "stats": {
            "strategy_return_pct": s_ret,
            "benchmark_return_pct": b_ret,
            "alpha_pct": s_ret - b_ret,
            "sharpe": sharpe(daily),
            "benchmark_sharpe": sharpe(actual_returns),
            "max_drawdown_pct": max_drawdown(strat),
            "benchmark_max_drawdown_pct": max_drawdown(bench),
            "long_days": longs,
            "win_rate_pct": 100.0 * wins / longs if longs else 0.0,
            "exposure_pct": 100.0 * longs / len(pred_returns),
            "final_value": strat[-1],
        },
    }
