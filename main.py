#!/usr/bin/env python3
"""Command-line interface: run the full pipeline and export an HTML report."""
import argparse
import os
import sys

from stockpredictor.data import PERIOD_DAYS
from stockpredictor.models import MODEL_NAMES
from stockpredictor.pipeline import run_pipeline
from stockpredictor.report import render_report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI Stock Price Predictor")
    p.add_argument("--ticker", default="AAPL", help="symbol, e.g. AAPL, MSFT, TCS.NS")
    p.add_argument("--period", default="1y", choices=list(PERIOD_DAYS))
    p.add_argument("--model", default="ensemble", choices=MODEL_NAMES)
    p.add_argument("--forecast", type=int, default=7, help="days ahead (1-30)")
    p.add_argument("--output", default=os.path.join("reports", "prediction_report.html"))
    p.add_argument("--no-html", action="store_true", help="skip the HTML report")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--refresh", action="store_true", help="ignore the disk cache")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        r = run_pipeline(a.ticker, a.period, a.model, a.forecast, a.refresh, a.offline)
    except ValueError as exc:
        print("error:", exc, file=sys.stderr)
        return 2
    st, nd = r["backtest"]["stats"], r["next_day"]
    print("\n%s | model=%s | %d bars from %s" % (r["ticker"], r["model"], r["bars"], r["data_source"]))
    if r["data_source"] == "synthetic":
        print("note: live data unavailable, using simulated prices (demo only)")
    print("-" * 78)
    print("%-18s %8s %8s %8s %9s %9s %8s" % ("model", "MAE", "RMSE", "MAPE%", "Direct.%", "Return%", "Sharpe"))
    for c in r["comparison"]:
        mark = "*" if c["model"] == r["model"] else " "
        print("%s%-17s %8.3f %8.3f %8.2f %9.1f %+9.1f %8.2f" % (
            mark, c["model"], c["mae"], c["rmse"], c["mape"], c["directional_accuracy"],
            c["strategy_return_pct"], c["sharpe"]))
    b = r["baseline"]
    print(" %-17s %8.3f %8.3f %8.2f   (naive: tomorrow = today)" % ("baseline", b["mae"], b["rmse"], b["mape"]))
    print("-" * 78)
    print("Backtest $10,000 -> $%s | strategy %+.1f%% vs buy&hold %+.1f%% | alpha %+.1f%% | maxDD %.1f%%" % (
        "{:,.0f}".format(st["final_value"]), st["strategy_return_pct"], st["benchmark_return_pct"],
        st["alpha_pct"], st["max_drawdown_pct"]))
    print("Next-day: %s, predicted close %.2f (%+.2f%%)\n" % (nd["signal"], nd["predicted_close"], nd["predicted_return_pct"]))
    print("%-12s %10s %10s %10s" % ("forecast", "price", "95% low", "95% high"))
    f = r["forecast"]
    for d, p, lo, hi in zip(f["dates"], f["price"], f["lower"], f["upper"]):
        print("%-12s %10.2f %10.2f %10.2f" % (d, p, lo, hi))
    if not a.no_html:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        with open(a.output, "w", encoding="utf-8") as fh:
            fh.write(render_report(r))
        print("\nHTML report written to", a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
