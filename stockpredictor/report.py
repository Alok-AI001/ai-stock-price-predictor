"""Standalone HTML report with inline SVG charts (no external assets)."""
from html import escape

INK, MUTED, GRID = "#1d2433", "#6b7385", "#e3e6ec"
BLUE, AMBER, GREEN, RED = "#2457c5", "#d98a00", "#1f8f5f", "#c23a3a"


def svg_chart(labels, series, band=None, w=860, h=300):
    ml, mr, mt, mb = 60, 14, 14, 26
    vals = [v for s in series for v in s["values"] if v is not None]
    if band:
        vals += [v for v in band[0] + band[1] if v is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo or 1.0) * 0.06
    lo, hi = lo - pad, hi + pad
    n = len(labels)

    def X(i):
        return ml + (w - ml - mr) * i / max(n - 1, 1)

    def Y(v):
        return mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo))

    out = ['<svg viewBox="0 0 %d %d" role="img" style="width:100%%;height:auto">' % (w, h)]
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="%s"/>' % (ml, w - mr, Y(v), Y(v), GRID))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="%s" text-anchor="end">%s</text>'
                   % (ml - 6, Y(v) + 4, MUTED, "{:,.2f}".format(v)))
    for k in range(5):
        i = round((n - 1) * k / 4)
        out.append('<text x="%.1f" y="%d" font-size="11" fill="%s" text-anchor="middle">%s</text>'
                   % (X(i), h - 8, MUTED, escape(labels[i])))
    if band:
        pts = [(i, lo_v, hi_v) for i, (lo_v, hi_v) in enumerate(zip(*band)) if lo_v is not None]
        poly = ["%.1f,%.1f" % (X(i), Y(u)) for i, _, u in pts] + \
               ["%.1f,%.1f" % (X(i), Y(l)) for i, l, _ in reversed(pts)]
        out.append('<polygon points="%s" fill="%s" opacity="0.15"/>' % (" ".join(poly), AMBER))
    for s in series:
        seg = []
        for i, v in enumerate(s["values"] + [None]):
            if v is not None:
                seg.append("%.1f,%.1f" % (X(i), Y(v)))
            elif seg:
                out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"%s/>'
                           % (" ".join(seg), s["color"], ' stroke-dasharray="6 4"' if s.get("dash") else ""))
                seg = []
    lx = ml + 8
    for s in series:
        out.append('<rect x="%d" y="%d" width="14" height="3" fill="%s"/>' % (lx, mt + 4, s["color"]))
        out.append('<text x="%d" y="%d" font-size="12" fill="%s">%s</text>' % (lx + 20, mt + 9, INK, escape(s["name"])))
        lx += 30 + 7 * len(s["name"])
    out.append("</svg>")
    return "".join(out)


def _price_chart(r):
    t, f = r["test"], r["forecast"]
    nt, nf = len(t["dates"]), len(f["dates"])
    labels = t["dates"] + f["dates"]
    pad = [None] * nf
    anchor = t["actual"][-1]
    fvals = [None] * (nt - 1) + [anchor] + f["price"]
    band = ([None] * (nt - 1) + [anchor] + f["lower"], [None] * (nt - 1) + [anchor] + f["upper"])
    return svg_chart(labels, [
        {"name": "Actual close", "values": t["actual"] + pad, "color": INK},
        {"name": "Predicted close", "values": t["predicted"] + pad, "color": BLUE},
        {"name": "Forecast (95% cone)", "values": fvals, "color": AMBER, "dash": True},
    ], band=band)


def _equity_chart(r):
    b = r["backtest"]
    return svg_chart(b["dates"], [
        {"name": "Model strategy", "values": b["strategy"], "color": BLUE},
        {"name": "Buy & hold", "values": b["benchmark"], "color": MUTED},
    ])


def _kpi(label, value, sub="", color=INK):
    return ('<div class="kpi"><div class="kl">%s</div><div class="kv" style="color:%s">%s</div>'
            '<div class="ks">%s</div></div>' % (escape(label), color, escape(value), escape(sub)))


def render_report(r):
    nd, st = r["next_day"], r["backtest"]["stats"]
    sel = next(c for c in r["comparison"] if c["model"] == r["model"])
    up = nd["predicted_return_pct"] >= 0
    kpis = "".join([
        _kpi("Next-day signal", nd["signal"], "%+.2f%% expected" % nd["predicted_return_pct"], GREEN if nd["signal"] == "LONG" else MUTED),
        _kpi("Predicted next close", "{:,.2f}".format(nd["predicted_close"]), "last close {:,.2f} on {}".format(r["last_close"], r["last_date"]), GREEN if up else RED),
        _kpi("Directional accuracy", "%.1f%%" % sel["directional_accuracy"], "50% is a coin flip"),
        _kpi("MAPE", "%.2f%%" % sel["mape"], "naive baseline %.2f%%" % r["baseline"]["mape"]),
        _kpi("Strategy return", "%+.1f%%" % st["strategy_return_pct"], "buy & hold %+.1f%%" % st["benchmark_return_pct"], GREEN if st["alpha_pct"] >= 0 else RED),
        _kpi("Sharpe / Max DD", "%.2f / %.1f%%" % (st["sharpe"], st["max_drawdown_pct"]), "benchmark %.2f / %.1f%%" % (st["benchmark_sharpe"], st["benchmark_max_drawdown_pct"])),
    ])
    rows = "".join(
        "<tr%s><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.2f%%</td><td>%.1f%%</td><td>%+.1f%%</td><td>%+.1f%%</td><td>%.2f</td><td>%.1f%%</td></tr>"
        % (' class="sel"' if c["model"] == r["model"] else "", c["model"].replace("_", " "), c["mae"], c["rmse"], c["mape"],
           c["directional_accuracy"], c["strategy_return_pct"], c["alpha_pct"], c["sharpe"], c["max_drawdown_pct"])
        for c in r["comparison"])
    frows = "".join("<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>" % (d, p, lo, up_)
                    for d, p, lo, up_ in zip(r["forecast"]["dates"], r["forecast"]["price"], r["forecast"]["lower"], r["forecast"]["upper"]))
    note = ""
    if r["data_source"] == "synthetic":
        note = '<p class="warn">Live data was unavailable, so this run used a simulated random-walk series. Results are for demonstration only.</p>'
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>%(t)s prediction report</title>
<style>
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:%(ink)s;background:#f6f7fa;margin:0;padding:24px}
.wrap{max-width:940px;margin:0 auto}h1{font-size:26px;margin:0 0 2px}h2{font-size:17px;margin:28px 0 8px}
.sub{color:%(mu)s;margin:0 0 18px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.kpi{background:#fff;border:1px solid %(gr)s;border-radius:8px;padding:12px 14px}.kl{color:%(mu)s;font-size:13px}
.kv{font-size:22px;font-weight:650;margin:2px 0}.ks{color:%(mu)s;font-size:12px}
.card{background:#fff;border:1px solid %(gr)s;border-radius:8px;padding:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%%;font-size:13px;background:#fff}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid %(gr)s}
th:first-child,td:first-child{text-align:left}th{color:%(mu)s;font-weight:600}tr.sel td{background:#eef3ff;font-weight:600}
.warn{background:#fff6e0;border:1px solid #f0d28a;border-radius:8px;padding:10px 12px}.fine{color:%(mu)s;font-size:12px;margin-top:24px}
</style></head><body><div class="wrap">
<h1>%(t)s &middot; %(m)s</h1>
<p class="sub">%(bars)d daily bars (%(p)s) from %(src)s &middot; %(tr)d training / %(te)d out-of-sample rows &middot; generated %(gen)s</p>
%(note)s<div class="kpis">%(kpis)s</div>
<h2>Out-of-sample prediction and forecast</h2><div class="card">%(c1)s</div>
<h2>Backtest: $10,000 starting capital</h2><div class="card">%(c2)s</div>
<h2>Model comparison (out-of-sample)</h2><div class="card"><table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Direction</th><th>Return</th><th>Alpha</th><th>Sharpe</th><th>Max DD</th></tr>%(rows)s</table></div>
<h2>Forecast</h2><div class="card"><table><tr><th>Date</th><th>Price</th><th>95%% low</th><th>95%% high</th></tr>%(frows)s</table></div>
<p class="fine">Educational project, not financial advice. Daily returns are very noisy; compare every model against the naive baseline (tomorrow = today) before trusting it.</p>
</div></body></html>""" % {
        "t": escape(r["ticker"]), "m": escape(r["model"].replace("_", " ")), "bars": r["bars"], "p": escape(r["period"]),
        "src": escape(r["data_source"]), "tr": r["train_rows"], "te": r["test_rows"], "gen": escape(r["generated_at"]),
        "note": note, "kpis": kpis, "c1": _price_chart(r), "c2": _equity_chart(r), "rows": rows, "frows": frows,
        "ink": INK, "mu": MUTED, "gr": GRID,
    }
