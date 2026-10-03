#!/usr/bin/env python3
"""Interactive web dashboard (standard library only). Run: python app.py"""
import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from stockpredictor.data import PERIOD_DAYS
from stockpredictor.models import MODEL_NAMES
from stockpredictor.pipeline import run_pipeline
from stockpredictor.report import render_report

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,12}$")


def parse_params(qs):
    def one(key, default):
        return qs.get(key, [default])[0]
    ticker = one("ticker", "AAPL").strip()
    if not TICKER_RE.match(ticker):
        raise ValueError("invalid ticker symbol")
    period, model = one("period", "1y"), one("model", "ensemble")
    if period not in PERIOD_DAYS:
        raise ValueError("period must be one of %s" % ", ".join(PERIOD_DAYS))
    if model not in MODEL_NAMES:
        raise ValueError("model must be one of %s" % ", ".join(MODEL_NAMES))
    try:
        horizon = int(one("forecast", "7"))
    except ValueError:
        raise ValueError("forecast must be a whole number")
    return dict(ticker=ticker, period=period, model=model, forecast_days=horizon,
                offline=one("offline", "0") == "1")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html")
            elif url.path == "/api/health":
                self._send(200, json.dumps({"status": "ok"}), "application/json")
            elif url.path == "/api/predict":
                result = run_pipeline(**parse_params(parse_qs(url.query)))
                self._send(200, json.dumps(result), "application/json")
            elif url.path == "/report":
                result = run_pipeline(**parse_params(parse_qs(url.query)))
                self._send(200, render_report(result), "text/html")
            else:
                self._send(404, "Not found", "text/plain")
        except ValueError as exc:
            self._send(400, "<p style='font:15px system-ui;padding:16px'>%s</p>" % str(exc).replace("<", "&lt;"), "text/html")
        except Exception as exc:  # keep the server alive on unexpected errors
            self._send(500, "<p style='font:15px system-ui;padding:16px'>Unexpected error: %s</p>" % type(exc).__name__, "text/html")

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser(description="AI Stock Price Predictor dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print("Dashboard running at http://%s:%d  (Ctrl+C to stop)" % (a.host, a.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
