"""Unit tests. Run: python test_predictor.py"""
import copy
import math
import os
import random
import tempfile
import unittest

from stockpredictor import metrics as M
from stockpredictor.backtest import run_backtest
from stockpredictor.data import synthetic_bars
from stockpredictor.features import FEATURE_NAMES, WARMUP, build_features, ema, rsi, sma
from stockpredictor.forecast import forecast_future, next_business_days
from stockpredictor.models import GradientBoosting, RandomForest, Ridge, train_all
from stockpredictor.pipeline import run_pipeline
from stockpredictor.report import render_report


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.bars = synthetic_bars("TEST", 200)

    def test_shape_and_finite(self):
        f = build_features(self.bars)
        self.assertEqual(len(f["X"]), len(self.bars) - WARMUP)
        self.assertEqual(len(f["X"][0]), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for row in f["X"] for v in row))

    def test_no_lookahead(self):
        base = build_features(self.bars)["X"]
        changed = copy.deepcopy(self.bars)
        changed[-1]["close"] *= 1.5          # alter only the final bar
        alt = build_features(changed)["X"]
        self.assertEqual(base[:-1], alt[:-1])  # every earlier row unchanged

    def test_indicators(self):
        self.assertEqual(sma([1, 2, 3, 4], 2)[1:], [1.5, 2.5, 3.5])
        self.assertAlmostEqual(ema([5, 5, 5], 3)[-1], 5)
        self.assertEqual(rsi([1, 2, 3, 4, 5, 6])[-1], 100.0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(300)]
        self.y = [0.5 * r[0] - 0.3 * r[1] + rng.gauss(0, 0.05) for r in self.X]

    def corr(self, p):
        mp, my = sum(p) / len(p), sum(self.y) / len(self.y)
        cov = sum((a - mp) * (b - my) for a, b in zip(p, self.y))
        return cov / math.sqrt(sum((a - mp) ** 2 for a in p) * sum((b - my) ** 2 for b in self.y))

    def test_ridge_recovers_signal(self):
        self.assertGreater(self.corr(Ridge().fit(self.X, self.y).predict(self.X)), 0.95)

    def test_trees_learn(self):
        self.assertGreater(self.corr(GradientBoosting(n_estimators=40).fit(self.X, self.y).predict(self.X)), 0.8)
        self.assertGreater(self.corr(RandomForest(n_estimators=20).fit(self.X, self.y).predict(self.X)), 0.6)

    def test_ensemble_is_blend(self):
        m = train_all(self.X[:200], self.y[:200])
        row = [self.X[250]]
        g, f, r = (m[k].predict(row)[0] for k in ("gradient_boosting", "random_forest", "ridge"))
        self.assertAlmostEqual(m["ensemble"].predict(row)[0], 0.65 * (g + f) / 2 + 0.35 * r)


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        self.assertAlmostEqual(M.mae([1, 2], [2, 2]), 0.5)
        self.assertAlmostEqual(M.rmse([0, 0], [3, 4]), math.sqrt(12.5))
        self.assertAlmostEqual(M.directional_accuracy([1, -1], [2, 3]), 50.0)
        self.assertAlmostEqual(M.max_drawdown([100, 120, 90]), -25.0)

    def test_backtest(self):
        bt = run_backtest("d0", ["d1", "d2"], [0.01, -0.01], [0.10, -0.10])
        self.assertAlmostEqual(bt["strategy"][-1], 11000.0)      # long day 1, cash day 2
        self.assertAlmostEqual(bt["benchmark"][-1], 10000 * 1.1 * 0.9)

    def test_business_days(self):
        self.assertEqual(next_business_days("2026-10-02", 2), ["2026-10-05", "2026-10-06"])  # Fri -> Mon, Tue

    def test_forecast_cone_widens(self):
        bars = synthetic_bars("TEST", 200)
        m = Ridge().fit(*_xy(bars))
        fc = forecast_future(bars, m, 5, 0.01)
        widths = [u - l for u, l in zip(fc["upper"], fc["lower"])]
        self.assertEqual(widths, sorted(widths))


def _xy(bars):
    f = build_features(bars)
    c = [b["close"] for b in bars]
    rows = [(x, i) for x, i in zip(f["X"], f["indices"]) if i < len(c) - 1]
    return [x for x, _ in rows], [c[i + 1] / c[i] - 1 for _, i in rows]


class PipelineTests(unittest.TestCase):
    def test_end_to_end(self):
        r = run_pipeline("UNITTEST", "1y", "ensemble", 5, offline=True)
        self.assertEqual(len(r["forecast"]["price"]), 5)
        self.assertEqual(len(r["comparison"]), 4)
        self.assertEqual(r["train_rows"] + r["test_rows"], r["bars"] - WARMUP - 1)
        # price reconstruction identity: P_hat = P_t * (1 + r_hat) stays positive and finite
        self.assertTrue(all(p > 0 and math.isfinite(p) for p in r["test"]["predicted"]))
        html = render_report(r)
        self.assertIn("<svg", html)
        self.assertIn("UNITTEST", html)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run_pipeline("AAPL", "1y", "nope", 7, offline=True)
        with self.assertRaises(ValueError):
            run_pipeline("AAPL", "1y", "ridge", 99, offline=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
