"""Pure-Python regressors: histogram GBDT, Random Forest, Ridge, blended ensemble."""
import math
import random
from bisect import bisect_right

MODEL_NAMES = ["gradient_boosting", "random_forest", "ridge", "ensemble"]
TREE_WEIGHT, RIDGE_WEIGHT = 0.65, 0.35


# ----------------------------------------------------------------- helpers
class Binner:
    """Quantile-bin each feature so tree split search is O(n * features)."""

    def __init__(self, n_bins=16):
        self.n_bins = n_bins
        self.edges = []

    def fit(self, X):
        n = len(X)
        self.edges = []
        for f in range(len(X[0])):
            col = sorted(r[f] for r in X)
            e = []
            for k in range(1, self.n_bins):
                v = col[min(n - 1, int(k * n / self.n_bins))]
                if not e or v > e[-1]:
                    e.append(v)
            self.edges.append(e)
        return self

    def transform(self, X):
        edges = self.edges
        nf = len(edges)
        return [[bisect_right(edges[f], r[f]) for f in range(nf)] for r in X]


def _solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            M[col][col] = 1e-12
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            if f:
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return x


# -------------------------------------------------------------------- tree
class _Tree:
    """CART regression tree on binned features (variance-reduction splits)."""

    def __init__(self, max_depth, min_leaf, n_bins, max_features=None, rng=None):
        self.max_depth, self.min_leaf, self.n_bins = max_depth, min_leaf, n_bins
        self.max_features, self.rng = max_features, rng or random.Random(0)
        self.root = None

    def fit(self, B, y, idx):
        self.nf = len(B[0])
        self.root = self._build(B, y, idx, 0)
        return self

    def _build(self, B, y, idx, depth):
        n = len(idx)
        total = sum(y[i] for i in idx)
        if depth >= self.max_depth or n < 2 * self.min_leaf:
            return (None, total / n)
        feats = range(self.nf)
        if self.max_features and self.max_features < self.nf:
            feats = self.rng.sample(range(self.nf), self.max_features)
        base, best = total * total / n, None
        for f in feats:
            cnt = [0] * self.n_bins
            sm = [0.0] * self.n_bins
            for i in idx:
                b = B[i][f]
                cnt[b] += 1
                sm[b] += y[i]
            cl, sl = 0, 0.0
            for b in range(self.n_bins - 1):
                cl += cnt[b]
                sl += sm[b]
                cr = n - cl
                if cl < self.min_leaf or cr < self.min_leaf:
                    continue
                gain = sl * sl / cl + (total - sl) ** 2 / cr - base
                if best is None or gain > best[0]:
                    best = (gain, f, b)
        if best is None or best[0] <= 1e-18:
            return (None, total / n)
        _, f, thr = best
        left = [i for i in idx if B[i][f] <= thr]
        right = [i for i in idx if B[i][f] > thr]
        return (f, thr, self._build(B, y, left, depth + 1), self._build(B, y, right, depth + 1))

    def predict_row(self, row):
        node = self.root
        while node[0] is not None:
            node = node[2] if row[node[0]] <= node[1] else node[3]
        return node[1]


# ------------------------------------------------------------------ models
class GradientBoosting:
    """Gradient Boosted Decision Trees with shrinkage (eta = 0.07)."""

    def __init__(self, n_estimators=120, learning_rate=0.07, max_depth=3,
                 min_leaf=8, subsample=0.8, seed=42):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.subsample, self.seed = subsample, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        n = len(y)
        self.init = sum(y) / n
        pred = [self.init] * n
        self.trees = []
        k = max(2 * self.min_leaf, int(n * self.subsample))
        for _ in range(self.n_estimators):
            resid = [y[i] - pred[i] for i in range(n)]          # negative gradient (L2)
            idx = rng.sample(range(n), min(k, n))
            tree = _Tree(self.max_depth, self.min_leaf, self.binner.n_bins).fit(B, resid, idx)
            self.trees.append(tree)
            for i in range(n):
                pred[i] += self.lr * tree.predict_row(B[i])
        return self

    def predict(self, X):
        B = self.binner.transform(X)
        return [self.init + self.lr * sum(t.predict_row(r) for t in self.trees) for r in B]


class RandomForest:
    """Bagged trees with random feature subspaces at every split."""

    def __init__(self, n_estimators=60, max_depth=5, min_leaf=5, seed=42):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_leaf, self.seed = min_leaf, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        n, nf = len(y), len(X[0])
        mf = max(1, int(math.sqrt(nf)))
        self.trees = []
        for _ in range(self.n_estimators):
            idx = [rng.randrange(n) for _ in range(n)]          # bootstrap sample
            self.trees.append(_Tree(self.max_depth, self.min_leaf, self.binner.n_bins,
                                    max_features=mf, rng=rng).fit(B, y, idx))
        return self

    def predict(self, X):
        B = self.binner.transform(X)
        return [sum(t.predict_row(r) for t in self.trees) / len(self.trees) for r in B]


class Ridge:
    """L2-regularised linear regression on z-scored features (lambda = 0.5)."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y):
        n, nf = len(X), len(X[0])
        self.mean = [sum(r[f] for r in X) / n for f in range(nf)]
        self.std = []
        for f in range(nf):
            var = sum((r[f] - self.mean[f]) ** 2 for r in X) / n
            self.std.append(math.sqrt(var) or 1.0)
        Z = [[(r[f] - self.mean[f]) / self.std[f] for f in range(nf)] for r in X]
        self.intercept = sum(y) / n
        yc = [v - self.intercept for v in y]
        A = [[0.0] * nf for _ in range(nf)]
        b = [0.0] * nf
        for z, t in zip(Z, yc):
            for i in range(nf):
                zi = z[i]
                b[i] += zi * t
                row = A[i]
                for j in range(i, nf):
                    row[j] += zi * z[j]
        for i in range(nf):
            for j in range(i):
                A[i][j] = A[j][i]
            A[i][i] += self.alpha
        self.coef = _solve(A, b)
        return self

    def predict(self, X):
        nf = len(self.coef)
        return [self.intercept + sum(self.coef[f] * (r[f] - self.mean[f]) / self.std[f]
                                     for f in range(nf)) for r in X]


class Ensemble:
    """Convex blend: 65% tree ensembles (GBDT + RF average) + 35% Ridge."""

    def __init__(self, gbdt, forest, ridge):
        self.gbdt, self.forest, self.ridge = gbdt, forest, ridge

    def predict(self, X):
        g, f, r = self.gbdt.predict(X), self.forest.predict(X), self.ridge.predict(X)
        return [TREE_WEIGHT * 0.5 * (a + b) + RIDGE_WEIGHT * c for a, b, c in zip(g, f, r)]


def train_all(X, y, seed=42):
    """Fit the three base learners and assemble the ensemble."""
    gbdt = GradientBoosting(seed=seed).fit(X, y)
    forest = RandomForest(seed=seed).fit(X, y)
    ridge = Ridge().fit(X, y)
    return {"gradient_boosting": gbdt, "random_forest": forest,
            "ridge": ridge, "ensemble": Ensemble(gbdt, forest, ridge)}
