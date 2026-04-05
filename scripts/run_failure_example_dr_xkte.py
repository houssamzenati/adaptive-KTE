from math import sqrt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from xkte_nonadaptive import xMMD2dr


def make_data(n, d, rng, sigma_eps=0.4, delta_scale=0.0):
    X = rng.normal(size=(n, d))
    v = rng.normal(size=d)
    v /= max(np.linalg.norm(v), 1e-12)
    u = rng.normal(size=d)
    u /= max(np.linalg.norm(u), 1e-12)
    f = X @ v
    delta = delta_scale * (X @ u)
    eps = rng.normal(scale=sigma_eps, size=n)
    return X, f, f + delta, u, eps


def policy_explore_commit(x, t, t0, eps, u, winner):
    if t < t0:
        return 0.5
    in_set = (x @ u) > 0
    prefer_one = (winner == 1 and in_set) or (winner == 0 and not in_set)
    return 1.0 - eps if prefer_one else eps


def run_one(seed, n=700, d=5, t0=15, eps=0.001, sigma_eps=1.0):
    rng = np.random.RandomState(seed)
    X, m0, m1, u, noise = make_data(n, d, rng, sigma_eps=sigma_eps, delta_scale=0.0)
    winner = 1 if rng.rand() < 0.5 else 0
    p = np.array([policy_explore_commit(X[t], t, t0, eps, u, winner) for t in range(n)])
    A = (rng.rand(n) < p).astype(int)
    Y = np.where(A == 1, m1, m0) + noise
    Y2d = Y[:, None]

    YY0 = Y2d[A == 0]
    YY1 = Y2d[A == 1]
    sigma2 = np.median(pairwise_distances(YY0, YY1, metric="euclidean")) ** 2
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.var(Y2d)) + 1e-6
    gamma = 1.0 / sigma2
    return xMMD2dr(Y2d, p, X, A, kernel_function="rbf", lam=1e-1, gamma=gamma)


def main():
    stats = np.array([run_one(seed) for seed in tqdm(range(500))])
    bins = np.linspace(np.floor(stats.min() * 2) / 2, np.ceil(stats.max() * 2) / 2, 35)
    xs = np.linspace(bins[0], bins[-1], 400)

    plt.figure(figsize=(7, 4))
    plt.hist(stats, bins=bins, density=True, alpha=0.75, label="DR-xKTE")
    plt.plot(xs, np.exp(-0.5 * xs**2) / sqrt(2 * np.pi), linestyle="--", color="black", label="N(0,1)")
    plt.title("Failure of DR-xKTE under adaptive ETC sampling")
    plt.xlabel("Z-statistic")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/dr_only_hist.png", dpi=300, bbox_inches="tight")

    reject_rate = np.mean(0.5 * (1.0 - erf(stats / sqrt(2.0))) < 0.05)
    print(f"Empirical rejection rate at alpha=0.05: {reject_rate:.3f}")


if __name__ == "__main__":
    main()
