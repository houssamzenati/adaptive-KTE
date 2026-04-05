from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def treatment_effect_vector(ns, scenario, rng, beta_mix=2.0, beta_uniform=4.0):
    if scenario == "I":
        return np.zeros(ns)
    if scenario == "II":
        return np.full(ns, 2.0)
    if scenario == "III":
        signs = rng.binomial(1, 0.5, size=ns) * 2 - 1
        return signs.astype(float) * beta_mix
    if scenario == "IV":
        return rng.uniform(-beta_uniform, beta_uniform, size=ns)
    raise ValueError(f"Unknown scenario: {scenario}")


def linear_base(signal):
    return signal


def cosine_base(signal):
    return np.cos(signal)


def sigmoidal_base(signal):
    return np.log(np.abs(16.0 * signal - 8.0) + 1.0) * np.sign(signal - 0.5)


def _ensure_rng(rng):
    return rng if rng is not None else np.random.RandomState(0)


def _epsilon_at(t, eps0, eps_min, power):
    return max(eps_min, eps0 / ((t + 1) ** power))


def _solve_theta(S, b):
    try:
        return np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(S, b, rcond=None)[0]


def _split_indices(ns, split):
    if split == "alternating":
        return np.arange(0, ns, 2), np.arange(1, ns, 2)
    if split == "chronological":
        cut = ns // 2
        return np.arange(0, cut), np.arange(cut, ns)
    raise ValueError("split must be 'alternating' or 'chronological'")


def collect_epsilon_greedy(
    ns,
    d,
    beta_vec,
    noise_std,
    scenario,
    base_fn,
    eps0=0.5,
    eps_min=0.2,
    power=0.5,
    lam=1e-2,
    rng=None,
    split="chronological",
    X_all=None,
    sample_replace=False,
):
    rng = _ensure_rng(rng)
    if X_all is None:
        X = rng.randn(ns, d)
    else:
        X_all = np.asarray(X_all, dtype=float)
        if ns > X_all.shape[0] and not sample_replace:
            raise ValueError("ns exceeds available rows and sample_replace=False")
        sample_idx = rng.choice(X_all.shape[0], size=ns, replace=sample_replace)
        X = X_all[sample_idx].copy()

    signal = X @ beta_vec
    base = base_fn(signal)
    delta = treatment_effect_vector(ns, scenario, rng)
    Y0 = base + noise_std * rng.randn(ns)
    Y1 = base + noise_std * rng.randn(ns) + delta

    X_aug = np.hstack([np.ones((ns, 1)), X])
    S0 = np.diag([0.0] + [lam] * d)
    S1 = np.diag([0.0] + [lam] * d)
    b0 = np.zeros(d + 1)
    b1 = np.zeros(d + 1)

    A = np.zeros(ns, dtype=int)
    w = np.zeros(ns, dtype=float)
    Y = np.zeros(ns, dtype=float)
    theta0_snap = np.zeros((ns, d + 1))
    theta1_snap = np.zeros((ns, d + 1))

    for t in range(ns):
        th0 = _solve_theta(S0, b0)
        th1 = _solve_theta(S1, b1)
        theta0_snap[t] = th0
        theta1_snap[t] = th1

        eps_t = _epsilon_at(t, eps0, eps_min, power)
        z_t = X_aug[t]
        q0 = z_t @ th0
        q1 = z_t @ th1
        if q1 > q0:
            pi1 = 1.0 - 0.5 * eps_t
        elif q1 < q0:
            pi1 = 0.5 * eps_t
        else:
            pi1 = 0.5

        a_t = 1 if rng.rand() < pi1 else 0
        y_t = Y1[t] if a_t == 1 else Y0[t]

        A[t] = a_t
        w[t] = pi1
        Y[t] = y_t

        if a_t == 0:
            S0 += np.outer(z_t, z_t)
            b0 += z_t * y_t
        else:
            S1 += np.outer(z_t, z_t)
            b1 += z_t * y_t

    idx0, idx1 = _split_indices(ns, split)
    Z0, Z1 = X_aug[idx0], X_aug[idx1]

    pi0_on_0 = np.empty((len(idx0), len(idx0)))
    for row, t in enumerate(idx0):
        th0 = theta0_snap[t]
        th1 = theta1_snap[t]
        eps_t = _epsilon_at(t, eps0, eps_min, power)
        q0 = Z0 @ th0
        q1 = Z0 @ th1
        pi0_on_0[row] = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )

    pi1_on_1 = np.empty((len(idx1), len(idx1)))
    for row, t in enumerate(idx1):
        th0 = theta0_snap[t]
        th1 = theta1_snap[t]
        eps_t = _epsilon_at(t, eps0, eps_min, power)
        q0 = Z1 @ th0
        q1 = Z1 @ th1
        pi1_on_1[row] = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )

    p_all = np.empty((ns, ns), dtype=np.float32)
    for t in range(ns):
        th0 = theta0_snap[t]
        th1 = theta1_snap[t]
        eps_t = _epsilon_at(t, eps0, eps_min, power)
        q0_all = X_aug @ th0
        q1_all = X_aug @ th1
        p_all[t] = np.where(
            q1_all > q0_all,
            1.0 - 0.5 * eps_t,
            np.where(q1_all < q0_all, 0.5 * eps_t, 0.5),
        )

    return X, A, Y[:, None], w, pi0_on_0, pi1_on_1, idx0, idx1, p_all


def collect_iid_logging(
    ns,
    d,
    beta_vec,
    noise_std,
    scenario,
    base_fn,
    rng=None,
    split="chronological",
    policy="logistic",
):
    rng = _ensure_rng(rng)
    X = rng.randn(ns, d)
    signal = X @ beta_vec
    base = base_fn(signal)
    delta = treatment_effect_vector(ns, scenario, rng)
    Y0 = base + noise_std * rng.randn(ns)
    Y1 = base + noise_std * rng.randn(ns) + delta

    if policy == "logistic":
        w = 1.0 / (1.0 + np.exp(-signal))
    elif policy == "uniform":
        w = np.full(ns, 0.5)
    else:
        raise ValueError("policy must be 'logistic' or 'uniform'")

    A = (rng.rand(ns) < w).astype(int)
    Y = np.where(A == 1, Y1, Y0)

    idx0, idx1 = _split_indices(ns, split)
    pi0_on_0 = np.tile(w[idx0], (len(idx0), 1))
    pi1_on_1 = np.tile(w[idx1], (len(idx1), 1))
    p_all = np.tile(w, (ns, 1))
    return X, A, Y[:, None], w, pi0_on_0, pi1_on_1, idx0, idx1, p_all


def load_ihdp_features(data_path=None):
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "ihdp.csv"
    data_path = Path(data_path)
    df = pd.read_csv(data_path, index_col=0)

    covs_cont = [
        "bw",
        "momage",
        "nnhealth",
        "birth.o",
        "parity",
        "moreprem",
        "cigs",
        "alcohol",
        "ppvt.imp",
    ]
    covs_cat = [
        "bwg",
        "female",
        "mlt.birt",
        "b.marry",
        "livwho",
        "language",
        "whenpren",
        "drugs",
        "othstudy",
    ]
    features = covs_cont + covs_cat
    frame = df[features + ["iqsb.36", "treat"]].dropna().copy()
    scaler = StandardScaler()
    frame[features] = scaler.fit_transform(frame[features])
    return frame[features].to_numpy(), features

