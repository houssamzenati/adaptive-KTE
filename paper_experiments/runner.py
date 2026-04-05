from __future__ import annotations

import time
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from baselines import cadr_test, fit_krr_predict, hadad_test
from dr_kte_adaptive import xMMD2_vsdr_fold_generic
from kte import kernel_two_sample_test_nonuniform
from projected_adaptive_kte import projected_adaptive_kte_test
from xkte_nonadaptive import kernel_dr_two_sample_test_agnostic


def method_slug(method):
    return method.lower().replace("-", "_").replace(" ", "_")


def _normal_sf(value):
    return 0.5 * erfc(value / sqrt(2.0))


def _gamma_on_outcomes(Y, A):
    Y = np.asarray(Y)
    YY0 = Y[A == 0]
    YY1 = Y[A == 1]
    if YY0.size == 0 or YY1.size == 0:
        return 1.0 / (float(np.var(Y)) + 1e-6)
    sigma2 = np.median(pairwise_distances(YY0, YY1, metric="euclidean")) ** 2 / 4.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.var(Y)) + 1e-6
    return 1.0 / sigma2


def _gamma_on_covariates(X):
    Dx = pairwise_distances(X, X, metric="euclidean") ** 2
    med2_x = np.median(Dx[np.triu_indices_from(Dx, k=1)])
    return 1.0 / max(med2_x, 1e-6)


def _fit_scalar_nuisances(X, A, y):
    gamma_x = _gamma_on_covariates(X)
    m0 = (
        fit_krr_predict(X[A == 0], y[A == 0], X, kernel_function="rbf", gamma=gamma_x, lam=1e-4)
        if np.any(A == 0)
        else np.full(X.shape[0], y.mean())
    )
    m1 = (
        fit_krr_predict(X[A == 1], y[A == 1], X, kernel_function="rbf", gamma=gamma_x, lam=1e-4)
        if np.any(A == 1)
        else np.full(X.shape[0], y.mean())
    )
    return m0, m1


def evaluate_method(method, X, A, Y, w, pi0_on_0, pi1_on_1, idx0, idx1, p_all):
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    y_scalar = Y.mean(axis=1) if Y.shape[1] > 1 else Y.reshape(-1)
    outcome_gamma = _gamma_on_outcomes(Y, A)

    started = time.time()
    if method == "PADR-KTE":
        out = projected_adaptive_kte_test(
            Y=Y,
            X=X,
            A=A,
            policy_matrix=p_all,
            idx_pilot=idx0,
            idx_infer=idx1,
            outcome_gamma=outcome_gamma,
        )
        stat = out["statistic"]
        p_value = out["p_value"]
    elif method == "VS-DR-KTE":
        stat = xMMD2_vsdr_fold_generic(
            Y=Y,
            w=w,
            X=X,
            A=A,
            kernel_function="rbf",
            Pi_0_on_0=pi0_on_0,
            Pi_1_on_1=pi1_on_1,
            idx0=idx0,
            idx1=idx1,
            gamma=outcome_gamma,
            lam=1e-2,
        )
        p_value = _normal_sf(stat)
    elif method == "CADR":
        m0, m1 = _fit_scalar_nuisances(X, A, y_scalar)
        stat = cadr_test(X=X, A=A, Y=y_scalar, p_realized=w, m0=m0, m1=m1, P_all=p_all)["stat"]
        p_value = _normal_sf(stat)
    elif method == "AW-AIPW":
        m0, m1 = _fit_scalar_nuisances(X, A, y_scalar)
        stat = hadad_test(
            A=A,
            Y=y_scalar,
            p=w,
            m0=m0,
            m1=m1,
            scheme="two_point",
            alpha=0.7,
            p_min=0.0,
            estimator="aipw",
        )["stat"]
        p_value = _normal_sf(stat)
    elif method == "DR-xKTE":
        stat, p_value = kernel_dr_two_sample_test_agnostic(
            Y,
            X,
            A,
            w,
            kernel_function="rbf",
            lam=1e-1,
            gamma=outcome_gamma,
            verbose=False,
        )
    elif method == "KTE":
        YY0 = Y[A == 0]
        YY1 = Y[A == 1]
        stat, _, p_value = kernel_two_sample_test_nonuniform(
            YY0,
            YY1,
            A,
            w,
            kernel_function="rbf",
            iterations=100,
            gamma=outcome_gamma,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "time": time.time() - started,
        "p_value": float(p_value),
        "stat": float(stat),
    }


def run_experiment_grid(output_dir, collector, scenarios, sample_sizes, methods, num_experiments=200, seed=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    for scenario in scenarios:
        for sample_size in sample_sizes:
            for method in methods:
                rows = []
                for _ in range(num_experiments):
                    local_rng = np.random.RandomState(rng.randint(1, 10**9))
                    X, A, Y, w, pi0_on_0, pi1_on_1, idx0, idx1, p_all = collector(
                        sample_size, scenario, local_rng
                    )
                    res = evaluate_method(
                        method, X, A, Y, w, pi0_on_0, pi1_on_1, idx0, idx1, p_all
                    )
                    res.update({"scenario": scenario, "sample_size": sample_size, "method": method})
                    rows.append(res)

                pd.DataFrame(rows).to_csv(
                    output_dir / f"scenario_{scenario}_n_{sample_size}_{method_slug(method)}.csv",
                    index=False,
                )


def load_result_frame(output_dir, scenario, sample_size, method):
    return pd.read_csv(
        Path(output_dir) / f"scenario_{scenario}_n_{sample_size}_{method_slug(method)}.csv"
    )


def summarize_rejections(output_dir, scenarios, sample_sizes, methods, alpha=0.05):
    rows = []
    for scenario in scenarios:
        for sample_size in sample_sizes:
            for method in methods:
                frame = load_result_frame(output_dir, scenario, sample_size, method)
                pvals = frame["p_value"].to_numpy()
                reject = float((pvals < alpha).mean())
                se95 = 1.96 * np.sqrt(reject * max(1.0 - reject, 0.0) / max(len(pvals), 1))
                rows.append(
                    {
                        "scenario": scenario,
                        "sample_size": sample_size,
                        "method": method,
                        "rejection_rate": reject,
                        "se95": se95,
                        "replications": len(pvals),
                    }
                )
    return pd.DataFrame(rows)

