# experiments_dsprite.py
import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from scipy.stats import norm

# === project imports ===
from dsprite_adaptive import collect_adaptive_kte_dsprite
from dr_kte_adaptive import xMMD2_vsdr_fold_generic
from xkte_nonadaptive import xMMD2dr
from baselines import cadr_test, hadad_test, fit_krr_predict

# -----------------------
# Config
# -----------------------
EXPNAME = "dsprite_kte_binary_adaptive"
RESULT_ROOT = f"results/{EXPNAME}"
PARAM_ROOT = "experiment_parameters"

NB_SEEDS = 200
SCENARIOS = ["I", "IV"]
METHODS = ["VS-DR-KTE", "CADR", "AW-AIPW"]  # Hadad shown as AW-AIPW

os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(PARAM_ROOT, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def _gamma_on_images(Y2d, T):
    """
    Robust RBF bandwidth for image kernel:
    median distance between arms (YY0 vs YY1) with fallback to Var(Y).
    Returns gamma = 1/sigma^2.
    """
    YY0, YY1 = Y2d[T == 0], Y2d[T == 1]
    if YY0.size == 0 or YY1.size == 0:
        return 1.0 / (float(np.var(Y2d)) + 1e-6)
    med = np.median(pairwise_distances(YY0, YY1, metric="euclidean"))
    sigma2 = (med**2) / 4.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.var(Y2d)) + 1e-6
    return 1.0 / sigma2


def _gamma_on_X(X):
    Dx = pairwise_distances(X, X, metric="euclidean") ** 2
    med2_x = np.median(Dx[np.triu_indices_from(Dx, k=1)])
    return 1.0 / max(med2_x, 1e-6)


# -----------------------
# Single experiment
# -----------------------
def run_single_experiment(scenario_id, method, seed):
    """
    Runs one replicate for a given scenario/method.
    CADR / AW-AIPW use the scalar proxy = mean pixel of the image.
    DR-KTE / VS-DR-KTE operate on the full images.
    """
    ns = 1000
    d = 2
    eps0 = 0.5
    eps_min = 0.2
    power = 0.5
    lam = 1e-2
    split = "alternating"
    one_indexed = True
    shift_pixels = (
        6  # strong, visible effect in Scenario IV (quadrant/shift inside collector)
    )

    rng = np.random.RandomState(seed)
    X, T, Y2d, w1, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all = (
        collect_adaptive_kte_dsprite(
            ns=ns,
            d=d,
            scenario=scenario_id,
            eps0=eps0,
            eps_min=eps_min,
            power=power,
            lam=lam,
            rng=rng,
            split=split,
            one_indexed=one_indexed,
            shift_pixels=shift_pixels,
        )
    )

    # Kernel bandwidth on images
    gamma_k = _gamma_on_images(Y2d, T)

    # Scalar proxy (mean pixel) & nuisances for CADR / AW-AIPW
    # Y2d is (n, 4096) so the mean pixel per image is simply mean over axis=1
    y_scalar = Y2d.mean(axis=1)
    gamma_x = _gamma_on_X(X)
    m0_hat = (
        fit_krr_predict(
            X[T == 0],
            y_scalar[T == 0],
            X,
            kernel_function="rbf",
            gamma=gamma_x,
            lam=1e-4,
        )
        if np.any(T == 0)
        else np.full(X.shape[0], y_scalar.mean())
    )
    m1_hat = (
        fit_krr_predict(
            X[T == 1],
            y_scalar[T == 1],
            X,
            kernel_function="rbf",
            gamma=gamma_x,
            lam=1e-4,
        )
        if np.any(T == 1)
        else np.full(X.shape[0], y_scalar.mean())
    )

    # Run method
    stat = np.nan
    pval = np.nan
    try:
        t0 = time.time()
        if method == "VS-DR-KTE":
            # variance-stabilized, adaptive-aware DR-KTE on images
            stat = xMMD2_vsdr_fold_generic(
                Y=Y2d,
                w=w1,
                X=X,
                A=T,
                kernel_function="rbf",
                Pi_0_on_0=Pi_0_on_0,
                Pi_1_on_1=Pi_1_on_1,
                idx0=idx0,
                idx1=idx1,
                gamma=gamma_k,
                lam=1e-2,
            )
            pval = norm.sf(stat)

        elif method == "CADR":
            out = cadr_test(
                X=X,
                A=T,
                Y=y_scalar,
                p_realized=w1,
                m0=m0_hat,
                m1=m1_hat,
                P_all=P_all,
            )
            stat = out["stat"]
            pval = norm.sf(stat)

        elif method == "AW-AIPW":
            out = hadad_test(
                A=T,
                Y=y_scalar,
                p=w1,
                m0=m0_hat,
                m1=m1_hat,
                scheme="two_point",  # or 'constant'
                alpha=0.7,
                p_min=0.0,
                estimator="aipw",
            )
            stat = out["stat"]
            pval = norm.sf(stat)

        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - t0
    except Exception as e:
        print(
            f"[warn] run failed (scenario={scenario_id}, method={method}, seed={seed}): {e}"
        )
        stat, pval, elapsed = np.nan, np.nan, 0.0

    return {
        "p_value": float(pval),
        "stat": float(stat),
        "time": float(elapsed),
        "scenario": scenario_id,
        "method": method,
        "seed": int(seed),
    }


# -----------------------
# Batch runners
# -----------------------
def sequential_run():
    for seed in tqdm(range(NB_SEEDS)):
        for method in METHODS:
            for scenario in SCENARIOS:
                res = run_single_experiment(
                    scenario_id=scenario,
                    method=method,
                    seed=seed,
                )
                df = pd.DataFrame([res])
                fname = f"{RESULT_ROOT}/scenario{scenario}_{method}_seed{seed}.csv"
                df.to_csv(fname, index=False)


def get_parameters_experiment():
    param_csv = os.path.join(PARAM_ROOT, f"{EXPNAME}_parameters.csv")
    seeds = list(range(NB_SEEDS))
    with open(param_csv, "w") as f:
        f.write("line,scenario,method,seed\n")
        line = 0
        for scenario in SCENARIOS:
            for method in METHODS:
                for seed in seeds:
                    f.write(f"{line},{scenario},{method},{seed}\n")
                    line += 1
    print(f"Parameter file written to {param_csv}")


def run_from_arguments(parameters):
    res = run_single_experiment(
        scenario_id=parameters.scenario,
        method=parameters.method,
        seed=parameters.seed,
    )
    df = pd.DataFrame([res])
    fname = f"{RESULT_ROOT}/scenario{parameters.scenario}_{parameters.method}_seed{parameters.seed}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}")


# -----------------------
# Aggregation
# -----------------------
def get_results_table():
    alpha = 0.05
    seeds = range(NB_SEEDS)

    n_scen, n_meth = len(SCENARIOS), len(METHODS)
    p_hat = np.full((n_scen, n_meth), np.nan)
    se = np.full((n_scen, n_meth), np.nan)
    n_eff = np.zeros((n_scen, n_meth), dtype=int)

    for i, scenario in enumerate(SCENARIOS):
        for j, method in enumerate(METHODS):
            vals = []
            for seed in seeds:
                fp = f"{RESULT_ROOT}/scenario{scenario}_{method}_seed{seed}.csv"
                try:
                    df = pd.read_csv(fp)
                    # One row per seed -> 0/1 reject indicator
                    vals.append(float((df["p_value"] < alpha).mean()))
                except FileNotFoundError:
                    continue
            k = len(vals)
            if k > 0:
                ph = float(np.mean(vals))
                p_hat[i, j] = ph
                se[i, j] = float(np.sqrt(ph * (1.0 - ph) / k))
                n_eff[i, j] = k

    # Print numeric summaries
    df_mean = pd.DataFrame(p_hat, index=SCENARIOS, columns=METHODS)
    df_se = pd.DataFrame(se, index=SCENARIOS, columns=METHODS)
    print("\nMean rejection rates:")
    print(df_mean.round(2))
    print("\nStandard error across seeds:")
    print(df_se.round(3))
    print("\nEffective seeds per cell:")
    print(pd.DataFrame(n_eff, index=SCENARIOS, columns=METHODS))

    # LaTeX table with mean ± SE
    df_pm = df_mean.copy().astype(object)
    for i in range(n_scen):
        for j in range(n_meth):
            if np.isnan(p_hat[i, j]):
                df_pm.iat[i, j] = "--"
            else:
                df_pm.iat[i, j] = f"${p_hat[i, j]:.2f}\\,\\pm\\,{se[i, j]:.2f}$"

    latex_path = os.path.join(RESULT_ROOT, "rejection_table_mean_se.tex")
    with open(latex_path, "w") as f:
        f.write(df_pm.T.to_latex(escape=False))
    print(f"LaTeX table saved to {latex_path}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sequential_run", action="store_true")
    parser.add_argument("--get_parameters_experiment", action="store_true")
    parser.add_argument("--results", action="store_true")

    parser.add_argument("--scenario", type=str, choices=SCENARIOS, default="I")
    parser.add_argument("--method", type=str, choices=METHODS, default="DR-KTE")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.get_parameters_experiment:
        get_parameters_experiment()
    if args.run:
        run_from_arguments(args)
    if args.sequential_run:
        sequential_run()
    if args.results:
        get_results_table()
