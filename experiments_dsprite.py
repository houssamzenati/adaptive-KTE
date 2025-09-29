# dsprite_kte_experiment_table.py
import os, time, argparse
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import pairwise_distances

# === your project imports (adjust paths/names if needed) ===
from dsprite_adaptive import collect_adaptive_kte_dsprite
from dr_kte_adaptive import xMMD2_vsdr_fold_generic
from baselines import cadr_test, hadad_test, fit_krr_predict

EXPNAME = "dsprite_kte_binary_adaptive"
RESULT_DIR = f"results/{EXPNAME}/"
PARAM_CSV = f"experiment_parameters/{EXPNAME}_parameters.csv"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("experiment_parameters", exist_ok=True)

NB_SEEDS = 100
SCENARIOS = ["I", "IV"]
METHODS = ["VS-DR-KTE", "CADR", "Hadad"]

def _rbf_gamma_on_images(Y2d, T):
    YY0 = Y2d[T == 0]
    YY1 = Y2d[T == 1]
    if YY0.size == 0 or YY1.size == 0:
        var = float(np.var(Y2d)) + 1e-6
        return 1.0 / var
    med = np.median(pairwise_distances(YY0, YY1, metric="euclidean"))
    sigma2 = (med ** 2) / 4.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        var = float(np.var(Y2d)) + 1e-6
        return 1.0 / var
    return 1.0 / sigma2

def _gamma_on_X(X):
    Dx = pairwise_distances(X, X, metric="euclidean") ** 2
    med2_x = np.median(Dx[np.triu_indices_from(Dx, k=1)])
    return 1.0 / max(med2_x, 1e-6)

def run_single_experiment(scenario_id, method, seed, parameters=None):
    # collector settings (edit here if you need different defaults)
    ns = parameters.get("ns", 1500) if parameters else 1500
    d = parameters.get("d", 2) if parameters else 2
    eps0 = parameters.get("eps0", 0.5) if parameters else 0.5
    eps_min = parameters.get("eps_min", 0.2) if parameters else 0.2
    power = parameters.get("power", 0.5) if parameters else 0.5
    lam = parameters.get("lam", 1e-4) if parameters else 1e-4
    split = parameters.get("split", "alternating") if parameters else "alternating"
    one_indexed = parameters.get("one_indexed", True) if parameters else True
    shift_pixels = parameters.get("shift_pixels", 4) if parameters else 4

    rng = np.random.RandomState(seed)
    X, T, Y2d, w1, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all = collect_adaptive_kte_dsprite(
        ns=ns, d=d, scenario=scenario_id,
        eps0=eps0, eps_min=eps_min, power=power, lam=lam,
        rng=rng, split=split, one_indexed=one_indexed, shift_pixels=shift_pixels
    )

    # VS-DR-KTE on images
    gamma_k = _rbf_gamma_on_images(Y2d, T)

    # CADR / Hadad work on scalar proxy: mean pixel
    y_scalar = Y2d.mean(axis=1)
    gamma_x = _gamma_on_X(X)

    m0_hat = fit_krr_predict(
        X[T == 0], y_scalar[T == 0], X,
        kernel_function="rbf", gamma=gamma_x, lam=1e-4
    ) if np.any(T == 0) else np.full(X.shape[0], y_scalar.mean())

    m1_hat = fit_krr_predict(
        X[T == 1], y_scalar[T == 1], X,
        kernel_function="rbf", gamma=gamma_x, lam=1e-4
    ) if np.any(T == 1) else np.full(X.shape[0], y_scalar.mean())

    p_logged = np.where(T == 1, w1, 1.0 - w1)

    try:
        t0 = time.time()
        if method == "VS-DR-KTE":
            stat = xMMD2_vsdr_fold_generic(
                Y=Y2d, w=w1, X=X, A=T,
                kernel_function="rbf",
                Pi_0_on_0=Pi_0_on_0, Pi_1_on_1=Pi_1_on_1,
                idx0=idx0, idx1=idx1,
                gamma=gamma_k, lam=1e-2,
            )
            pval = 0.5 * (1.0 - st.erf(stat / np.sqrt(2.0)))

        elif method == "CADR":
            out = cadr_test(
                X=X, A=T, Y=y_scalar,
                p_realized=p_logged,
                m0=m0_hat, m1=m1_hat,
                P_all=P_all
            )
            stat = out["stat"]
            pval = 0.5 * (1.0 - st.erf(stat / np.sqrt(2.0)))

        elif method == "Hadad":
            out = hadad_test(
                A=T, Y=y_scalar, p=p_logged,
                m0=m0_hat, m1=m1_hat,
                scheme="two_point", alpha=0.7,
                p_min=0.0, estimator="aipw"
            )
            stat = out["stat"]
            pval = 0.5 * (1.0 - st.erf(stat / np.sqrt(2.0)))
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - t0
    except Exception:
        stat, pval, elapsed = np.nan, np.nan, 0.0

    return {
        "p_value": float(pval),
        "stat": float(stat),
        "time": float(elapsed),
        "scenario": scenario_id,
        "method": method,
        "seed": int(seed),
    }

def get_parameters_experiment():
    seeds = list(range(NB_SEEDS))
    with open(PARAM_CSV, "w") as f:
        f.write("line,scenario,method,seed\n")
        line = 0
        for scenario in SCENARIOS:
            for method in METHODS:
                for seed in seeds:
                    f.write(f"{line},{scenario},{method},{seed}\n")
                    line += 1
    print(f"Parameter file written to {PARAM_CSV}")

def run_from_arguments(parameters):
    res = run_single_experiment(
        scenario_id=parameters.scenario,
        method=parameters.method,
        seed=parameters.seed,
    )
    df = pd.DataFrame([res])
    fname = f"{RESULT_DIR}/scenario{parameters.scenario}_{parameters.method}_seed{parameters.seed}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}")

def get_results_table():
    alpha = 0.05
    seeds = range(NB_SEEDS)

    rejection_matrix = np.zeros((len(SCENARIOS), len(METHODS)))
    for i, scenario in enumerate(SCENARIOS):
        for j, method in enumerate(METHODS):
            vals = []
            for seed in seeds:
                fp = f"{RESULT_DIR}/scenario{scenario}_{method}_seed{seed}.csv"
                try:
                    df = pd.read_csv(fp)
                    vals.append((df["p_value"] < alpha).mean())
                except FileNotFoundError:
                    continue
            rejection_matrix[i, j] = np.mean(vals) if len(vals) else np.nan

    df_reject = pd.DataFrame(rejection_matrix, index=SCENARIOS, columns=METHODS)
    print(df_reject)

    latex_path = os.path.join(RESULT_DIR, "rejection_table.tex")
    with open(latex_path, "w") as f:
        f.write(df_reject.T.to_latex(float_format="{:0.2f}".format))
    print(f"LaTeX table saved to {latex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--get_parameters_experiment", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--scenario", type=str, choices=SCENARIOS, default="I")
    parser.add_argument("--method", type=str, choices=METHODS, default="VS-DR-KTE")
    parser.add_argument("--seed", type=int, default=0)

    parameters = parser.parse_args()

    if parameters.get_parameters_experiment:
        get_parameters_experiment()
    if parameters.run:
        run_from_arguments(parameters)
    if parameters.results:
        get_results_table()
