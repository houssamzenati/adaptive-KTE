from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from paper_experiments import (
    collect_epsilon_greedy,
    cosine_base,
    load_ihdp_features,
    plot_calibration,
    plot_power,
)
from paper_experiments.plotting import save_latex_table
from paper_experiments.runner import run_experiment_grid, summarize_rejections


def main():
    ihdp_path = Path("data/ihdp.csv")
    if not ihdp_path.exists():
        raise FileNotFoundError("Missing data/ihdp.csv")

    X_all, _ = load_ihdp_features(ihdp_path)
    beta_vec = np.ones(X_all.shape[1])
    sample_sizes = list(range(100, 901, 50)) + [908]
    output_dir = "results/main_ihdp"

    collector = lambda ns, scenario, rng: collect_epsilon_greedy(
        ns=ns,
        d=X_all.shape[1],
        beta_vec=beta_vec,
        noise_std=0.5,
        scenario=scenario,
        base_fn=cosine_base,
        eps0=0.5,
        eps_min=0.2,
        power=0.5,
        lam=1e-2,
        rng=rng,
        split="chronological",
        X_all=X_all,
    )

    run_experiment_grid(output_dir, collector, ["I"], sample_sizes, ["PADR-KTE"], num_experiments=200)
    run_experiment_grid(
        output_dir,
        collector,
        ["II", "III", "IV"],
        sample_sizes,
        ["PADR-KTE", "CADR", "AW-AIPW"],
        num_experiments=200,
    )

    plot_calibration(
        output_dir,
        sample_sizes,
        "PADR-KTE",
        "figures/REVIEWED_null_dr_adaptive_IHDP1_errorbar.png",
        "IHDP",
    )
    plot_power(
        output_dir,
        sample_sizes,
        ["PADR-KTE", "CADR", "AW-AIPW"],
        "figures/scenarios_II_III_IV_adaptive_ihdp_power.png",
        "IHDP",
    )

    summary = summarize_rejections(output_dir, ["II", "III", "IV"], [908], ["PADR-KTE", "CADR", "AW-AIPW"])
    summary["cell"] = summary.apply(lambda row: f"${row.rejection_rate:.2f} \\pm {row.se95:.2f}$", axis=1)
    save_latex_table(
        summary,
        row_key="method",
        column_key="scenario",
        value_key="cell",
        save_path="results/main_ihdp/ihdp_table.tex",
    )
    summary.to_csv("results/main_ihdp/ihdp_table.csv", index=False)


if __name__ == "__main__":
    main()
