from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from paper_experiments import collect_epsilon_greedy, collect_iid_logging, sigmoidal_base
from paper_experiments.plotting import save_latex_table
from paper_experiments.runner import run_experiment_grid, summarize_rejections


def _write_table(summary, save_path):
    summary["col"] = summary["scenario"] + "_n" + summary["sample_size"].astype(str)
    summary["cell"] = summary.apply(lambda row: f"${row.rejection_rate:.2f} \\pm {row.se95:.2f}$", axis=1)
    save_latex_table(summary, row_key="method", column_key="col", value_key="cell", save_path=save_path)


def main():
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sample_sizes = list(range(100, 550, 50))
    methods = ["PADR-KTE", "DR-xKTE", "KTE"]

    iid_output = "results/appendix_iid_baselines_iid"
    adaptive_output = "results/appendix_iid_baselines_adaptive"

    iid_collector = lambda ns, scenario, rng: collect_iid_logging(
        ns=ns,
        d=5,
        beta_vec=beta_vec,
        noise_std=0.1,
        scenario=scenario,
        base_fn=sigmoidal_base,
        rng=rng,
        split="chronological",
        policy="logistic",
    )
    adaptive_collector = lambda ns, scenario, rng: collect_epsilon_greedy(
        ns=ns,
        d=5,
        beta_vec=beta_vec,
        noise_std=0.1,
        scenario=scenario,
        base_fn=sigmoidal_base,
        eps0=0.5,
        eps_min=0.2,
        power=0.5,
        lam=1e-2,
        rng=rng,
        split="chronological",
    )

    run_experiment_grid(iid_output, iid_collector, ["I", "II"], sample_sizes, methods, num_experiments=200)
    run_experiment_grid(adaptive_output, adaptive_collector, ["I", "II"], sample_sizes, methods, num_experiments=200)

    iid_summary = summarize_rejections(iid_output, ["I", "II"], sample_sizes, methods)
    adaptive_summary = summarize_rejections(adaptive_output, ["I", "II"], sample_sizes, methods)
    iid_summary.to_csv("results/appendix_iid_baselines_iid/summary.csv", index=False)
    adaptive_summary.to_csv("results/appendix_iid_baselines_adaptive/summary.csv", index=False)
    _write_table(iid_summary, "results/appendix_iid_baselines_iid/table.tex")
    _write_table(adaptive_summary, "results/appendix_iid_baselines_adaptive/table.tex")


if __name__ == "__main__":
    main()
