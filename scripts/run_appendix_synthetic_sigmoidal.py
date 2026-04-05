from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from paper_experiments import collect_epsilon_greedy, plot_calibration, plot_power, sigmoidal_base
from paper_experiments.runner import run_experiment_grid


def main():
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    calibration_sizes = list(range(100, 1050, 50))
    power_sizes = list(range(100, 450, 50))
    output_dir = "results/appendix_sigmoidal_synthetic"

    collector = lambda ns, scenario, rng: collect_epsilon_greedy(
        ns=ns,
        d=5,
        beta_vec=beta_vec,
        noise_std=0.5,
        scenario=scenario,
        base_fn=sigmoidal_base,
        eps0=0.5,
        eps_min=0.2,
        power=0.5,
        lam=1e-1,
        rng=rng,
        split="chronological",
    )

    run_experiment_grid(output_dir, collector, ["I"], calibration_sizes, ["PADR-KTE"], num_experiments=200)
    run_experiment_grid(
        output_dir,
        collector,
        ["II", "III", "IV"],
        power_sizes,
        ["PADR-KTE", "CADR", "AW-AIPW"],
        num_experiments=200,
    )
    plot_calibration(
        output_dir,
        calibration_sizes,
        "PADR-KTE",
        "figures/REVIEWED_null_dr_adaptive_sigmoidal1_errorbar.png",
        "Sigmoidal synthetic",
    )
    plot_power(
        output_dir,
        power_sizes,
        ["PADR-KTE", "CADR", "AW-AIPW"],
        "figures/scenarios_II_III_IV_adaptive_sigmoidal_power.png",
        "Sigmoidal synthetic",
    )


if __name__ == "__main__":
    main()
