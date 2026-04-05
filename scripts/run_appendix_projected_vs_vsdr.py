from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from paper_experiments import collect_epsilon_greedy, cosine_base, plot_power
from paper_experiments.runner import run_experiment_grid


def main():
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sample_sizes = list(range(100, 450, 50))
    output_dir = "results/appendix_projected_vs_vsdr"

    collector = lambda ns, scenario, rng: collect_epsilon_greedy(
        ns=ns,
        d=5,
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
    )

    run_experiment_grid(
        output_dir,
        collector,
        ["I", "II", "III", "IV"],
        sample_sizes,
        ["PADR-KTE", "VS-DR-KTE"],
        num_experiments=200,
    )
    plot_power(
        output_dir,
        sample_sizes,
        ["PADR-KTE", "VS-DR-KTE"],
        "figures/appendix_projected_vs_vsdr_power.png",
        "Projected vs legacy",
    )


if __name__ == "__main__":
    main()
