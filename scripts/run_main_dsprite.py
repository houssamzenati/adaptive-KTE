from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dsprite_adaptive import collect_adaptive_kte_dsprite
from paper_experiments.plotting import save_latex_table
from paper_experiments.runner import run_experiment_grid, summarize_rejections


def main():
    output_dir = "results/main_dsprite"
    collector = lambda ns, scenario, rng: collect_adaptive_kte_dsprite(
        ns=ns,
        d=2,
        scenario=scenario,
        eps0=0.5,
        eps_min=0.2,
        power=0.5,
        lam=1e-2,
        rng=rng,
        split="chronological",
        shift_pixels=6,
    )

    run_experiment_grid(output_dir, collector, ["I", "IV"], [1000], ["PADR-KTE", "CADR", "AW-AIPW"], num_experiments=200)
    summary = summarize_rejections(output_dir, ["I", "IV"], [1000], ["PADR-KTE", "CADR", "AW-AIPW"])
    summary["cell"] = summary.apply(lambda row: f"${row.rejection_rate:.2f} \\pm {row.se95:.2f}$", axis=1)
    save_latex_table(
        summary,
        row_key="method",
        column_key="scenario",
        value_key="cell",
        save_path="results/main_dsprite/dsprite_table.tex",
    )
    summary.to_csv("results/main_dsprite/dsprite_table.csv", index=False)


if __name__ == "__main__":
    main()
