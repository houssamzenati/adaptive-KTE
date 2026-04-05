from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from .runner import load_result_frame, summarize_rejections


def _savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)


def plot_calibration(output_dir, sample_sizes, method, save_path, title, alpha=0.05):
    summary = summarize_rejections(output_dir, ["I"], sample_sizes, [method], alpha=alpha)
    largest_n = max(sample_sizes)
    frame = load_result_frame(output_dir, "I", largest_n, method)
    stats = frame["stat"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    lo, hi = stats.min(), stats.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 30)
    axes[0].hist(stats, bins=bins, density=True, alpha=0.75, color="#1b9e77")
    xs = np.linspace(bins[0], bins[-1], 400)
    axes[0].plot(xs, st.norm.pdf(xs), linestyle="--", color="black", linewidth=1.5)
    axes[0].set_title(f"{title}: histogram")
    axes[0].set_xlabel("Statistic")

    st.probplot(stats, dist="norm", plot=axes[1])
    axes[1].set_title(f"{title}: Q-Q plot")

    axes[2].errorbar(
        summary["sample_size"],
        summary["rejection_rate"],
        yerr=summary["se95"],
        marker="o",
        linestyle="--",
        color="#d95f02",
        capsize=4,
    )
    axes[2].axhline(alpha, linestyle=":", color="black", linewidth=1)
    axes[2].set_title(f"{title}: type-I error")
    axes[2].set_xlabel("Sample size")
    axes[2].set_ylabel("Rejection rate")
    axes[2].set_ylim(-0.02, 0.25)

    fig.tight_layout()
    _savefig(save_path)
    plt.close(fig)


def plot_power(output_dir, sample_sizes, methods, save_path, title_prefix, alpha=0.05):
    summary = summarize_rejections(output_dir, ["II", "III", "IV"], sample_sizes, methods, alpha=alpha)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    colors = {
        "PADR-KTE": "#1b9e77",
        "VS-DR-KTE": "#d95f02",
        "CADR": "#7570b3",
        "AW-AIPW": "#e7298a",
        "DR-xKTE": "#66a61e",
        "KTE": "#e6ab02",
    }
    labels = {"AW-AIPW": "AW-AIPW"}

    for ax, scenario in zip(axes, ["II", "III", "IV"]):
        scenario_rows = summary[summary["scenario"] == scenario]
        for method in methods:
            method_rows = scenario_rows[scenario_rows["method"] == method]
            ax.errorbar(
                method_rows["sample_size"],
                method_rows["rejection_rate"],
                yerr=method_rows["se95"],
                marker="o",
                linestyle="--",
                capsize=4,
                color=colors.get(method, None),
                label=labels.get(method, method),
            )
        ax.set_title(f"{title_prefix} Scenario {scenario}")
        ax.set_xlabel("Sample size")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Rejection rate")
    axes[0].legend(loc="best")
    _savefig(save_path)
    plt.close(fig)


def save_rejection_table(output_dir, scenarios, sample_sizes, methods, save_path, alpha=0.05):
    summary = summarize_rejections(output_dir, scenarios, sample_sizes, methods, alpha=alpha)
    summary.to_csv(save_path, index=False)
    return summary


def save_latex_table(summary, row_key, column_key, value_key, save_path, formatter=None):
    table = summary.pivot(index=row_key, columns=column_key, values=value_key)
    if formatter is not None:
        table = table.applymap(formatter)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as handle:
        handle.write(table.to_latex(escape=False))
