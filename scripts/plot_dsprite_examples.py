from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from dsprite_adaptive import (
    _get_base_heart,
    _load_dsprite,
    _render_image_roll_quadrant,
    collect_adaptive_kte_dsprite,
)


def save_observational_examples():
    rng = np.random.RandomState(0)
    _, _, Y, _, _, _, _, _, _ = collect_adaptive_kte_dsprite(
        ns=12,
        d=2,
        scenario="IV",
        rng=rng,
        split="chronological",
    )
    fig, axes = plt.subplots(3, 4, figsize=(6, 4.5))
    for ax, img in zip(axes.flat, Y.reshape(-1, 64, 64)):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("figures/observational_dsprite.png", dpi=300, bbox_inches="tight")


def save_counterfactual_pairs():
    imgs, latents_bases = _load_dsprite()
    base = _get_base_heart(imgs, latents_bases)
    U = np.array([[0.2, 0.2], [0.4, 0.65], [0.7, 0.3], [0.8, 0.7]])
    Y0 = _render_image_roll_quadrant(U, np.zeros(len(U), dtype=int), base, "IV")
    Y1 = _render_image_roll_quadrant(U, np.ones(len(U), dtype=int), base, "IV")

    fig, axes = plt.subplots(len(U), 2, figsize=(3, 6))
    for row in range(len(U)):
        axes[row, 0].imshow(Y0[row], cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(Y1[row], cmap="gray")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig("figures/counterfactual_pairs_dsprite.png", dpi=300, bbox_inches="tight")


def main():
    save_observational_examples()
    save_counterfactual_pairs()


if __name__ == "__main__":
    main()
