# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from xkte_nonadaptive import xMMD2dr  # DR-xKTE (non-stabilized)
from dr_kte_adaptive import xMMD2_vsdr_fold_generic

# ------------------------ data + policy helpers ------------------------ #
import numpy as np

def make_data(n, d, rng, sigma_eps=0.4, delta_scale=1.0):
    """
    Simple linear DGP.

    parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    rng : np.random.RandomState
        Random generator (use a seeded instance for reproducibility).
    sigma_eps : float, optional
        Noise std for outcomes.
    delta_scale : float, optional
        Scales the treatment effect; 0.0 => null (H0).

    returns
    -------
    X : (n, d) array
    m0 : (n,) array
        Baseline (control) regression f(X).
    m1 : (n,) array
        Treated regression f(X) + Delta(X).
    Delta : (n,) array
        Heterogeneous treatment effect.
    eps : (n,) array
        Observation noise.
    u : (d,) array
        Unit vector defining the TE direction.
    """
    X = rng.normal(size=(n, d))

    # Unit directions for baseline and treatment effect
    v = rng.normal(size=d)
    v /= max(np.linalg.norm(v), 1e-12)

    u = rng.normal(size=d)
    u /= max(np.linalg.norm(u), 1e-12)

    # Linear baseline and linear heterogeneous treatment effect
    f = X @ v                                   # baseline signal
    Delta = delta_scale * (X @ u)               # TE; set delta_scale=0 for H0

    m0 = f
    m1 = f + Delta
    eps = rng.normal(scale=sigma_eps, size=n)

    return X, m0, m1, Delta, eps, u


def policy_explore_commit(x, t, t0, eps, u, winner):
    if t < t0:
        return 0.5
    inS = (x @ u) > 0
    prefer1 = (winner == 1 and inS) or (winner == 0 and not inS)
    return (1.0 - eps) if prefer1 else eps

def simulate_ate_run_for_rkhs(
    n=3000, d=5, t0=40, eps=0.005, sigma_eps=0.35, rng=None, delta_scale=1.0
):
    """
    Returns everything needed to construct fold-wise evaluation policy matrices.
    """
    rng = rng or np.random.RandomState(42)
    X, m0, m1, Delta, eps_noise, u = make_data(
        n, d, rng, sigma_eps=sigma_eps, delta_scale=delta_scale
    )
    winner = 1 if rng.rand() < 0.5 else 0  # commit winner (predictable)

    # predictable propensities at decision time t for X_t
    p_pred = np.array([policy_explore_commit(X[t], t, t0, eps, u, winner)
                       for t in range(n)])

    # actions drawn from p_pred; outcomes with noise
    A = (rng.rand(n) < p_pred).astype(int)
    Y = np.where(A == 1, m1, m0) + eps_noise

    return X, A, Y, p_pred, u, winner, t0, eps

# ---------------------- fold-wise policy matrices ---------------------- #
def alternating_folds(n):
    idx0 = np.arange(0, n, 2)  # 0,2,4,... (odd in 1-based)
    idx1 = np.arange(1, n, 2)  # 1,3,5,...
    return idx0, idx1

def build_Pi_fold_on_fold(X, fold_idx, u, winner, t0, eps):
    """
    Build Π_{fold←fold} in chronological order within the fold:
      Rows correspond to evaluation *times t* (GLOBAL time indices) in this fold.
      Columns correspond to *contexts in the same fold*, ordered chronologically.
    """
    nF = fold_idx.size
    Pi = np.empty((nF, nF), dtype=float)
    # global times for each row; contexts for each column
    for r, t_global in enumerate(fold_idx):
        # evaluation policy at time t_global applied to ALL contexts in the fold
        x_block = X[fold_idx]
        Pi[r] = np.array([
            policy_explore_commit(x_block[j], t_global, t0, eps, u, winner)
            for j in range(nF)
        ])
    return Pi

# ----------------------------- main compare ---------------------------- #
def compare_DR_vs_VSDR(
    R=300, n=3000, d=5, t0=40, eps=0.005, sigma_eps=0.35,
    seed=123, delta_scale=0.0, split="alternating", kernel_function='rbf'
):
    rng_master = np.random.RandomState(seed)
    Z_dr = np.zeros(R)    # DR-xKTE with true predictable propensities
    Z_vs = np.zeros(R)    # VS-DR-KTE with fold-wise stabilization

    for r in tqdm(range(R)):
        rng = np.random.RandomState(rng_master.randint(1, 10**9))
        X, T, Y, p_pred, u, winner, t0_run, eps_run = simulate_ate_run_for_rkhs(
            n=n, d=d, t0=t0, eps=eps, sigma_eps=sigma_eps, rng=rng,
            delta_scale=delta_scale
        )
        Y2d = Y[:, None]

        # outcome RBF bandwidth (same recipe)
        YY0 = Y2d[T == 0]; YY1 = Y2d[T == 1]
        sigma2 = np.median(pairwise_distances(YY0, YY1, metric="euclidean"))**2
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.var(Y2d)) + 1e-6
        gamma = 1.0 / sigma2

        # ---- DR-xKTE (non-stabilized) with true predictable props
        Z_dr[r] = xMMD2dr(Y2d, p_pred, X, T, kernel_function=kernel_function, gamma=gamma)

        # ---- VS-DR-KTE: build fold-wise Π matrices in chronological order
        if split == "alternating":
            idx0, idx1 = alternating_folds(n)
        elif split == "chronological":
            cut = n // 2
            idx0 = np.arange(0, cut)
            idx1 = np.arange(cut, n)
        else:
            raise ValueError("split must be 'alternating' or 'chronological'.")

        Pi_0_on_0 = build_Pi_fold_on_fold(X, idx0, u, winner, t0_run, eps_run)
        Pi_1_on_1 = build_Pi_fold_on_fold(X, idx1, u, winner, t0_run, eps_run)

        Z_vs[r] = xMMD2_vsdr_fold_generic(
            Y=Y2d,
            w=p_pred,
            X=X,
            A=T,
            kernel_function=kernel_function,
            Pi_0_on_0=Pi_0_on_0,
            Pi_1_on_1=Pi_1_on_1,
            idx0=idx0,
            idx1=idx1,
            gamma=gamma,   # outcome-kernel parameter
        )

    # ---------- plotting ----------
    def overlay(ax, lo, hi):
        xs = np.linspace(lo, hi, 500)
        ax.plot(
            xs,
            np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            linestyle="--", color="k", linewidth=1.8, label="N(0,1)"
        )

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharex=False)

    lo, hi = Z_dr.min(), Z_dr.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    axs[0].hist(Z_dr, bins=bins, density=True, alpha=0.75,
                label="DR-xKTE")
    overlay(axs[0], bins[0], bins[-1])
    axs[0].set_title("Z: DR-xKTE")
    axs[0].set_xlabel("Z"); axs[0].grid(alpha=0.3); axs[0].legend()

    lo, hi = Z_vs.min(), Z_vs.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    axs[1].hist(Z_vs, bins=bins, density=True, alpha=0.75,
                label="VS-DR-KTE")
    overlay(axs[1], bins[0], bins[-1])
    axs[1].set_title("Z: VS-DR-KTE")
    axs[1].set_xlabel("Z"); axs[1].grid(alpha=0.3); axs[1].legend()
    fig.savefig("dr_vs_vsdr_hist.png", dpi=300, bbox_inches="tight")
    plt.tight_layout(); plt.show()

# Example: run under H0 (Δ = 0) with ETC design
compare_DR_vs_VSDR(
    R=300, n=700, d=5, t0=15, eps=0.005, sigma_eps=1,
    seed=123, delta_scale=0.0, split="alternating", kernel_function='poly'
)

# %%
