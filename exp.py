# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

from xkte_nonadaptive import xMMD2dr          # DR-xKTE (non-stabilized)
from dr_kte_adaptive import xMMD2_vsdr_fold_generic  # <-- VS-DR-xKTE (self-normalized)  # NEW
import scipy.stats as st

# ------------------------ data + policy helpers ------------------------ #
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
    f = X @ v            # baseline signal
    Delta = delta_scale * (X @ u)   # TE; set delta_scale=0 for H0

    m0 = f
    m1 = f + Delta
    eps = rng.normal(scale=sigma_eps, size=n)

    return X, m0, m1, Delta, eps, u


def policy_explore_commit(x, t, t0, eps, u, winner):
    """
    Epsilon explore-then-commit with linear separator on x @ u.

    parameters
    ----------
    x : (d,) array
    t : int
    t0 : int
    eps : float
    u : (d,) array
    winner : int in {0,1}

    returns
    -------
    p1 : float
        Probability of action 1 at time t given context x.
    """
    if t < t0:
        return 0.5
    inS = (x @ u) > 0
    prefer1 = (winner == 1 and inS) or (winner == 0 and not inS)
    return (1.0 - eps) if prefer1 else eps


def simulate_ate_run_for_rkhs(
    n=3000, d=5, t0=40, eps=0.005, sigma_eps=0.35, rng=None, delta_scale=1.0
):
    """
    Simulate adaptive ETC design and produce predictable propensities.

    returns
    -------
    X : (n,d)
    A : (n,)
        Realized actions.
    Y : (n,)
        Observed outcomes.
    p_pred : (n,)
        Predictable evaluation propensities for action 1 at time t.
    u : (d,)
        TE direction.
    winner : int
        Committed winner after t0.
    t0, eps : as passed
    """
    rng = rng or np.random.RandomState(42)
    X, m0, m1, Delta, eps_noise, u = make_data(
        n, d, rng, sigma_eps=sigma_eps, delta_scale=delta_scale
    )
    winner = 1 if rng.rand() < 0.5 else 0  # commit winner (predictable)

    # predictable propensities at decision time t for X_t
    p_pred = np.array(
        [policy_explore_commit(X[t], t, t0, eps, u, winner) for t in range(n)]
    )

    # actions drawn from p_pred; outcomes with noise
    A = (rng.rand(n) < p_pred).astype(int)
    Y = np.where(A == 1, m1, m0) + eps_noise

    return X, A, Y, p_pred, u, winner, t0, eps


# ------------------ helpers for VS-DR-xKTE (folds + Pi) ----------------- #
def chronological_folds(n):
    """Split indices into two chronological halves."""
    cut = n // 2
    idx0 = np.arange(cut)
    idx1 = np.arange(cut, n)
    return idx0, idx1


def rejection_table(Z_dr, Z_vs, alphas=(0.1, 0.05, 0.01)):
    rows = []
    for alpha in alphas:
        # one-sided test: reject H0 if Z > z_{1-alpha}
        crit = st.norm.ppf(1 - alpha)
        rej_dr = np.mean(Z_dr > crit)
        rej_vs = np.mean(Z_vs > crit)
        rows.append((alpha, crit, rej_dr, rej_vs))
    return rows

def build_Pi_matrices(X, u, winner, t0, eps, split="chronological"):
    """
    Build Pi_0_on_0 and Pi_1_on_1 for VS-DR-xKTE, using the same ETC policy.

    For each row t in fold k, we evaluate π_t(A=1 | X_s) on all contexts X_s
    from that fold, matching the interface expected by xMMD2_vsdr_fold_generic.
    """
    n = X.shape[0]
    if split == "chronological":
        idx0, idx1 = chronological_folds(n)
    else:
        idx0 = np.arange(0, n, 2)
        idx1 = np.arange(1, n, 2)

    Z0 = X[idx0]
    Z1 = X[idx1]
    N0, N1 = len(idx0), len(idx1)

    Pi_0_on_0 = np.empty((N0, N0))
    for r, t in enumerate(idx0):
        # predict propensities at time t for all contexts in fold 0
        Pi_0_on_0[r] = [
            policy_explore_commit(Z0[s], t, t0, eps, u, winner) for s in range(N0)
        ]

    Pi_1_on_1 = np.empty((N1, N1))
    for r, t in enumerate(idx1):
        Pi_1_on_1[r] = [
            policy_explore_commit(Z1[s], t, t0, eps, u, winner) for s in range(N1)
        ]

    return Pi_0_on_0, Pi_1_on_1, idx0, idx1


# --------------------------- core: DR & VS-DR --------------------------- #
def run_DR_xKTE_sim(
    R=300,
    n=3000,
    d=5,
    t0=40,
    eps=0.005,
    sigma_eps=0.35,
    seed=123,
    delta_scale=0.0,
    kernel_function="rbf",
):
    """
    Monte Carlo for Z-stat of DR-xKTE (non-stabilized) and VS-DR-xKTE
    using true predictable props under an ETC design.

    returns
    -------
    Z_dr : (R,)
        Z-statistics of DR-xKTE over repetitions.
    Z_vs : (R,)
        Z-statistics of VS-DR-xKTE over repetitions.
    """
    rng_master = np.random.RandomState(seed)
    Z_dr = np.zeros(R)
    Z_vs = np.zeros(R)

    for r in tqdm(range(R)):
        rng = np.random.RandomState(rng_master.randint(1, 10**9))
        X, A, Y, p_pred, u, winner, t0_run, eps_run = simulate_ate_run_for_rkhs(
            n=n,
            d=d,
            t0=t0,
            eps=eps,
            sigma_eps=sigma_eps,
            rng=rng,
            delta_scale=delta_scale,
        )
        Y2d = Y[:, None]

        # outcome RBF bandwidth by median heuristic (fallback to var)
        YY0 = Y2d[A == 0]
        YY1 = Y2d[A == 1]
        sigma2 = np.median(pairwise_distances(YY0, YY1, metric="euclidean")) ** 2
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.var(Y2d)) + 1e-6
        gamma = 1.0 / sigma2

        # 1) DR-xKTE (non-stabilized) with true predictable props
        Z_dr[r] = xMMD2dr(
            Y2d, p_pred, X, A, kernel_function=kernel_function, 
            lam=1e-3,
            gamma=gamma
        )

        # 2) VS-DR-xKTE (variance-stabilized) with the same predictable props
        Pi_0_on_0, Pi_1_on_1, idx0, idx1 = build_Pi_matrices(
            X, u, winner, t0_run, eps_run, split="chronological"
        )
        Z_vs[r] = xMMD2_vsdr_fold_generic(
            Y,
            p_pred,
            X,
            A,
            kernel_function=kernel_function,
            Pi_0_on_0=Pi_0_on_0,
            Pi_1_on_1=Pi_1_on_1,
            idx0=idx0,
            idx1=idx1,
            gamma=gamma,
            lam=1e-3,
        )

    rows = rejection_table(Z_dr, Z_vs)
    for alpha, crit, rej_dr, rej_vs in rows:
        print(f"alpha={alpha:.2f}, z_crit={crit:.3f}, "
            f"rej_DR={rej_dr:.3f}, rej_VS-DR={rej_vs:.3f}")

    return Z_dr, Z_vs


# ----------------------------- plotting only ---------------------------- #
def plot_DR_hist(Z_dr, save_path="dr_hist.png", title="Z: DR-xKTE"):
    """
    Plot histogram of DR-xKTE Z-stat with N(0,1) overlay.
    """

    def overlay(ax, lo, hi):
        xs = np.linspace(lo, hi, 500)
        ax.plot(
            xs,
            np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            linestyle="--",
            color="k",
            linewidth=1.8,
            label="N(0,1)",
        )

    lo, hi = np.min(Z_dr), np.max(Z_dr)
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)

    plt.figure(figsize=(7, 4))
    plt.hist(Z_dr, bins=bins, density=True, alpha=0.75, label="DR-xKTE")
    overlay(plt.gca(), bins[0], bins[-1])
    plt.title(title)
    plt.xlabel("Z")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_DR_vs_hist(Z_dr, Z_vs, save_path="dr_vs_vsdr_hist.png",
                    title="Z: DR-xKTE vs VS-DR-xKTE"):
    """
    Plot histograms of DR-xKTE and VS-DR-xKTE with N(0,1) overlay.
    """

    def overlay(ax, lo, hi):
        xs = np.linspace(lo, hi, 500)
        ax.plot(
            xs,
            np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            linestyle="--",
            color="k",
            linewidth=1.8,
            label="N(0,1)",
        )

    lo = min(np.min(Z_dr), np.min(Z_vs))
    hi = max(np.max(Z_dr), np.max(Z_vs))
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)

    plt.figure(figsize=(7, 4))
    plt.hist(Z_dr, bins=bins, density=True, alpha=0.5, label="DR-xKTE")
    plt.hist(Z_vs, bins=bins, density=True, alpha=0.5, label="VS-DR-xKTE")
    overlay(plt.gca(), bins[0], bins[-1])
    plt.title(title)
    plt.xlabel("Z")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
# Example usage: run under H0 (Δ = 0) with ETC design, then plot DR vs VS-DR
if __name__ == "__main__":
    Z_dr, Z_vs = run_DR_xKTE_sim(
        R=200,
        n=500,
        d=5,
        t0=20,
        eps=0.1,
        sigma_eps=1.0,
        seed=123,
        delta_scale=0.0,        # H0
        kernel_function="rbf",
    )
    plot_DR_vs_hist(Z_dr, Z_vs,
                    save_path="dr_vs_vsdr_hist.png",
                    title="Z: DR-xKTE vs VS-DR-xKTE (H0)")

# %%
