# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------- core helpers ----------------
def make_data(n, d, rng, sigma_eps=0.4):
    X = rng.randn(n, d)
    u = rng.randn(d)
    u /= np.linalg.norm(u) + 1e-12
    v = rng.randn(d)
    v /= np.linalg.norm(v) + 1e-12
    lin = X @ v
    phi = np.cos(X @ u + 0.8)
    f = 0.6 * lin + 0.8 * phi + 0.3 * (lin * phi)
    Delta = 1.0 + 0.7 * np.tanh(lin)
    m0 = f
    m1 = f + Delta
    eps = sigma_eps * rng.randn(n)
    return X, m0, m1, Delta, eps


def policy_explore_commit(x, t, t0, eps, u, winner):
    if t < t0:
        return 0.5
    inS = (x @ u) > 0
    if inS:
        return 1 - eps if winner == 1 else eps
    return eps if winner == 1 else 1 - eps


def clip01(p, p_min):
    return np.minimum(1.0 - p_min, np.maximum(p_min, p))


def dr_estimate(Y, A, p, m0_hat, m1_hat):
    return np.mean(
        (m1_hat - m0_hat) + (A / p) * (Y - m1_hat) - ((1 - A) / (1 - p)) * (Y - m0_hat)
    )


# ------------- main sim: ONLY DR with realized p_t -------------
def simulate_DR_realized_with_Z(
    n=4000,
    d=5,
    t0=40,
    eps=0.005,
    sigma_eps=0.4,
    rng=None,
    p_min=1e-5,
):
    rng = rng or np.random.RandomState(42)
    X, m0, m1, _, eps_noise = make_data(n, d, rng, sigma_eps=sigma_eps)

    u = rng.randn(d)
    u /= np.linalg.norm(u) + 1e-12
    winner = 1 if rng.rand() < 0.5 else 0

    p = np.empty(n)
    A = np.empty(n, int)
    Y = np.empty(n)
    for t in range(n):
        p[t] = policy_explore_commit(X[t], t, t0, eps, u, winner)
        A[t] = (rng.rand() < p[t]).astype(int)
        Y[t] = (m1[t] if A[t] else m0[t]) + eps_noise[t]

    # clip propensities as before
    p_clipped = clip01(p, p_min)

    tau_true = np.mean(m1 - m0)

    # DR with realized (clipped) propensities p_t
    tau_dr = dr_estimate(Y, A, p_clipped, m0, m1)

    psi_dr = (
        (m1 - m0)
        + (A / p_clipped) * (Y - m1)
        - ((1 - A) / (1 - p_clipped)) * (Y - m0)
        - tau_true
    )
    se_dr = np.sqrt(np.var(psi_dr, ddof=1) / n)
    Z_dr = (tau_dr - tau_true) / (se_dr + 1e-12)

    return Z_dr


# ------------- driver: compare to N(0,1) -------------
def compare_to_standard_normal(
    R=200, n=1000, d=5, t0=10, eps=0.001, sigma_eps=0.35, seed=123
):
    rng = np.random.RandomState(seed)
    Z = np.zeros(R)

    for r in tqdm(range(R)):
        Z[r] = simulate_DR_realized_with_Z(
            n=n,
            d=d,
            t0=t0,
            eps=eps,
            sigma_eps=sigma_eps,
            rng=np.random.RandomState(rng.randint(1, 10**9)),
            p_min=1e-5,
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    lo, hi = Z.min(), Z.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    ax.hist(Z, bins=bins, density=True, alpha=0.7, label="DR (realized p_t)")

    xs = np.linspace(bins[0], bins[-1], 500)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), lw=1.8, label="N(0,1)")

    ax.set_title("Z = studentized DR with realized propensities")
    ax.set_xlabel("Z")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


compare_to_standard_normal(
    R=200, n=1000, d=5, t0=500, eps=0.001, sigma_eps=0.35, seed=123
)
# %%
