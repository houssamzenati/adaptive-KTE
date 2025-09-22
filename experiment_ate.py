# %%
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


def Dprime_ATE(p, a, y, q0, q1, p_min=1e-6):
    p = clip01(p, p_min)
    return (q1 - q0) + (a / p) * (y - q1) - ((1 - a) / (1 - p)) * (y - q0)


def dr_estimate(Y, A, p, m0_hat, m1_hat):
    return np.mean(
        (m1_hat - m0_hat) + (A / p) * (Y - m1_hat) - ((1 - A) / (1 - p)) * (Y - m0_hat)
    )


# ------------- main sim with clipping/trimming -------------
def simulate_EC_DR_with_Z(
    n=4000,
    d=5,
    t0=40,
    eps=0.005,
    sigma_eps=0.4,
    rng=None,
    # new knobs:
    p_min=0.02,  # floor/ceiling for all propensities
    r_max=50.0,  # cap for importance ratios g_t/g_s
    window=None,  # sliding window size for σ̂_t^2 (None => all past)
    min_arm=6,  # need this many per arm in window to form σ̂_t^2
    weight_trim_q=99.0,  # drop top-q% of w after construction
):
    rng = rng or np.random.RandomState(42)
    X, m0, m1, _, eps_noise = make_data(n, d, rng, sigma_eps=sigma_eps)

    u = rng.randn(d)
    u /= np.linalg.norm(u) + 1e-12
    winner = 1 if rng.rand() < 0.5 else 0

    # generate actions/outcomes
    p = np.empty(n)
    A = np.empty(n, int)
    Y = np.empty(n)
    for t in range(n):
        p[t] = policy_explore_commit(X[t], t, t0, eps, u, winner)
        A[t] = (rng.rand() < p[t]).astype(int)
        Y[t] = (m1[t] if A[t] else m0[t]) + eps_noise[t]

    # clipped realized propensities
    # p = clip01(p_raw, p_min)
    tau_true = np.mean(m1 - m0)

    # ---- DR panels (unchanged logic, but with p clipped) ----
    tau_bad = dr_estimate(Y, A, p, m0, m1)
    tau_eval = dr_estimate(Y, A, 0.5, m0, m1)

    psi_bad = (m1 - m0) + (A / p) * (Y - m1) - ((1 - A) / (1 - p)) * (Y - m0) - tau_true
    psi_eval = (m1 - m0) + (A / 0.5) * (Y - m1) - ((1 - A) / 0.5) * (Y - m0) - tau_true
    se_bad = np.sqrt(np.var(psi_bad, ddof=1) / n)
    se_eval = np.sqrt(np.var(psi_eval, ddof=1) / n)
    Z_bad = (tau_bad - tau_true) / (se_bad + 1e-12)
    Z_eval = (tau_eval - tau_true) / (se_eval + 1e-12)

    # ---- CADR pieces ----
    D = Dprime_ATE(p, A, Y, m0, m1, p_min=p_min)  # stabilized score at t
    w = np.zeros(n)  # σ̂_t^{-1}

    W = n if window is None else int(window)
    for t in range(1, n):
        s0 = 0 if window is None else max(0, t - W)
        idx = np.arange(s0, t)
        # arm coverage check
        if (np.sum(A[idx] == 0) < min_arm) or (np.sum(A[idx] == 1) < min_arm):
            w[t] = 0.0
            continue

        # g_t(1|X_s) for s∈window, clipped
        pt_on_xs = np.array(
            [policy_explore_commit(X[s], t, t0, eps, u, winner) for s in idx]
        )
        pt_on_xs = clip01(pt_on_xs, p_min)
        gs_on_xs = clip01(p[idx], p_min)

        # importance ratios r = g_t/g_s at (A_s, X_s), with ratio clipping
        num = np.where(A[idx] == 1, pt_on_xs, 1.0 - pt_on_xs)
        den = np.where(A[idx] == 1, gs_on_xs, 1.0 - gs_on_xs)
        r = num / (den + 1e-12)
        if np.isfinite(r_max):
            r = np.minimum(r, r_max)

        # D'(g_t,·) evaluated at O_s using same plug-in (m0,m1)
        Ds_t = Dprime_ATE(pt_on_xs, A[idx], Y[idx], m0[idx], m1[idx], p_min=p_min)

        m1_hat = np.mean(r * (Ds_t**2))
        m2_hat = (np.mean(r * Ds_t)) ** 2
        var_t = m1_hat - m2_hat
        w[t] = 0.0 if var_t <= 0.0 else 1.0 / np.sqrt(var_t + 1e-12)

    # gentle trimming of tail weights (keeps your behavior)
    if weight_trim_q is not None:
        cutoff = np.percentile(w, weight_trim_q)
        keep = w < cutoff
        w = w[keep]
        D = D[keep]

    # CADR mean + weighted plug-in variance
    Wsum = w.sum()
    if Wsum <= 1e-12:
        cadr_mean = float(D.mean())
        cadr_var = float(np.var(D, ddof=1) / n)
    else:
        cadr_mean = float((w @ D) / Wsum)
        cadr_var = float(((w**2) @ (D - cadr_mean) ** 2) / (Wsum**2))

    Z_cadr = (cadr_mean - tau_true) / np.sqrt(cadr_var + 1e-12)
    return Z_bad, Z_eval, Z_cadr


# ------------- small driver (unchanged) -------------
def compare_to_standard_normal(
    R=200, n=1000, d=5, t0=10, eps=0.001, sigma_eps=0.35, seed=123
):
    rng = np.random.RandomState(seed)
    Zb = np.zeros(R)
    Ze = np.zeros(R)
    Zc = np.zeros(R)
    for r in tqdm(range(R)):
        Z_bad, Z_eval, Z_cadr = simulate_EC_DR_with_Z(
            n=n,
            d=d,
            t0=t0,
            eps=eps,
            sigma_eps=sigma_eps,
            rng=np.random.RandomState(rng.randint(1, 10**9)),
            # knobs (adjust if you like)
            p_min=0.02,
            r_max=50.0,
            window=None,
            min_arm=6,
            weight_trim_q=99.0,
        )
        Zb[r] = Z_bad
        Ze[r] = Z_eval
        Zc[r] = Z_cadr

    def overlay(ax, lo, hi):
        xs = np.linspace(lo, hi, 500)
        ax.plot(
            xs,
            np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            lw=1.8,
            label="standard Normal N(0,1)",
        )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=False)

    lo, hi = Zb.min(), Zb.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    axs[0].hist(Zb, bins=bins, density=True, alpha=0.7, label="DR (realized p_t)")
    overlay(axs[0], bins[0], bins[-1])
    axs[0].set_title("Z = studentized DR with realized propensities")
    axs[0].set_xlabel("Z")
    axs[0].set_ylabel("Density")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    lo, hi = Ze.min(), Ze.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    axs[1].hist(
        Ze,
        bins=bins,
        density=True,
        alpha=0.7,
        color="tab:orange",
        label="DR (predictable p)",
    )
    overlay(axs[1], bins[0], bins[-1])
    axs[1].set_title("Z = studentized DR with predictable propensities")
    axs[1].set_xlabel("Z")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    lo, hi = Zc.min(), Zc.max()
    bins = np.linspace(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2, 35)
    axs[2].hist(Zc, bins=bins, density=True, alpha=0.7, color="tab:green", label="CADR")
    overlay(axs[2], bins[0], bins[-1])
    axs[2].set_title("Z = studentized CADR")
    axs[2].set_xlabel("Z")
    axs[2].grid(alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


compare_to_standard_normal(
    R=200, n=1000, d=5, t0=10, eps=0.001, sigma_eps=0.35, seed=123
)
# %%
