import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from scipy.special import erf

# Import methods
from xkte_nonadaptive import xMMD2dr  # DR-xKTE (non-stabilized)
from dr_kte_adaptive import xMMD2_vsdr_fold_generic  # VS-DR-KTE (stabilized)

# ------------------------ Adaptive DGP (Epsilon-Greedy from Paper) ------------------------ #
def treatment_effect_vector(ns, scenario, rng, beta_mix=2.0, beta_uniform=4.0):
    """Per-time treatment effect δ[t]; Y1 = base + δ, Y0 = base."""
    if scenario == 'I':
        return np.zeros(ns)
    if scenario == 'II':
        return np.full(ns, 2)
    if scenario == 'III':
        signs = rng.binomial(1, 0.5, size=ns) * 2 - 1
        return signs.astype(float) * beta_mix
    if scenario == 'IV':
        return rng.uniform(-beta_uniform, beta_uniform, size=ns)
    return np.zeros(ns)

def collect_epsilon_greedy(
    ns, d, beta_vec, noise_var, scenario,
    eps0=0.2, eps_min=0.05, power=0.99, lam=1e-2, rng=None,
    split="alternating"
):
    """ε-greedy (2 arms) with per-arm online ridge (no clipping)."""
    rng = rng or np.random.RandomState(0)

    # contexts and potential outcomes (cosine base)
    X = rng.randn(ns, d)
    base = np.cos(X @ beta_vec)
    delta = treatment_effect_vector(ns, scenario, rng)
    Y0 = base + noise_var * rng.randn(ns)
    Y1 = base + noise_var * rng.randn(ns) + delta

    # add bias feature and keep it unpenalized
    X_aug = np.hstack([np.ones((ns, 1)), X])  # [1, x]

    # online ridge state per arm (0 penalty on bias coord)
    S0 = np.diag([0.0] + [lam]*d); S1 = np.diag([0.0] + [lam]*d)
    b0 = np.zeros(d+1);            b1 = np.zeros(d+1)

    def solve_theta(S, b):
        try:
            return np.linalg.solve(S, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(S, b, rcond=None)[0]

    def eps_at(t):
        return max(eps_min, eps0 / ((t + 1) ** power))

    # logs + parameter snapshots
    T = np.zeros(ns, dtype=int)       # action (0/1)
    w = np.zeros(ns, dtype=float)     # π_t(1|X_t) at decision time t
    Y = np.zeros(ns, dtype=float)
    theta0_snap = np.zeros((ns, d+1))
    theta1_snap = np.zeros((ns, d+1))

    for t in range(ns):
        th0 = solve_theta(S0, b0)
        th1 = solve_theta(S1, b1)
        theta0_snap[t] = th0
        theta1_snap[t] = th1

        eps_t = eps_at(t)
        z_t = X_aug[t]
        q0 = z_t @ th0
        q1 = z_t @ th1
        if q1 > q0:
            pi1 = 1.0 - 0.5 * eps_t
        elif q1 < q0:
            pi1 = 0.5 * eps_t
        else:
            pi1 = 0.5

        a = 1 if rng.rand() < pi1 else 0
        T[t] = a
        w[t] = pi1

        y_t = Y1[t] if a == 1 else Y0[t]
        Y[t] = y_t

        if a == 0:
            S0 += np.outer(z_t, z_t); b0 += z_t * y_t
        else:
            S1 += np.outer(z_t, z_t); b1 += z_t * y_t

    # choose folds
    if split == "alternating":
        idx0 = np.arange(0, ns, 2)
        idx1 = np.arange(1, ns, 2)
    elif split == "chronological":
        cut = ns // 2
        idx0 = np.arange(0, cut)
        idx1 = np.arange(cut, ns)
    else:
        raise ValueError("split must be 'alternating' or 'chronological'.")

    # fold views (chronological inside each fold) — use augmented X for propensities
    Z0, Z1 = X_aug[idx0], X_aug[idx1]

    # build Π_{fold←fold}: rows = evaluation times t in that fold (chrono),
    # columns = contexts from the same fold (chrono)
    N0, N1 = idx0.size, idx1.size
    Pi_fold0_on_0 = np.empty((N0, N0))
    for r, t in enumerate(idx0):
        th0 = theta0_snap[t]; th1 = theta1_snap[t]; eps_t = eps_at(t)
        q0 = Z0 @ th0; q1 = Z0 @ th1
        Pi_fold0_on_0[r] = np.where(q1 > q0, 1.0 - 0.5 * eps_t,
                                    np.where(q1 < q0, 0.5 * eps_t, 0.5))

    Pi_fold1_on_1 = np.empty((N1, N1))
    for r, t in enumerate(idx1):
        th0 = theta0_snap[t]; th1 = theta1_snap[t]; eps_t = eps_at(t)
        q0 = Z1 @ th0; q1 = Z1 @ th1
        Pi_fold1_on_1[r] = np.where(q1 > q0, 1.0 - 0.5 * eps_t,
                                    np.where(q1 < q0, 0.5 * eps_t, 0.5))

    return X, T, Y[:, None], w, Pi_fold0_on_0, Pi_fold1_on_1, idx0, idx1

# ------------------------ Experiment Runner ------------------------ #
def run_comparison(
    R=200,
    n=700,
    d=5,
    seed=123,
    scenario='I',
    alpha=0.05
):
    rng_master = np.random.RandomState(seed)
    
    reject_dr = 0
    reject_vs = 0
    
    noise_var = 0.5
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    print(f"Running comparison with scenario={scenario} (H{'0' if scenario=='I' else '1'})...")
    
    for r in tqdm(range(R)):
        rng = np.random.RandomState(rng_master.randint(1, 10**9))
        
        X, T, Y, w, Pi_0_on_0, Pi_1_on_1, idx0, idx1 = collect_epsilon_greedy(
            ns=n, d=d, beta_vec=beta_vec, noise_var=noise_var, scenario=scenario,
            eps0=0.5, eps_min=0.2, power=0.5, lam=1e-4, rng=rng,
            split="alternating"
        )

        # Ensure shapes (N,1)
        if Y.ndim == 1:
            Y = Y[:, None]

        # Gaussian RBF kernel bandwidth on Y blocks
        YY0 = Y[T == 0]
        YY1 = Y[T == 1]
        sigma2 = np.median(pairwise_distances(YY0, YY1, metric='euclidean'))**2 / 4
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.var(Y)) + 1e-6

        # 2. DR-xKTE (Non-stabilized)
        stat_dr = xMMD2dr(Y, w, X, T, kernel_function="rbf", gamma=1.0/sigma2)
        pval_dr = 0.5 * (1.0 - erf(stat_dr / np.sqrt(2.0)))
        if pval_dr < alpha:
            reject_dr += 1

        # 3. VS-DR-KTE (Stabilized)
        stat_vs = xMMD2_vsdr_fold_generic(
            Y=Y, w=w, X=X, A=T,
            kernel_function='rbf',
            Pi_0_on_0=Pi_0_on_0,
            Pi_1_on_1=Pi_1_on_1,
            idx0=idx0,
            idx1=idx1,
            gamma=1.0/sigma2,
            lam=1e-2,
        )
        
        pval_vs = 0.5 * (1.0 - erf(stat_vs / np.sqrt(2.0)))
        if pval_vs < alpha:
            reject_vs += 1

    print(f"Results (R={R}, n={n}, scenario={scenario}):")
    print(f"DR-xKTE Rejection Rate: {reject_dr/R:.3f}")
    print(f"VS-DR-KTE Rejection Rate: {reject_vs/R:.3f}")
    return reject_dr/R, reject_vs/R

if __name__ == "__main__":
    # H0: Both should be ~0.05 (well-calibrated)
    print("="*60)
    run_comparison(R=200, n=700, scenario='I')
    
    # H1 - Mean shift: Both should have perfect power
    print("="*60)
    run_comparison(R=200, n=700, scenario='II')
    
    # H1 - Distributional shift (no mean change): VS-DR-KTE should excel
    print("="*60)
    run_comparison(R=200, n=700, scenario='III')
    
    print("="*60)
    run_comparison(R=200, n=700, scenario='IV')
