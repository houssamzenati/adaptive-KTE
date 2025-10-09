# %%
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
T = 2000              # horizon
t0 = 200              # explore length (fixed)
epsilon = 0.1         # ε-greedy in commit phase
true_mean = 0.5
num_experiments = 8000
rng = np.random.RandomState(12345)

# -------------------------
# Single run
# -------------------------
def run_once(parameters):
    T, t0, epsilon, true_mean, rng = parameters
    # Pre-draw rewards
    r1 = rng.normal(true_mean, 1.0, T)
    r2 = rng.normal(true_mean, 1.0, T)

    A = np.zeros(T, dtype=int)     # actions in {1,2}
    Y = np.zeros(T)                # observed rewards
    p1 = np.zeros(T)               # P(A_t=1 | history)

    # --- Explore: uniform 1/2
    for t in range(t0):
        A[t] = rng.choice([1, 2])
        Y[t] = r1[t] if A[t] == 1 else r2[t]
        p1[t] = 0.5

    # --- Decide "best" from explore-only empirical means
    best = 1 if np.mean(r1[:t0]) >= np.mean(r2[:t0]) else 2

    # --- Commit: ε-greedy around best
    for t in range(t0, T):
        a = best if rng.rand() > epsilon else 3 - best
        A[t] = a
        Y[t] = r1[t] if a == 1 else r2[t]
        p1[t] = (1 - epsilon) if best == 1 else epsilon  # prob of choosing arm 1

    # HT (unbiased; mixture-normal CLT with √T scaling)
    mu_ht = np.sum((A == 1) * Y / p1) / T

    # CM/EM (empirical mean on played arm; N(0,1) CLT with √N1 scaling)
    N1 = np.sum(A == 1)
    mu_em = np.sum((A == 1) * Y) / N1 if N1 > 0 else np.nan

    return mu_ht, mu_em, N1

# -------------------------
# Run many experiments
# -------------------------
params = (T, t0, epsilon, true_mean, rng)
results = [run_once(params) for _ in range(num_experiments)]
mu_ht = np.array([r[0] for r in results])
mu_em = np.array([r[1] for r in results])
N1     = np.array([r[2] for r in results])

# Remove rare NaNs if N1==0
mask = np.isfinite(mu_em) & (N1 > 0)
mu_em, N1 = mu_em[mask], N1[mask]

# -------------------------
# Scale
# -------------------------
z_ht = np.sqrt(T) * (mu_ht - true_mean)
z_em = np.sqrt(N1) * (mu_em - true_mean)

# -------------------------
# Theoretical overlays
# -------------------------
def std_normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def mixture_pdf(x, eps):
    v_large = 2 + ((1 - eps)**2) / 2.0   # branch where arm1 loses (p1≈ε)
    v_small = 2 + (eps**2) / 2.0         # branch where arm1 wins  (p1≈1-ε)
    comp1 = np.exp(-x**2 / (2 * v_large)) / np.sqrt(2 * np.pi * v_large)
    comp2 = np.exp(-x**2 / (2 * v_small)) / np.sqrt(2 * np.pi * v_small)
    return 0.5 * comp1 + 0.5 * comp2, v_large, v_small

# x-axes
x_ht = np.linspace(np.percentile(z_ht, 0.1), np.percentile(z_ht, 99.9), 1200)
x_em = np.linspace(np.percentile(z_em, 0.1), np.percentile(z_em, 99.9), 1200)
mix_ht, vL, vS = mixture_pdf(x_ht, epsilon)
phi_ht = std_normal_pdf(x_ht)
phi_em = std_normal_pdf(x_em)

print(f"HT scaled: mean={z_ht.mean():.3f}, var={z_ht.var():.3f}, "
      f"theory variances: large={vL:.3f}, small={vS:.3f}")
print(f"EM scaled: mean={z_em.mean():.3f}, var={z_em.var():.3f} (target N(0,1))")

# -------------------------
# Plots
# -------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# (A) HT: histogram + mixture overlay + standard normal overlay
ax[0].hist(z_ht, bins=60, density=True, alpha=0.6, label="HT simulated")
ax[0].plot(x_ht, mix_ht, lw=2, label="Mixture (theory)")
ax[0].plot(x_ht, phi_ht, lw=2, ls="--", label="Standard normal N(0,1)")
ax[0].set_title("HT: √T-scaled")
ax[0].legend()

# (B) EM: histogram + standard normal overlay
ax[1].hist(z_em, bins=60, density=True, alpha=0.6, label="EM simulated")
ax[1].plot(x_em, phi_em, lw=2, label="Standard normal N(0,1)")
ax[1].set_title("CM/EM: √N₁-scaled")
ax[1].legend()

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
parameters = {
    "T": 3000,            # horizon
    "t0": 300,            # explore length (fixed)
    "epsilon": 0.1,       # ε-greedy in commit
    "true_mean": 0.5,     # μ
    "theta": 0.8,         # strength of X effect (larger => stronger context dependence)
    "num_experiments": 6000,
    "seed": 1234,
}

# -------------------------
# Utilities
# -------------------------
def std_normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def ht_theory_mixture_pdf(x, epsilon):
    # Same unconditional mixture (explore contributes +2; commit branch adds ((1-eps)^2)/2 or (eps^2)/2)
    v_large = 2 + ((1 - epsilon)**2) / 2.0   # branch where p1≈ε
    v_small = 2 + (epsilon**2) / 2.0         # branch where p1≈1-ε
    comp1 = np.exp(-x**2 / (2 * v_large)) / np.sqrt(2 * np.pi * v_large)
    comp2 = np.exp(-x**2 / (2 * v_small)) / np.sqrt(2 * np.pi * v_small)
    return 0.5 * comp1 + 0.5 * comp2, v_large, v_small

def fit_lin(Y, X):
    # OLS on [1, X] -> returns (alpha_hat, beta_hat); handles empty set by fallback to zeros
    if len(Y) == 0:
        return 0.0, 0.0
    Xmat = np.column_stack([np.ones(len(X)), X])
    theta_hat, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    return float(theta_hat[0]), float(theta_hat[1])

# -------------------------
# One experiment with covariate X
# -------------------------
def run_once(parameters):
    T = parameters["T"]
    t0 = parameters["t0"]
    epsilon = parameters["epsilon"]
    true_mean = parameters["true_mean"]
    theta = parameters["theta"]
    rng = parameters["rng"]

    # Generate covariates and noise
    X = rng.normal(0.0, 1.0, T)
    eps1 = rng.normal(0.0, 1.0, T)
    eps2 = rng.normal(0.0, 1.0, T)

    # Potential outcomes
    r1 = true_mean + theta * X + eps1
    r2 = true_mean - theta * X + eps2

    A = np.zeros(T, dtype=int)   # action in {1,2}
    Y = np.zeros(T)              # observed reward
    p1 = np.zeros(T)             # P(A_t=1 | history, X_t)

    # --- EXPLORE: uniform 1/2, collect data to fit contextual models
    X1_obs, Y1_obs, X2_obs, Y2_obs = [], [], [], []
    for t in range(t0):
        a = rng.choice([1, 2])
        A[t] = a
        Y[t] = r1[t] if a == 1 else r2[t]
        p1[t] = 0.5
        if a == 1:
            X1_obs.append(X[t]); Y1_obs.append(Y[t])
        else:
            X2_obs.append(X[t]); Y2_obs.append(Y[t])

    # Fit contextual linear models on explore data
    a1_hat, b1_hat = fit_lin(np.array(Y1_obs), np.array(X1_obs))  # E[Y|A=1,X] ≈ a1_hat + b1_hat X
    a2_hat, b2_hat = fit_lin(np.array(Y2_obs), np.array(X2_obs))  # E[Y|A=2,X] ≈ a2_hat + b2_hat X

    # --- COMMIT: contextual ε-greedy using those models
    for t in range(t0, T):
        # Predict arm means given X_t
        m1 = a1_hat + b1_hat * X[t]
        m2 = a2_hat + b2_hat * X[t]
        best = 1 if m1 >= m2 else 2

        # ε-greedy around best
        a = best if rng.rand() > epsilon else 3 - best
        A[t] = a
        Y[t] = r1[t] if a == 1 else r2[t]

        # Propensity for arm 1 given X_t
        p1[t] = (1 - epsilon) if best == 1 else epsilon

    # --- Estimators
    # Horvitz–Thompson (targets μ = E[Y(1)] under the *marginal* X)
    mu_ht = np.sum((A == 1) * Y / p1) / T

    # Empirical mean on played arm 1 (will generally be biased because selection depends on X)
    N1 = np.sum(A == 1)
    mu_em = np.sum((A == 1) * Y) / N1 if N1 > 0 else np.nan

    return mu_ht, mu_em, N1

# -------------------------
# Run many experiments
# -------------------------
rng = np.random.RandomState(parameters["seed"])
parameters["rng"] = rng
out = [run_once(parameters) for _ in range(parameters["num_experiments"])]

mu_ht = np.array([o[0] for o in out])
mu_em = np.array([o[1] for o in out])
N1    = np.array([o[2] for o in out])

mask = np.isfinite(mu_em) & (N1 > 0)
mu_em, N1 = mu_em[mask], N1[mask]

T = parameters["T"]
epsilon = parameters["epsilon"]
true_mean = parameters["true_mean"]

# Scaled statistics
z_ht = np.sqrt(T) * (mu_ht - true_mean)          # HT: √T scaling
z_em = np.sqrt(N1) * (mu_em - true_mean)         # EM: √N1 scaling

print(f"HT  scaled -> mean={z_ht.mean():.3f}, var={z_ht.var():.3f}")
print(f"EM  scaled -> mean={z_em.mean():.3f}, var={z_em.var():.3f}")
print(f"Avg N1 over runs: {N1.mean():.1f} of T={T}")

# -------------------------
# Plots
# -------------------------
# HT panel: histogram + mixture and standard normal overlays
x_ht = np.linspace(np.percentile(z_ht, 0.1), np.percentile(z_ht, 99.9), 1200)
mix_ht, vL, vS = ht_theory_mixture_pdf(x_ht, epsilon)
phi_ht = std_normal_pdf(x_ht)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(z_ht, bins=60, density=True, alpha=0.6, label="HT simulated")
ax[0].plot(x_ht, mix_ht, lw=2, label="Mixture (theory)")
ax[0].plot(x_ht, phi_ht, lw=2, ls="--", label="Standard normal N(0,1)")
ax[0].set_title("HT (contextual): √T-scaled")
ax[0].legend()

# EM panel: histogram + standard normal overlay
x_em = np.linspace(np.percentile(z_em, 0.1), np.percentile(z_em, 99.9), 1200)
phi_em = std_normal_pdf(x_em)
ax[1].hist(z_em, bins=60, density=True, alpha=0.6, label="EM simulated")
ax[1].plot(x_em, phi_em, lw=2, label="Standard normal N(0,1)")
ax[1].set_title("CM/EM (contextual): √N₁-scaled")
ax[1].legend()

plt.tight_layout()
plt.show()

# %%
