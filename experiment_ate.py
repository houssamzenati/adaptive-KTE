# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- same data + DR helpers as before ----
def make_data(n, d, rng, sigma_eps=0.4):
    X = rng.randn(n, d)
    u = rng.randn(d); u /= np.linalg.norm(u) + 1e-12
    v = rng.randn(d); v /= np.linalg.norm(v) + 1e-12
    lin = X @ v
    phi = np.cos(X @ u + 0.8)
    f = 0.6*lin + 0.8*phi + 0.3*(lin*phi)
    Delta = 1.0 + 0.7*np.tanh(lin)
    m0 = f
    m1 = f + Delta
    eps = sigma_eps * rng.randn(n)
    return X, m0, m1, Delta, eps

def policy_explore_commit(X, t, t0, eps, u, winner):
    if t < t0:
        return 0.5
    inS = (X @ u) > 0
    if inS:
        return 1 - eps if winner == 1 else eps
    else:
        return eps if winner == 1 else 1 - eps

def dr_estimate(Y, A, X, p_realized, m0_hat, m1_hat, p_eval=None):
    p = p_realized if p_eval is None else p_eval
    return np.mean((m1_hat - m0_hat)
                   + (A/p)*(Y - m1_hat)
                   - ((1-A)/(1-p))*(Y - m0_hat))

# %%
def simulate_EC_DR_with_Z(n=4000, d=5, t0=40, eps=0.005, sigma_eps=0.4, rng=None):
    rng = rng or np.random.RandomState(42)
    X, m0, m1, Delta, eps_noise = make_data(n, d, rng, sigma_eps=sigma_eps)
    u = rng.randn(d); u /= np.linalg.norm(u) + 1e-12
    winner = 1 if rng.rand() < 0.5 else 0

    p = np.zeros(n); A = np.zeros(n, int); Y = np.zeros(n)
    for t in range(n):
        p[t] = policy_explore_commit(X[t], t, t0, eps, u, winner)
        A[t] = 1 if rng.rand() < p[t] else 0
        Y[t] = (m1[t] if A[t] == 1 else m0[t]) + eps_noise[t]

    tau_true = np.mean(m1 - m0)

    # --- DR estimates (unchanged) ---
    tau_bad  = dr_estimate(Y, A, X, p,   m0, m1, p_eval=None)  # realized p_t
    tau_eval = dr_estimate(Y, A, X, 0.5, m0, m1, p_eval=0.5)   # predictable p = 0.5

    psi_bad  = (m1 - m0) + (A/p)*(Y - m1) - ((1-A)/(1-p))*(Y - m0) - tau_true
    psi_eval = (m1 - m0) + (A/0.5)*(Y - m1) - ((1-A)/0.5)*(Y - m0) - tau_true

    se_bad  = np.sqrt(np.var(psi_bad,  ddof=1) / n)
    se_eval = np.sqrt(np.var(psi_eval, ddof=1) / n)
    Z_bad   = (tau_bad  - tau_true) / (se_bad  + 1e-12)
    Z_eval  = (tau_eval - tau_true) / (se_eval + 1e-12)

    # --- CADR for ATE (new) ---
    # D'_t,t with outcome plug-in (use m0,m1 here to stay comparable to DR setup)
    def Dprime_ATE(pt, a, y, q0, q1):
        return (q1 - q0) + (a / (pt + 1e-12)) * (y - q1) - ((1 - a) / (1 - pt + 1e-12)) * (y - q0)

    # Per-time stabilized weights σ̂_t^{-1} from past via importance ratios g_t/g_s
    w = np.zeros(n)        # σ̂_t^{-1}
    D = np.zeros(n)        # D′(g_t, Q̂_{t-1})(O_t)  -- here we use (m0,m1)
    for t in range(n):
        D[t] = Dprime_ATE(p[t], A[t], Y[t], m0[t], m1[t])

        if t == 0:
            w[t] = 0.0
            continue

        ratios = np.empty(t)
        Ds_t   = np.empty(t)
        for s in range(t):
            pt_on_xs = policy_explore_commit(X[s], t, t0, eps, u, winner)  # g_t(1|X_s)
            gs_on_xs = p[s]                                              # g_s(1|X_s)

            # importance ratio g_t/g_s evaluated at (A_s, X_s)
            num = pt_on_xs if A[s] == 1 else (1.0 - pt_on_xs)
            den = gs_on_xs if A[s] == 1 else (1.0 - gs_on_xs)
            ratios[s] = num / (den + 1e-12)

            # D′(g_t, Q̂_{t-1}) evaluated at O_s (use the same (m0,m1) plug-in)
            Ds_t[s] = Dprime_ATE(pt_on_xs, A[s], Y[s], m0[s], m1[s])

        m1_hat = np.mean(ratios * (Ds_t**2))
        m2_hat = (np.mean(ratios * Ds_t))**2
        var_t  = m1_hat - m2_hat
        w[t]   = 0.0 if var_t <= 0 else 1.0 / np.sqrt(var_t + 1e-12)

    # CADR estimator 
    W = w.sum()
    if W <= 1e-12:
        cadr_mean = D.mean()
        cadr_var  = np.var(D, ddof=1) / n      # conservative fallback
    else:
        cadr_mean = (w @ D) / W
        cadr_var  = ((w**2) @ (D - cadr_mean)**2) / (W**2)


    # inputs: arrays D (length T), w (length T), and tau_true
    tau0 = float(tau_true)
    U = w * (D - cadr_mean)                                     # centered stabilized summands
    num = (w * (D - tau0)).sum()
    den = np.sqrt((U**2).sum() + 1e-12)
    Z_cadr_sn = num / den


    return Z_bad, Z_eval, Z_cadr_sn

# %%

# from sklearn.linear_model import Ridge

# def _fit_q_past(X, Y, A, alpha=1e-1):
#     # simple ridge per arm on the provided past sample
#     if len(X) == 0:
#         return None, None
#     mask0 = (A == 0); mask1 = ~mask0
#     q0 = Ridge(alpha=alpha).fit(X[mask0], Y[mask0]) if mask0.any() else None
#     q1 = Ridge(alpha=alpha).fit(X[mask1], Y[mask1]) if mask1.any() else None
#     return q0, q1

# def simulate_EC_DR_with_Z(n=4000, d=5, t0=40, eps=0.005, sigma_eps=0.4, rng=None):
#     rng = rng or np.random.RandomState(42)
#     X, m0, m1, Delta, eps_noise = make_data(n, d, rng, sigma_eps=sigma_eps)
#     u = rng.randn(d); u /= np.linalg.norm(u) + 1e-12
#     winner = 1 if rng.rand() < 0.5 else 0

#     p = np.zeros(n); A = np.zeros(n, int); Y = np.zeros(n)
#     for t in range(n):
#         p[t] = policy_explore_commit(X[t], t, t0, eps, u, winner)
#         A[t] = 1 if rng.rand() < p[t] else 0
#         Y[t] = (m1[t] if A[t] == 1 else m0[t]) + eps_noise[t]

#     tau_true = np.mean(m1 - m0)

#     # --- DR (unchanged) ---
#     tau_bad  = dr_estimate(Y, A, X, p,   m0, m1, p_eval=None)
#     tau_eval = dr_estimate(Y, A, X, 0.5, m0, m1, p_eval=0.5)
#     psi_bad  = (m1 - m0) + (A/p)*(Y - m1) - ((1-A)/(1-p))*(Y - m0) - tau_true
#     psi_eval = (m1 - m0) + (A/0.5)*(Y - m1) - ((1-A)/0.5)*(Y - m0) - tau_true
#     se_bad  = np.sqrt(np.var(psi_bad,  ddof=1) / n)
#     se_eval = np.sqrt(np.var(psi_eval, ddof=1) / n)
#     Z_bad   = (tau_bad  - tau_true) / (se_bad  + 1e-12)
#     Z_eval  = (tau_eval - tau_true) / (se_eval + 1e-12)

#     # --- CADR for ATE using Q_hat_{t-1} learned on past data ---
#     def Dprime_ATE(pt, a, y, q0x, q1x):
#         return (q1x - q0x) + (a/(pt+1e-12))*(y - q1x) - ((1-a)/(1-pt+1e-12))*(y - q0x)

#     w = np.zeros(n)     # 1/sqrt(sigma_t^2)
#     D = np.zeros(n)     # D'(g_t, Q_hat_{t-1})(O_t)

#     for t in range(n):
#         # fit Q_hat_{t-1} on {0,...,t-1}
#         if t == 0:
#             # no past; neutral weights
#             q0_hat = q1_hat = None
#             q0x_t, q1x_t = m0[t], m1[t]   # fallback (won't affect much, w[0]=0)
#         else:
#             q0_hat, q1_hat = _fit_q_past(X[:t], Y[:t], A[:t], alpha=1e-1)
#             q0x_t = q0_hat.predict(X[t:t+1])[0] if q0_hat else m0[t]
#             q1x_t = q1_hat.predict(X[t:t+1])[0] if q1_hat else m1[t]

#         D[t] = Dprime_ATE(p[t], A[t], Y[t], q0x_t, q1x_t)

#         if t == 0:
#             w[t] = 0.0
#             continue

#         # variance at time t via past reweighting with Q_hat_{t-1} (same fit)
#         ratios = np.empty(t); Ds_t = np.empty(t)
#         for s in range(t):
#             pt_on_xs = policy_explore_commit(X[s], t, t0, eps, u, winner)
#             gs_on_xs = p[s]
#             num = pt_on_xs if A[s] == 1 else (1.0 - pt_on_xs)
#             den = gs_on_xs if A[s] == 1 else (1.0 - gs_on_xs)
#             ratios[s] = num / (den + 1e-12)

#             q0x_s = q0_hat.predict(X[s:s+1])[0] if q0_hat else m0[s]
#             q1x_s = q1_hat.predict(X[s:s+1])[0] if q1_hat else m1[s]
#             Ds_t[s] = Dprime_ATE(pt_on_xs, A[s], Y[s], q0x_s, q1x_s)

#         m1_hat = np.mean(ratios * (Ds_t**2))
#         m2_hat = (np.mean(ratios * Ds_t))**2
#         var_t  = max(m1_hat - m2_hat, 0.0)
#         w[t]   = 0.0 if var_t <= 0 else 1.0 / np.sqrt(var_t + 1e-12)

#     # CADR mean & variance (your notebook’s formula)
#     W = w.sum()
#     if W <= 1e-12:
#         cadr_mean = D.mean()
#         cadr_var  = np.var(D, ddof=1)/n
#     else:
#         cadr_mean = (w @ D) / W
#         cadr_var  = ((w**2) @ (D - cadr_mean)**2) / (W**2)

#     Z_cadr = (cadr_mean - tau_true) / np.sqrt(cadr_var + 1e-12)
#     return Z_bad, Z_eval, Z_cadr


def compare_to_standard_normal(R=1000, n=4000, d=5, t0=40, eps=0.005, sigma_eps=0.35, seed=123):
    rng = np.random.RandomState(seed)
    Zb = np.zeros(R); Ze = np.zeros(R); Zc = np.zeros(R)
    for r in tqdm(range(R)):
        Z_bad, Z_eval, Z_cadr = simulate_EC_DR_with_Z(
            n=n, d=d, t0=t0, eps=eps, sigma_eps=sigma_eps,
            rng=np.random.RandomState(rng.randint(1, 10**9))
        )
        Zb[r] = Z_bad; Ze[r] = Z_eval; Zc[r] = Z_cadr

    # each subplot keeps its own x axis
    fig, axs = plt.subplots(1, 3, figsize=(14,4), sharex=False)

    def overlay_standard_normal(ax, lo, hi):
        xs = np.linspace(lo, hi, 500)
        ax.plot(xs, np.exp(-0.5*xs**2)/np.sqrt(2*np.pi), lw=1.8, label="standard Normal N(0,1)")

    # panel 1: DR realized p_t
    lo1, hi1 = Zb.min(), Zb.max()
    bins1 = np.linspace(np.floor(lo1*2)/2, np.ceil(hi1*2)/2, 35)
    axs[0].hist(Zb, bins=bins1, density=True, alpha=0.7, label="DR (realized p_t)")
    overlay_standard_normal(axs[0], bins1[0], bins1[-1])
    axs[0].set_title("Z = studentized DR with realized propensities")
    axs[0].set_xlabel("Z"); axs[0].set_ylabel("Density"); axs[0].grid(alpha=0.3); axs[0].legend()

    # panel 2: DR predictable p
    lo2, hi2 = Ze.min(), Ze.max()
    bins2 = np.linspace(np.floor(lo2*2)/2, np.ceil(hi2*2)/2, 35)
    axs[1].hist(Ze, bins=bins2, density=True, alpha=0.7, color='tab:orange', label="DR (predictable p)")
    overlay_standard_normal(axs[1], bins2[0], bins2[-1])
    axs[1].set_title("Z = studentized DR with predictable propensities")
    axs[1].set_xlabel("Z"); axs[1].grid(alpha=0.3); axs[1].legend()

    # panel 3: CADR
    lo3, hi3 = Zc.min(), Zc.max()
    bins3 = np.linspace(np.floor(lo3*2)/2, np.ceil(hi3*2)/2, 35)
    axs[2].hist(Zc, bins=bins3, density=True, alpha=0.7, color='tab:green', label="CADR")
    overlay_standard_normal(axs[2], bins3[0], bins3[-1])
    axs[2].set_title("Z = studentized CADR")
    axs[2].set_xlabel("Z"); axs[2].grid(alpha=0.3); axs[2].legend()

    plt.tight_layout(); plt.show()

compare_to_standard_normal(R=100, n=1000, d=5, t0=5, eps=0.001, sigma_eps=0.35)
# %%
