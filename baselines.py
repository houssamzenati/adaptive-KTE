import numpy as np
import numpy as np
from sklearn.metrics import pairwise_kernels


def _kernel_matrix(XA, XB, kernel_function="rbf", gamma=None):
    if kernel_function == "rbf":
        return pairwise_kernels(XA, XB, metric="rbf", gamma=gamma)
    elif kernel_function == "linear":
        return pairwise_kernels(XA, XB, metric="linear")
    else:
        raise ValueError(f"Unsupported kernel_function: {kernel_function}")


def fit_krr(X_train, y_train, kernel_function="rbf", gamma=None, lam=1e-4):
    """
    Kernel Ridge Regression (KRR).
    Returns a predictor: predict(X_eval) -> y_hat.
    """
    y_train = np.asarray(y_train).reshape(-1)
    if X_train.shape[0] == 0:
        return lambda X_eval: np.full(X_eval.shape[0], np.nan)
    K_tr = _kernel_matrix(X_train, X_train, kernel_function, gamma)
    n_tr = K_tr.shape[0]
    alpha = np.linalg.solve(K_tr + lam * np.eye(n_tr), y_train)

    def predict(X_eval):
        K_ev = _kernel_matrix(X_eval, X_train, kernel_function, gamma)
        return K_ev @ alpha

    return predict


# convenience wrapper if you want the direct "predict now" version
def fit_krr_predict(
    X_train, y_train, X_eval, kernel_function="rbf", gamma=None, lam=1e-4
):
    return fit_krr(
        X_train, y_train, kernel_function=kernel_function, gamma=gamma, lam=lam
    )(X_eval)


def clip01(x, p_min):
    if p_min is None or p_min <= 0:
        return x
    return np.minimum(1.0 - p_min, np.maximum(p_min, x))


def _phi_cdf(z):
    # Φ(z) using erf (no external deps)
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def Dprime_ATE(p, A, Y, m0, m1, p_min=0.0):
    # Uncentered DR score with E[D'] = τ
    # (kept identical to your previous definition; p_min ignored in theory mode)
    term1 = m1 - m0
    term2 = (A / (p + 1e-12)) * (Y - m1)
    term3 = ((1 - A) / (1 - p + 1e-12)) * (Y - m0)
    return term1 + term2 - term3


def _cadr_theory_predictable_weights(X, A, Y, p_realized, m0, m1, policy_fn):
    """
    Build w_t = 1 / σ̂_t using only data up to t-1.
    At time t, estimate Var_t[D'] under g_t by importance-reweighting from realized g_s.
    No clipping/caps/trim/windows; uses all s < t.
    """
    n = len(A)
    D = Dprime_ATE(p_realized, A, Y, m0, m1, p_min=0.0)
    w = np.zeros(n, dtype=float)

    for t in range(1, n):
        idx = np.arange(0, t)
        if idx.size < 2:
            continue

        # g_t(1|X_s) and realized g_s(1|X_s)
        pt_on_xs = np.array([policy_fn(X[s], t) for s in idx])  # in (0,1)
        gs_on_xs = p_realized[idx]  # in (0,1)

        # importance ratios r = g_t/g_s evaluated at (A_s, X_s)
        num = np.where(A[idx] == 1, pt_on_xs, 1.0 - pt_on_xs)
        den = np.where(A[idx] == 1, gs_on_xs, 1.0 - gs_on_xs)
        r = num / (den + 1e-12)

        # D' evaluated at O_s but with g_t in the denominators
        Ds_t = Dprime_ATE(pt_on_xs, A[idx], Y[idx], m0[idx], m1[idx], p_min=0.0)

        # Predictable variance under g_t (self-normalized IS *not* used here)
        m1_hat = np.mean(r * (Ds_t**2))
        m2_hat = (np.mean(r * Ds_t)) ** 2
        var_t = m1_hat - m2_hat

        if var_t > 0.0 and np.isfinite(var_t):
            w[t] = 1.0 / np.sqrt(var_t + 1e-18)
        else:
            w[t] = 0.0

    return w, D


def _cadr_theory_estimate(X, A, Y, p_realized, m0, m1, policy_fn):
    """
    τ̂ = (∑ w_t D'_t) / (∑ w_t),    Var̂(τ̂) = (∑ w_t^2 (D'_t - τ̂)^2) / (∑ w_t)^2
    """
    w, D = _cadr_theory_predictable_weights(X, A, Y, p_realized, m0, m1, policy_fn)
    Wsum = float(np.sum(w))
    if Wsum <= 1e-12:
        # degenerate fallback (should not happen if assumptions hold)
        tau_hat = float(np.mean(D))
        var_hat = float(np.var(D, ddof=1) / max(len(D), 1))
        return tau_hat, var_hat, w, D

    tau_hat = float((w @ D) / Wsum)
    var_hat = float(((w**2) @ (D - tau_hat) ** 2) / (Wsum**2) + 1e-18)
    return tau_hat, var_hat, w, D


def cadr_test(X, A, Y, p_realized, m0, m1, P_all):
    """
    Theory-faithful CADR using ONLY propensities:
      - p_realized[s] = g_s(1|X_s)
      - P_all[t, s]   = g_t(1|X_s)
    No clipping/caps/trim/windows. Normal p-values.

    returns
    -------
    dict: {'stat', 'p_value', 'tau_hat', 'se', 'method'}
    """
    n = len(A)
    Y = np.asarray(Y).reshape(-1)
    m0 = np.asarray(m0)
    m1 = np.asarray(m1)
    A = np.asarray(A)
    p_realized = np.asarray(p_realized)

    # Uncentered DR scores under realized logging g_s
    D = Dprime_ATE(p_realized, A, Y, m0, m1, p_min=0.0)

    # Predictable inverse-std weights w_t = 1 / sqrt(Var_t[D'])
    w_pred = np.zeros(n, dtype=float)
    for t in range(1, n):
        idx = np.arange(0, t)
        pt_on_xs = P_all[t, idx]  # g_t(1 | X_s) for s < t
        gs_on_xs = p_realized[idx]  # g_s(1 | X_s)
        num = np.where(A[idx] == 1, pt_on_xs, 1.0 - pt_on_xs)
        den = np.where(A[idx] == 1, gs_on_xs, 1.0 - gs_on_xs)
        r = num / (den + 1e-12)
        # D' evaluated with g_t inside the denominators
        Ds_t = Dprime_ATE(pt_on_xs, A[idx], Y[idx], m0[idx], m1[idx], p_min=0.0)
        m1_hat = np.mean(r * (Ds_t**2))
        m2_hat = (np.mean(r * Ds_t)) ** 2
        var_t = m1_hat - m2_hat
        w_pred[t] = (
            1.0 / np.sqrt(var_t + 1e-18) if (var_t > 0 and np.isfinite(var_t)) else 0.0
        )

    Wsum = float(np.sum(w_pred))
    if Wsum <= 1e-12:
        tau_hat = float(np.mean(D))
        var_hat = float(np.var(D, ddof=1) / max(len(D), 1))
    else:
        tau_hat = float((w_pred @ D) / Wsum)
        var_hat = float(((w_pred**2) @ (D - tau_hat) ** 2) / (Wsum**2) + 1e-18)

    se = float(np.sqrt(max(var_hat, 0.0)))
    stat = tau_hat / (se + 1e-12)

    from math import erf, sqrt

    Phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    p_value = 2.0 * (1.0 - Phi(abs(stat)))

    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "tau_hat": float(tau_hat),
        "se": float(se),
        "method": "CADR (theory, P-matrix) normal",
    }


# Hadad et al.


def _aipw_single_arm(w, A, Y, e, m0, m1, p_min=0.0):
    e1 = clip01(e, p_min)
    e0 = 1.0 - e1
    if w == 1:
        inv = A / (e1 + 1e-12)
        mhat = m1
    else:
        inv = (1 - A) / (e0 + 1e-12)
        mhat = m0
    return inv * Y + (1.0 - inv) * mhat


def _lambda_constant(t, T, e_t, alpha):
    return 1.0 / (T - t + 1.0)


def _lambda_two_point(t, T, e_t, alpha):
    if alpha >= 1.0:
        alpha = 0.999999
    term_hi = 1.0 / (T - t + 1.0)
    t_pow = t ** (-alpha)
    tail = (T ** (1.0 - alpha) - t ** (1.0 - alpha)) / (1.0 - alpha)
    term_lo = t_pow / (t_pow + tail + 1e-18)
    return e_t * term_hi + (1.0 - e_t) * term_lo


def _hadad_weights(e, scheme="two_point", alpha=0.7, p_min=0.0):
    T = len(e)
    e = clip01(np.asarray(e, float), p_min)
    h2_over_e = np.zeros(T, float)
    rem = 1.0
    for t in range(T):
        et = float(e[t])
        tt = t + 1
        if scheme == "constant":
            lam = _lambda_constant(tt, T, et, alpha)
        elif scheme == "two_point":
            lam = _lambda_two_point(tt, T, et, alpha)
        else:
            raise ValueError("scheme must be 'constant' or 'two_point'")
        if t == T - 1:
            lam = 1.0
        lam = max(0.0, min(float(lam), 1.0 - 1e-12)) if t < T - 1 else 1.0
        h2_over_e[t] = rem * lam
        rem -= h2_over_e[t]
        rem = max(rem, 0.0)
    return np.sqrt(h2_over_e * e)  # h_t


def _hadad_value(
    w, A, Y, p, m0, m1, scheme="two_point", alpha=0.7, p_min=0.0, estimator="aipw"
):
    A = np.asarray(A)
    Y = np.asarray(Y)
    p = clip01(np.asarray(p, float), p_min)
    m0 = np.asarray(m0)
    m1 = np.asarray(m1)
    e_w = p if w == 1 else 1.0 - p
    h = _hadad_weights(e_w, scheme=scheme, alpha=alpha, p_min=p_min)

    if estimator == "aipw":
        Gamma = _aipw_single_arm(w, A, Y, e_w, m0, m1, p_min=p_min)
        Hsum = float(h.sum())
        Q_hat = float((h @ Gamma) / (Hsum + 1e-18))
        V_hat = float(((h**2) @ (Gamma - Q_hat) ** 2) / ((Hsum + 1e-18) ** 2))
    elif estimator == "h-avg":
        num = float((h * (((A if w == 1 else (1 - A)) / (e_w + 1e-12)) * Y)).sum())
        den = float((h * ((A if w == 1 else (1 - A)) / (e_w + 1e-12))).sum() + 1e-18)
        Q_hat = num / den
        V_hat = float(
            (
                (h**2)
                * ((((A if w == 1 else (1 - A)) / (e_w + 1e-12)) * (Y - Q_hat)) ** 2)
            ).sum()
            / (den**2 + 1e-18)
        )
    else:
        raise ValueError("estimator must be 'aipw' or 'h-avg'")

    return Q_hat, V_hat, h


def hadad_test(
    A, Y, p, m0, m1, scheme="two_point", alpha=0.7, p_min=0.0, estimator="aipw"
):
    """
    Returns {'stat', 'p_value', 'tau_hat', 'method'} using NORMAL reference:
      stat = (Q1 - Q0) / sqrt(V1 + V0), p_value = 2*(1-Φ(|stat|))
    """
    Q1, V1, _ = _hadad_value(
        1, A, Y, p, m0, m1, scheme=scheme, alpha=alpha, p_min=p_min, estimator=estimator
    )
    Q0, V0, _ = _hadad_value(
        0, A, Y, p, m0, m1, scheme=scheme, alpha=alpha, p_min=p_min, estimator=estimator
    )
    tau_hat = Q1 - Q0
    var_hat = V1 + V0
    se = float(np.sqrt(max(var_hat, 0.0)))
    stat = tau_hat / (se + 1e-12)
    p_value = 2.0 * (1.0 - _phi_cdf(abs(stat)))
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "tau_hat": float(tau_hat),
        "method": f"Hadad AIPW ({scheme}) normal",
    }
