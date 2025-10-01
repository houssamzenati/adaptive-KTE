import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances
import scipy.stats as st

def make_psd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure the matrix is positive semi-definite (PSD).

    Given a matrix A, this function ensures it is positive semi-definite by making it symmetric
    and adding a small positive value to its diagonal.

    Parameters:
    - A (np.ndarray): Input matrix.

    Returns:
    - np.ndarray: Positive semi-definite matrix.
    """
    n = A.shape[0]
    return (A + A.T) / 2 + eps * np.eye(n)

# ---------- splits
def chronological_folds(N):
    cut = N // 2
    return np.arange(cut), np.arange(cut, N)


def alternating_folds(N, one_indexed=True):
    all_idx = np.arange(N)
    if one_indexed:
        return all_idx[0::2], all_idx[1::2]
    return all_idx[::2], all_idx[1::2]


# ---------- DR blocks
def _build_mu_R_Delta(KXX_fold, A_fold, lam):

    n = A_fold.size
    idx_control = np.where(A_fold == 0)[0]
    idx_treated = np.where(A_fold == 1)[0]
    # if idx_control.size == 0 or idx_treated.size == 0:
    #     raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

    m_control, n_treated = idx_control.size, idx_treated.size
    cols_perm = np.r_[idx_control, idx_treated]

    K_cc = KXX_fold[np.ix_(idx_control, idx_control)]
    K_ct = KXX_fold[np.ix_(idx_control, idx_treated)]
    K_tc = KXX_fold[np.ix_(idx_treated, idx_control)]
    K_tt = KXX_fold[np.ix_(idx_treated, idx_treated)]

    mu0 = np.linalg.solve(K_cc + m_control * lam * np.eye(m_control), np.hstack([K_cc, K_ct]))
    mu1 = np.linalg.solve(K_tt + n_treated * lam * np.eye(n_treated), np.hstack([K_tc, K_tt]))

    mu_arm = np.zeros((n, n))
    mu_arm[idx_control, :][:, cols_perm] = mu0
    mu_arm[idx_treated, :][:, cols_perm] = mu1

    mu_chrono = np.zeros((n, n))
    mu_chrono[:, cols_perm] = mu_arm

    mu1_pad = np.zeros((n, n))
    mu1_pad[idx_treated, :] = mu_chrono[idx_treated, :]
    mu0_pad = np.zeros((n, n))
    mu0_pad[idx_control, :] = mu_chrono[idx_control, :]

    mu = mu0_pad + mu1_pad
    R = np.eye(n) - mu
    Delta = mu1_pad - mu0_pad
    return R, Delta


# ---------- fast variance precomputation
def _precompute_fold_quadratics(KFF, R, Delta):
    KDelta = KFF @ Delta
    KR = KFF @ R
    v_dd = np.sum(Delta * (KFF @ Delta), axis=0)  # diag(ΔᵀKΔ)
    v_dr = np.sum(Delta * KR, axis=0)  # diag(ΔᵀKR)
    v_rr = np.sum(R * KR, axis=0)  # diag(RᵀKR)
    return v_dd, v_dr, v_rr, KDelta, KR


def _fold_omegas(KFF, R, Delta, A_fold, w_fold, Pi_fold_on_fold, eps=1e-12):
    """
    Pi_fold_on_fold: shape (n, n). Row t = [π_t(A_s|X_s)]_{s=0..n-1} for contexts in this fold, in chronological order.
    """
    n = A_fold.size
    # logging propensities on the realized arm
    denom = np.where(A_fold == 0, 1.0 - w_fold, w_fold)  # (n,)

    v_dd, v_dr, v_rr, KDelta, KR = _precompute_fold_quadratics(KFF, R, Delta)

    omega = np.zeros(n, dtype=float)
    for t in range(n):
        # vector over s: π_t(A_s|X_s) for this fold
        p_t = Pi_fold_on_fold[t]  # shape (n,)
        p_t = np.clip(p_t, 1e-6, 1.0 - 1e-6)  # safety

        # DR multiplier pattern for arm observed at s
        s_t = np.where(A_fold == 0, -1.0 / (1.0 - p_t + eps), 1.0 / (p_t + eps))  # (n,)
        # importance ratio π_t/π_s on realized arm
        num = np.where(A_fold == 0, 1.0 - p_t, p_t)  # (n,)
        rho = (num + eps) / (denom + eps)  # (n,)
        rho = np.clip(rho, -1e2, 1e2)

        past = np.arange(n) < t
        S = int(past.sum())
        if S == 0:
            omega[t] = 1.0  # or set 0.0 and skip t=0 upstream—just be consistent
            continue

        u_t = (rho * past) / S
        v_t = s_t * u_t

        # predictable variance (second moment) and mean-squared term
        q_diag = v_dd + 2.0 * s_t * v_dr + (s_t**2) * v_rr
        M2 = np.sum((rho * past) * q_diag) / S

        z = Delta @ u_t
        q = R @ v_t
        Kz = KDelta @ u_t
        Kq = KR @ v_t
        M1sq = z @ Kz + 2.0 * (z @ Kq) + q @ Kq

        var_t = M2 - M1sq
        var_t = max(var_t, 1e-8)
        omega[t] = 1.0 / np.sqrt(var_t)
        # omega[t] = 0.0 if var_t <= 0.0 else 1.0 / np.sqrt(var_t + eps)

    return omega


# ---------- VS-DR xKTE (chronology preserved)
def xMMD2_vsdr_fold_generic(
    Y, w, X, A, kernel_function, Pi_0_on_0, Pi_1_on_1, idx0, idx1, lam=None, **kwargs
):
    eps = 1e-12

    Y = np.asarray(Y)
    Y = Y[:, None] if Y.ndim == 1 else Y
    X = np.asarray(X)
    X = X[:, None] if X.ndim == 1 else X
    A = np.asarray(A, int).ravel()
    w = np.asarray(w, dtype=float).ravel()

    Y0, Y1 = Y[idx0], Y[idx1]
    X0, X1 = X[idx0], X[idx1]
    A0, A1 = A[idx0], A[idx1]
    w0, w1 = w[idx0], w[idx1]

    gamma = np.median(pairwise_distances(X1, X1, metric="euclidean")) ** 2
    lam = gamma if lam is None else lam

    KX0 = pairwise_kernels(X0, X0, metric=kernel_function, gamma=1.0 / gamma)
    KX1 = pairwise_kernels(X1, X1, metric=kernel_function, gamma=1.0 / gamma)

    R0, Delta0 = _build_mu_R_Delta(make_psd(KX0), A0, lam)
    R1, Delta1 = _build_mu_R_Delta(make_psd(KX1), A1, lam)

    W0 = np.where(A0 == 0, -1.0 / (1.0 - w0 + eps), 1.0 / (w0 + eps))
    W1 = np.where(A1 == 0, -1.0 / (1.0 - w1 + eps), 1.0 / (w1 + eps))

    M0 = Delta0 + np.diag(W0) @ R0
    M1 = Delta1 + np.diag(W1) @ R1

    K_YY = make_psd(pairwise_kernels(Y, Y, metric=kernel_function, 
                            **kwargs
                            ))
    # K01 = pairwise_kernels(Y0, Y1, metric=kernel_function, 
    #                         **kwargs
    #                         )
    K01 = K_YY[np.ix_(idx0, idx1)]
    G0 = M0.T @ K01 @ M1

    omega0 = _fold_omegas(
        # KFF=make_psd(pairwise_kernels(Y0, Y0, metric=kernel_function, 
        #                                 **kwargs
        #                                 )),
        KFF = K_YY[np.ix_(idx0, idx0)],
        R=R0,
        Delta=Delta0,
        A_fold=A0,
        w_fold=w0,
        Pi_fold_on_fold=Pi_0_on_0,
    )
    omega1 = _fold_omegas(
        # KFF=make_psd(pairwise_kernels(Y1, Y1, metric=kernel_function, 
        #                                 **kwargs
        #                                 )),
        KFF = K_YY[np.ix_(idx1, idx1)],
        R=R1,
        Delta=Delta1,
        A_fold=A1,
        w_fold=w1,
        Pi_fold_on_fold=Pi_1_on_1,
    )

    G = G0 / (omega0[:, None] * omega1[None, :] + eps)
    # G = G0

    S = np.mean(G)
    psi_hat = np.mean(G**2)

    return float(np.sqrt(G.size) * S / np.sqrt(psi_hat + eps))
