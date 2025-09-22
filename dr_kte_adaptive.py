import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances
import scipy.stats as st

# ---------- helpers


def chronological_folds(N):
    """I_0 = {0,...,N//2-1}, I_1 = {N//2,...,N-1} (0-based)."""
    cut = N // 2
    return np.arange(cut), np.arange(cut, N)


def alternating_folds(N, one_indexed=True):
    """
    I_0 = odd times (1,3,5,...) -> 0-based [0,2,4,...]
    I_1 = even times (2,4,6,...) -> 0-based [1,3,5,...]
    """
    all_idx = np.arange(N)
    if one_indexed:
        idx0 = all_idx[0::2]
        idx1 = all_idx[1::2]
    else:
        idx0 = all_idx[::2]
        idx1 = all_idx[1::2]
    return idx0, idx1


def _precompute_fold_quadratics(KFF, R, Delta):
    KDelta = KFF @ Delta
    KR = KFF @ R
    v_dd = np.sum(Delta * (KFF @ Delta), axis=0)
    v_dr = np.sum(Delta * KR, axis=0)
    v_rr = np.sum(R * KR, axis=0)
    return v_dd, v_dr, v_rr, KDelta, KR


def _build_mu_R_Delta(KXX_fold, A_fold, lam):
    """
    Chronological KRR-based CME rows for each arm

    Inputs:
      - KXX_fold: (n×n) Gram on X within the fold, in chronological order.
      - A_fold  : (n,) actions in the same chronological order.
      - lam     : ridge parameter (λ).

    Returns (all n×n, chronological-by-fold):
      - R     = I - μ
      - Delta = μ_1 - μ_0   with rows filled only for their own arm:
                rows of controls get  (- μ_0 rows), rows of treated get (+ μ_1 rows)
    """
    n = A_fold.size
    idx_c = np.where(A_fold == 0)[0]
    idx_t = np.where(A_fold == 1)[0]

    if idx_c.size == 0 or idx_t.size == 0:
        raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

    # Blocks
    K_cc = KXX_fold[np.ix_(idx_c, idx_c)]
    K_tt = KXX_fold[np.ix_(idx_t, idx_t)]
    K_c_all = KXX_fold[np.ix_(idx_c, np.arange(n))]  # controls vs ALL cols (chrono)
    K_t_all = KXX_fold[np.ix_(idx_t, np.arange(n))]  # treated  vs ALL cols (chrono)

    # Solve Arm-wise ridge systems to get CME *rows* in chronological column order
    try:
        mu0_rows = np.linalg.solve(
            K_cc + lam * np.eye(K_cc.shape[0]), K_c_all
        )  # shape (m_c, n)
    except np.linalg.LinAlgError:
        mu0_rows = np.linalg.lstsq(
            K_cc + lam * np.eye(K_cc.shape[0]), K_c_all, rcond=None
        )[0]

    try:
        mu1_rows = np.linalg.solve(
            K_tt + lam * np.eye(K_tt.shape[0]), K_t_all
        )  # shape (m_t, n)
    except np.linalg.LinAlgError:
        mu1_rows = np.linalg.lstsq(
            K_tt + lam * np.eye(K_tt.shape[0]), K_t_all, rcond=None
        )[0]

    mu = np.zeros((n, n), dtype=KXX_fold.dtype)
    mu[idx_c, :] = mu0_rows
    mu[idx_t, :] = mu1_rows

    # R and Δ in chronological order
    R = np.eye(n, dtype=KXX_fold.dtype) - mu

    Delta = np.zeros_like(mu)
    # rows of controls carry -μ0; rows of treated carry +μ1
    Delta[idx_c, :] = -mu0_rows
    Delta[idx_t, :] = +mu1_rows

    return R, Delta


def _fold_omegas(KFF, R, Delta, A_fold, w_fold, Pi_fold_on_fold):
    n = A_fold.size
    denom = np.where(A_fold == 0, 1.0 - w_fold, w_fold)

    v_dd, v_dr, v_rr, KDelta, KR = _precompute_fold_quadratics(KFF, R, Delta)
    omega = np.zeros(n, dtype=float)

    for t in range(n):
        p_t = Pi_fold_on_fold[t]
        s_t = np.where(A_fold == 0, -1.0 / (1.0 - p_t), 1.0 / p_t)
        num = np.where(A_fold == 0, 1.0 - p_t, p_t)
        rho = num / denom

        past_mask = (np.arange(n) < t).astype(float)
        S_size = int(past_mask.sum())
        if S_size == 0:
            omega[t] = 0.0
            continue

        u_t = (rho * past_mask) / S_size
        v_t = s_t * u_t

        q_diag = v_dd + 2.0 * s_t * v_dr + (s_t**2) * v_rr
        M2 = np.sum((rho * past_mask) * q_diag) / S_size

        z = Delta @ u_t
        q = R @ v_t
        Kz = KDelta @ u_t
        Kq = KR @ v_t
        M1_sq = z @ Kz + 2.0 * (z @ Kq) + q @ Kq

        var_t = M2 - M1_sq
        omega[t] = 0.0 if var_t <= 0.0 else 1.0 / np.sqrt(var_t)

    return omega


def xMMD2_vsdr_fold_generic(
    Y, w, X, A, kernel_function, Pi_0_on_0, Pi_1_on_1, idx0, idx1, lam=None, **kwargs
):
    Y = np.asarray(Y)
    Y = Y[:, None] if Y.ndim == 1 else Y
    X = np.asarray(X)
    X = X[:, None] if X.ndim == 1 else X
    A = np.asarray(A, int).ravel()
    w = np.asarray(w).ravel()

    # Split folds
    Y_split0, Y_split1 = Y[idx0], Y[idx1]
    X_split0, X_split1 = X[idx0], X[idx1]
    A_split0, A_split1 = A[idx0], A[idx1]
    w_split0, w_split1 = w[idx0], w[idx1]

    # Choose gamma from second fold (as before); lam defaults to gamma
    gamma = np.median(pairwise_distances(X_split1, X_split1, metric="euclidean")) ** 2
    if lam is None:
        lam = gamma

    # Covariate kernel within each fold
    KX0 = pairwise_kernels(X_split0, X_split0, metric="rbf", gamma=1.0 / gamma)
    KX1 = pairwise_kernels(X_split1, X_split1, metric="rbf", gamma=1.0 / gamma)

    # Dual-space R, Δ
    R0, Delta0 = _build_mu_R_Delta(KX0, A_split0, lam)
    R1, Delta1 = _build_mu_R_Delta(KX1, A_split1, lam)

    # Observed DR multipliers
    W_split0_obs = np.where(A_split0 == 0, -1.0 / (1.0 - w_split0), 1.0 / w_split0)
    W_split1_obs = np.where(A_split1 == 0, -1.0 / (1.0 - w_split1), 1.0 / w_split1)

    # DR operators
    M0 = Delta0 + R0 * W_split0_obs[None, :]
    M1 = Delta1 + R1 * W_split1_obs[None, :]

    # Outcome kernels
    K00 = pairwise_kernels(Y_split0, Y_split0, metric=kernel_function, **kwargs)
    K11 = pairwise_kernels(Y_split1, Y_split1, metric=kernel_function, **kwargs)
    K01 = pairwise_kernels(Y_split0, Y_split1, metric=kernel_function, **kwargs)

    # Cross matrix
    G0 = M0.T @ K01 @ M1

    # Variance stabilization weights
    omega_split0 = _fold_omegas(K00, R0, Delta0, A_split0, w_split0, Pi_0_on_0)
    omega_split1 = _fold_omegas(K11, R1, Delta1, A_split1, w_split1, Pi_1_on_1)

    # Stabilize + normalize
    G = G0 / (omega_split0[:, None] * omega_split1[None, :] + 1e-12)
    S = np.mean(G)
    psi_cross_hat = np.mean(G**2)
    T_stat = np.sqrt(G.shape[0] * G.shape[1]) * S / np.sqrt(psi_cross_hat + 1e-12)
    return float(T_stat)


def xMMD2_vsdr_fold(
    XY,
    w,
    Xcov,
    A,
    kernel_function,
    Pi_fold0_on_fold0,
    Pi_fold1_on_fold1,
    split="chronological",
    one_indexed=True,
    lam=None,
    **kwargs
):
    """
    Unified VS-DR xKTE:
      split = "chronological" -> first half vs second half
      split = "alternating"   -> odd vs even times (controlled by one_indexed)
    Pi_* matrices must be aligned with the *chronological order within each chosen fold*.
    """
    N = len(A)
    if split == "chronological":
        idx0, idx1 = chronological_folds(N)
    elif split == "alternating":
        idx0, idx1 = alternating_folds(N, one_indexed=one_indexed)
    else:
        raise ValueError("split must be 'chronological' or 'alternating'.")

    return xMMD2_vsdr_fold_generic(
        XY,
        w,
        Xcov,
        A,
        kernel_function,
        Pi_0_on_0=Pi_fold0_on_fold0,
        Pi_1_on_1=Pi_fold1_on_fold1,
        idx0=idx0,
        idx1=idx1,
        lam=lam,
        **kwargs
    )
