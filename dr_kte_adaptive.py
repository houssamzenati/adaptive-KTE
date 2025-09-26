# import numpy as np
# from sklearn.metrics import pairwise_kernels, pairwise_distances
# import scipy.stats as st


# # ---------- helpers


# def chronological_folds(N):
#     """
#     I_0 = {0,...,N//2-1}, I_1 = {N//2,...,N-1} (0-based).
#     """
#     cut = N // 2
#     return np.arange(cut), np.arange(cut, N)


# def alternating_folds(N, one_indexed=True):
#     """
#     I_0 = odd times (1,3,5,...) -> 0-based [0,2,4,...]
#     I_1 = even times (2,4,6,...) -> 0-based [1,3,5,...]
#     """
#     all_idx = np.arange(N)
#     if one_indexed:
#         idx0 = all_idx[0::2]
#         idx1 = all_idx[1::2]
#     else:
#         idx0 = all_idx[::2]
#         idx1 = all_idx[1::2]
#     return idx0, idx1


# def _precompute_fold_quadratics(KFF, R, Delta):
#     KDelta = KFF @ Delta
#     KR = KFF @ R
#     v_dd = np.sum(Delta * (KFF @ Delta), axis=0)
#     v_dr = np.sum(Delta * KR, axis=0)
#     v_rr = np.sum(R * KR, axis=0)
#     return v_dd, v_dr, v_rr, KDelta, KR


# def _build_mu_R_Delta(KXX_fold, A_fold, lam):
#     n = A_fold.size
#     idx0 = np.where(A_fold == 0)[0]
#     idx1 = np.where(A_fold == 1)[0]
#     if idx0.size == 0 or idx1.size == 0:
#         raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

#     K00 = KXX_fold[np.ix_(idx0, idx0)]
#     K11 = KXX_fold[np.ix_(idx1, idx1)]
#     Kr0 = KXX_fold[:, idx0]  # K_{r,0}
#     Kr1 = KXX_fold[:, idx1]  # K_{r,1}

#     # Column selectors E0, E1 (place identity into the correct columns)
#     E0 = np.zeros((idx0.size, n))
#     E0[np.arange(idx0.size), idx0] = 1.0
#     E1 = np.zeros((idx1.size, n))
#     E1[np.arange(idx1.size), idx1] = 1.0

#     mu0 = Kr0 @ np.linalg.solve(
#         K00 + lam * np.eye(K00.shape[0]), E0
#     )  # n×n, nonzero only at idx0 columns
#     mu1 = Kr1 @ np.linalg.solve(
#         K11 + lam * np.eye(K11.shape[0]), E1
#     )  # n×n, nonzero only at idx1 columns

#     mu = mu0 + mu1
#     R = np.eye(n) - mu
#     Delta = mu1 - mu0
#     return R, Delta


# # def _build_mu_R_Delta(KXX_fold, A_fold, lam):
# #     n = A_fold.size
# #     idx_control = np.where(A_fold == 0)[0]
# #     idx_treated = np.where(A_fold == 1)[0]
# #     if idx_control.size == 0 or idx_treated.size == 0:
# #         raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

# #     m_control, n_treated = idx_control.size, idx_treated.size
# #     cols_perm = np.r_[idx_control, idx_treated]  # arm-stacked order

# #     # Gram blocks in the chronological index system
# #     K_cc = KXX_fold[np.ix_(idx_control, idx_control)]
# #     K_ct = KXX_fold[np.ix_(idx_control, idx_treated)]
# #     K_tc = KXX_fold[np.ix_(idx_treated, idx_control)]
# #     K_tt = KXX_fold[np.ix_(idx_treated, idx_treated)]

# #     # Dual operators on the arm-stacked column order [controls, treated]
# #     mu0 = np.linalg.solve(K_cc + lam * np.eye(m_control), np.hstack([K_cc, K_ct]))
# #     mu1 = np.linalg.solve(K_tt + lam * np.eye(n_treated), np.hstack([K_tc, K_tt]))

# #     # Build μ in the ARM-STACKED row+col system first
# #     mu_arm = np.zeros((n, n))
# #     # rows: controls → mu0 ; treated → mu1 ; columns already arm-stacked
# #     mu_arm[idx_control, :][:, cols_perm] = mu0
# #     mu_arm[idx_treated, :][:, cols_perm] = mu1

# #     mu_chrono = np.zeros((n, n))
# #     mu_chrono[:, cols_perm] = mu_arm

# #     # Split into μ̃1 and μ̃0 parts in chronological order
# #     mu1_pad = np.zeros((n, n))
# #     mu1_pad[idx_treated, :] = mu_chrono[idx_treated, :]
# #     mu0_pad = np.zeros((n, n))
# #     mu0_pad[idx_control, :] = mu_chrono[idx_control, :]

# #     mu = mu0_pad + mu1_pad
# #     R = np.eye(n) - mu
# #     Delta = mu1_pad - mu0_pad
# #     return R, Delta


# # def _build_mu_R_Delta(KXX_fold, A_fold, lam):
# #     """
# #     Chronological KRR-based CME rows for each arm, no reordering, no padding.

# #     Inputs:
# #       - KXX_fold: (n×n) Gram on X within the fold, in chronological order.
# #       - A_fold  : (n,) actions in the same chronological order.
# #       - lam     : ridge parameter (λ).

# #     Returns (all n×n, chronological-by-fold):
# #       - R     = I - μ
# #       - Delta = μ_1 - μ_0   with rows filled only for their own arm:
# #                 rows of controls get  (- μ_0 rows), rows of treated get (+ μ_1 rows)
# #     """
# #     n = A_fold.size
# #     idx_c = np.where(A_fold == 0)[0]
# #     idx_t = np.where(A_fold == 1)[0]

# #     if idx_c.size == 0 or idx_t.size == 0:
# #         raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

# #     # Blocks to *solve* (rows restricted to arm, columns = all, still chronological)
# #     K_cc = KXX_fold[np.ix_(idx_c, idx_c)]
# #     K_tt = KXX_fold[np.ix_(idx_t, idx_t)]
# #     K_c_all = KXX_fold[np.ix_(idx_c, np.arange(n))]  # controls vs ALL cols (chrono)
# #     K_t_all = KXX_fold[np.ix_(idx_t, np.arange(n))]  # treated  vs ALL cols (chrono)

# #     # Solve Arm-wise ridge systems to get CME *rows* in chronological column order
# #     # (K_aa + λI) * μ_a_rows = K_a,all  ⇒  μ_a_rows = (K_aa + λI)^{-1} K_a,all
# #     try:
# #         mu0_rows = np.linalg.solve(
# #             K_cc + lam * np.eye(K_cc.shape[0]), K_c_all
# #         )  # shape (m_c, n)
# #     except np.linalg.LinAlgError:
# #         mu0_rows = np.linalg.lstsq(
# #             K_cc + lam * np.eye(K_cc.shape[0]), K_c_all, rcond=None
# #         )[0]

# #     try:
# #         mu1_rows = np.linalg.solve(
# #             K_tt + lam * np.eye(K_tt.shape[0]), K_t_all
# #         )  # shape (m_t, n)
# #     except np.linalg.LinAlgError:
# #         mu1_rows = np.linalg.lstsq(
# #             K_tt + lam * np.eye(K_tt.shape[0]), K_t_all, rcond=None
# #         )[0]

# #     # Assemble μ in *chronological* row+col system (no reindexing anywhere)
# #     mu = np.zeros((n, n), dtype=KXX_fold.dtype)
# #     mu[idx_c, :] = mu0_rows
# #     mu[idx_t, :] = mu1_rows

# #     # R and Δ in chronological order
# #     R = np.eye(n, dtype=KXX_fold.dtype) - mu

# #     Delta = np.zeros_like(mu)
# #     # rows of controls carry -μ0; rows of treated carry +μ1
# #     Delta[idx_c, :] = -mu0_rows
# #     Delta[idx_t, :] = +mu1_rows

# #     return R, Delta


# def _fold_omegas(KFF, R, Delta, A_fold, w_fold, Pi_fold_on_fold):
#     n = A_fold.size
#     denom = np.where(A_fold == 0, 1.0 - w_fold, w_fold)
#     v_dd, v_dr, v_rr, KDelta, KR = _precompute_fold_quadratics(KFF, R, Delta)

#     omega = np.zeros(n, dtype=float)
#     for t in range(n):
#         p_t = Pi_fold_on_fold[t]
#         s_t = np.where(A_fold == 0, -1.0 / (1.0 - p_t), 1.0 / p_t)
#         num = np.where(A_fold == 0, 1.0 - p_t, p_t)
#         rho = num / denom

#         past_mask = (np.arange(n) < t).astype(float)
#         S_size = int(past_mask.sum())
#         if S_size == 0:
#             omega[t] = 1
#             continue

#         u_t = (rho * past_mask) / S_size
#         v_t = s_t * u_t

#         q_diag = v_dd + 2.0 * s_t * v_dr + (s_t**2) * v_rr
#         M2 = np.sum((rho * past_mask) * q_diag) / S_size

#         z = Delta @ u_t
#         q = R @ v_t
#         Kz = KDelta @ u_t
#         Kq = KR @ v_t
#         M1_sq = z @ Kz + 2.0 * (z @ Kq) + q @ Kq

#         var_t = M2 - M1_sq
#         omega[t] = 0.0 if var_t <= 0.0 else 1.0 / np.sqrt(var_t)
#     return omega


# # def _fold_omegas(
# #     KFF, R, Delta, A_fold, w_fold, Pi_fold_on_fold, clip_eps=1e-12, var_floor=1e-12
# # ):
# #     n = A_fold.size
# #     w_fold = np.clip(w_fold, clip_eps, 1.0 - clip_eps)
# #     denom = np.where(A_fold == 0, 1.0 - w_fold, w_fold)

# #     KDelta = KFF @ Delta
# #     KR = KFF @ R
# #     v_dd = np.sum(Delta * KDelta, axis=0)
# #     v_dr = np.sum(Delta * KR, axis=0)
# #     v_rr = np.sum(R * KR, axis=0)

# #     Pi_arr = np.asarray(Pi_fold_on_fold)
# #     use_matrix = Pi_arr.ndim == 2 and Pi_arr.shape[0] >= n

# #     omega = np.zeros(n, dtype=float)
# #     ar = np.arange(n)

# #     for t in range(n):
# #         if use_matrix:
# #             p_ts = np.clip(Pi_arr[t, :].reshape(-1), clip_eps, 1.0 - clip_eps)
# #         else:
# #             p_t = float(np.asarray(Pi_arr[t]).reshape(-1)[0])
# #             p_ts = np.full(n, np.clip(p_t, clip_eps, 1.0 - clip_eps))

# #         s_t = np.where(A_fold == 0, -1.0 / (1.0 - p_ts), 1.0 / p_ts)
# #         num = np.where(A_fold == 0, 1.0 - p_ts, p_ts)
# #         rho = num / denom

# #         past = (ar < t).astype(float)
# #         S = int(past.sum())
# #         if S == 0:
# #             omega[t] = np.nan  # we’ll drop these later
# #             continue

# #         u_t = (rho * past) / S
# #         v_t = s_t * u_t

# #         q_diag = v_dd + 2.0 * s_t * v_dr + (s_t**2) * v_rr
# #         M2 = np.sum((rho * past) * q_diag) / S

# #         z = Delta @ u_t
# #         q = R @ v_t
# #         Kz = KDelta @ u_t
# #         Kq = KR @ v_t
# #         M1_sq = z @ Kz + 2.0 * (z @ Kq) + q @ Kq

# #         var_t = max(M2 - M1_sq, var_floor)
# #         omega[t] = 1.0 / np.sqrt(var_t)

# #     return omega


# def xMMD2_vsdr_fold_generic(
#     Y, w, X, A, kernel_function, Pi_0_on_0, Pi_1_on_1, idx0, idx1, lam=None, **kwargs
# ):
#     Y = np.asarray(Y)
#     Y = Y[:, None] if Y.ndim == 1 else Y
#     X = np.asarray(X)
#     X = X[:, None] if X.ndim == 1 else X
#     A = np.asarray(A, int).ravel()
#     w = np.asarray(w).ravel()

#     # Split folds
#     Y_split0, Y_split1 = Y[idx0], Y[idx1]
#     X_split0, X_split1 = X[idx0], X[idx1]
#     A_split0, A_split1 = A[idx0], A[idx1]
#     w_split0, w_split1 = w[idx0], w[idx1]

#     # Choose gamma from second fold (as before); lam defaults to gamma
#     gamma = np.median(pairwise_distances(X_split1, X_split1, metric="euclidean")) ** 2
#     if lam is None:
#         lam = gamma

#     # Covariate kernel within each fold
#     KX0 = pairwise_kernels(X_split0, X_split0, metric="rbf", gamma=1.0 / gamma)
#     KX1 = pairwise_kernels(X_split1, X_split1, metric="rbf", gamma=1.0 / gamma)

#     # Dual-space R, Δ
#     R0, Delta0 = _build_mu_R_Delta(KX0, A_split0, lam)
#     R1, Delta1 = _build_mu_R_Delta(KX1, A_split1, lam)

#     # Observed DR multipliers
#     W_split0_obs = np.where(A_split0 == 0, -1.0 / (1.0 - w_split0), 1.0 / w_split0)
#     W_split1_obs = np.where(A_split1 == 0, -1.0 / (1.0 - w_split1), 1.0 / w_split1)

#     # DR operators
#     M0 = Delta0 + R0 * W_split0_obs[None, :]
#     M1 = Delta1 + R1 * W_split1_obs[None, :]

#     # Outcome kernels
#     K00 = pairwise_kernels(Y_split0, Y_split0, metric=kernel_function, **kwargs)
#     K11 = pairwise_kernels(Y_split1, Y_split1, metric=kernel_function, **kwargs)
#     K01 = pairwise_kernels(Y_split0, Y_split1, metric=kernel_function, **kwargs)

#     # Cross matrix
#     G0 = M0.T @ K01 @ M1

#     # Variance stabilization weights
#     omega_split0 = _fold_omegas(K00, R0, Delta0, A_split0, w_split0, Pi_0_on_0)
#     omega_split1 = _fold_omegas(K11, R1, Delta1, A_split1, w_split1, Pi_1_on_1)

#     # # Stabilize + normalize
#     # G = G0 / (omega_split0[:, None] * omega_split1[None, :] + 1e-12)

#     valid0 = np.isfinite(omega_split0) & (omega_split0 > 0)
#     valid1 = np.isfinite(omega_split1) & (omega_split1 > 0)
#     if not np.any(valid0) or not np.any(valid1):
#         raise ValueError(
#             "No valid indices after filtering stabilizers (check Π alignment)."
#         )

#     G0 = G0[np.ix_(valid0, valid1)]
#     omega0 = omega_split0[valid0]
#     omega1 = omega_split1[valid1]

#     # then do the stabilization with only valid rows/cols (no +1e-12 needed):
#     G = G0 / (omega0[:, None] * omega1[None, :])

#     S = np.mean(G)
#     psi_cross_hat = np.mean(G**2)

#     T_stat = np.sqrt(G.shape[0] * G.shape[1]) * S / np.sqrt(psi_cross_hat + 1e-12)
#     return float(T_stat)


# def xMMD2_vsdr_fold(
#     XY,
#     w,
#     Xcov,
#     A,
#     kernel_function,
#     Pi_fold0_on_fold0,
#     Pi_fold1_on_fold1,
#     split="chronological",
#     one_indexed=True,
#     lam=None,
#     **kwargs
# ):
#     """
#     Unified VS-DR xKTE:
#       - split = "chronological" -> first half vs second half
#       - split = "alternating" -> odd vs even times (controlled by one_indexed)

#     Pi_* matrices must be aligned with the *chronological order
#     within each chosen fold*.
#     """
#     N = len(A)
#     if split == "chronological":
#         idx0, idx1 = chronological_folds(N)
#     elif split == "alternating":
#         idx0, idx1 = alternating_folds(N, one_indexed=one_indexed)
#     else:
#         raise ValueError("split must be 'chronological' or 'alternating'.")

#     return xMMD2_vsdr_fold_generic(
#         XY,
#         w,
#         Xcov,
#         A,
#         kernel_function,
#         Pi_0_on_0=Pi_fold0_on_fold0,
#         Pi_1_on_1=Pi_fold1_on_fold1,
#         idx0=idx0,
#         idx1=idx1,
#         lam=lam,
#         **kwargs
#     )

import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances


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
    if idx_control.size == 0 or idx_treated.size == 0:
        raise ValueError("A fold has no samples for one arm; cannot build Δ and R.")

    m_control, n_treated = idx_control.size, idx_treated.size
    cols_perm = np.r_[idx_control, idx_treated]

    K_cc = KXX_fold[np.ix_(idx_control, idx_control)]
    K_ct = KXX_fold[np.ix_(idx_control, idx_treated)]
    K_tc = KXX_fold[np.ix_(idx_treated, idx_control)]
    K_tt = KXX_fold[np.ix_(idx_treated, idx_treated)]

    mu0 = np.linalg.solve(K_cc + lam * np.eye(m_control), np.hstack([K_cc, K_ct]))
    mu1 = np.linalg.solve(K_tt + lam * np.eye(n_treated), np.hstack([K_tc, K_tt]))

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


# ---------- per-fold conditional variances ω
def _fold_omegas(KFF, R, Delta, A_fold, w_fold, Pi_fold_on_fold):
    eps = 1e-12
    n = A_fold.size

    # π_s(A_s|X_s) under logging (per-time s)
    denom = np.where(A_fold == 0, 1.0 - w_fold, w_fold)  # shape (n,)

    v_dd, v_dr, v_rr, KDelta, KR = _precompute_fold_quadratics(KFF, R, Delta)

    omega = np.zeros(n, dtype=float)
    for t in range(n):
        # π_t(A_s|X_s) over s=0..n-1 (must be a vector length n)
        p_t = Pi_fold_on_fold[:, t] if Pi_fold_on_fold.ndim == 2 else Pi_fold_on_fold
        if p_t.shape[0] != n:
            p_t = Pi_fold_on_fold[t]  # fallback if stored transposed (n,) expected

        s_t = np.where(
            A_fold == 0, -1.0 / (1.0 - p_t + eps), 1.0 / (p_t + eps)
        )  # multiplier for DR score at time t (applied to all s)
        num = np.where(A_fold == 0, 1.0 - p_t, p_t)  # π_t(A_s|X_s)
        rho = (num + eps) / (denom + eps)  # importance ratio π_t/π_s on realized arm

        past = np.arange(n) < t
        S = past.sum()
        if S == 0:
            omega[t] = 0.0
            continue

        u_t = (rho * past) / S
        v_t = s_t * u_t

        q_diag = v_dd + 2.0 * s_t * v_dr + (s_t**2) * v_rr
        M2 = np.sum((rho * past) * q_diag) / S

        z = Delta @ u_t
        q = R @ v_t
        Kz = KDelta @ u_t
        Kq = KR @ v_t
        M1_sq = z @ Kz + 2.0 * (z @ Kq) + q @ Kq

        var_t = M2 - M1_sq
        omega[t] = 0.0 if var_t <= 0.0 else 1.0 / np.sqrt(var_t + eps)
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

    KX0 = pairwise_kernels(X0, X0, metric="rbf", gamma=1.0 / gamma)
    KX1 = pairwise_kernels(X1, X1, metric="rbf", gamma=1.0 / gamma)

    R0, Delta0 = _build_mu_R_Delta(KX0, A0, lam)
    R1, Delta1 = _build_mu_R_Delta(KX1, A1, lam)

    W0 = np.where(A0 == 0, -1.0 / (1.0 - w0 + eps), 1.0 / (w0 + eps))
    W1 = np.where(A1 == 0, -1.0 / (1.0 - w1 + eps), 1.0 / (w1 + eps))

    M0 = Delta0 + np.diag(W0) @ R0
    M1 = Delta1 + np.diag(W1) @ R1

    K01 = pairwise_kernels(Y0, Y1, metric=kernel_function, **kwargs)
    G0 = M0.T @ K01 @ M1

    omega0 = _fold_omegas(
        KFF=pairwise_kernels(Y0, Y0, metric=kernel_function, **kwargs),
        R=R0,
        Delta=Delta0,
        A_fold=A0,
        w_fold=w0,
        Pi_fold_on_fold=Pi_0_on_0,
    )
    omega1 = _fold_omegas(
        KFF=pairwise_kernels(Y1, Y1, metric=kernel_function, **kwargs),
        R=R1,
        Delta=Delta1,
        A_fold=A1,
        w_fold=w1,
        Pi_fold_on_fold=Pi_1_on_1,
    )

    G = G0 / (omega0[:, None] * omega1[None, :] + eps)
    S = np.mean(G)
    psi_hat = np.mean(G**2)

    return float(np.sqrt(G.size) * S / np.sqrt(psi_hat + eps))
