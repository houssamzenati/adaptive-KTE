from math import erfc, sqrt

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels


def chronological_folds(n_samples):
    cut = n_samples // 2
    return np.arange(cut), np.arange(cut, n_samples)


def median_rbf_gamma(X):
    X = _ensure_2d(X)
    if X.shape[0] < 2:
        return 1.0
    distances_sq = pairwise_distances(X, X, metric="euclidean") ** 2
    upper = distances_sq[np.triu_indices_from(distances_sq, k=1)]
    median_sq = np.median(upper)
    if not np.isfinite(median_sq) or median_sq <= 0:
        median_sq = float(np.var(X)) + 1e-6
    return 1.0 / median_sq


def outcome_rbf_gamma(Y, A):
    Y = _ensure_2d(Y)
    A = np.asarray(A, dtype=int).ravel()
    y0 = Y[A == 0]
    y1 = Y[A == 1]
    if y0.size == 0 or y1.size == 0:
        return 1.0 / (float(np.var(Y)) + 1e-6)
    median = np.median(pairwise_distances(y0, y1, metric="euclidean"))
    sigma_sq = (median**2) / 4.0
    if not np.isfinite(sigma_sq) or sigma_sq <= 0:
        sigma_sq = float(np.var(Y)) + 1e-6
    return 1.0 / sigma_sq


class _SequentialKernelInverse:
    def __init__(self, kernel_block, ridge):
        self.kernel_block = np.asarray(kernel_block, dtype=float)
        self.ridge = float(ridge)
        self.past_idx = []
        self.inverse = None

    def predict_coefficients(self, target_idx):
        if not self.past_idx:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        cross = self.kernel_block[np.ix_(self.past_idx, [target_idx])].ravel()
        coeffs = self.inverse @ cross
        return np.asarray(self.past_idx, dtype=int), coeffs

    def update(self, obs_idx):
        diag_value = float(self.kernel_block[obs_idx, obs_idx] + self.ridge)
        if self.inverse is None:
            self.inverse = np.array([[1.0 / max(diag_value, 1e-10)]], dtype=float)
            self.past_idx.append(obs_idx)
            return

        cross = self.kernel_block[np.ix_(self.past_idx, [obs_idx])].ravel()
        inv_cross = self.inverse @ cross
        schur = diag_value - float(cross @ inv_cross)
        schur = max(schur, 1e-10)

        top_left = self.inverse + np.outer(inv_cross, inv_cross) / schur
        top_right = (-inv_cross / schur)[:, None]
        bottom_left = top_right.T
        bottom_right = np.array([[1.0 / schur]], dtype=float)

        self.inverse = np.block(
            [[top_left, top_right], [bottom_left, bottom_right]]
        )
        self.past_idx.append(obs_idx)


def projected_adaptive_kte_test(
    Y,
    X,
    A,
    policy_matrix,
    idx_pilot=None,
    idx_infer=None,
    outcome_kernel="rbf",
    covariate_kernel="rbf",
    outcome_gamma=None,
    covariate_gamma=None,
    ridge=1e-2,
    clip_sd=None,
    clip_exponent=0.25,
):
    """
    Two-fold projected adaptive DR-KTE statistic.

    Parameters
    ----------
    Y : array-like, shape (n, d_y) or (n,)
        Observed outcomes.
    X : array-like, shape (n, d_x) or (n,)
        Observed covariates.
    A : array-like, shape (n,)
        Binary actions in {0, 1}.
    policy_matrix : array-like, shape (n, n)
        Entry (t, s) is pi_t(1 | X_s), the evaluation-time arm-1 propensity
        for context X_s.
    idx_pilot, idx_infer : array-like
        Fold indices. Defaults to the chronological split.
    outcome_kernel, covariate_kernel : str
        Kernels passed to sklearn.metrics.pairwise_kernels.
    outcome_gamma, covariate_gamma : float, optional
        RBF gammas for the outcome and covariate kernels.
    ridge : float
        Kernel-ridge regularization level.
    clip_sd : float, optional
        Lower clip for the sequential standard deviation. If omitted, uses the
        manuscript default scale `n_infer^{-clip_exponent}`.
    clip_exponent : float
        Exponent rho in the clip sequence epsilon_n = n^{-rho}.
    """
    Y = _ensure_2d(Y)
    X = _ensure_2d(X)
    A = np.asarray(A, dtype=int).ravel()
    policy_matrix = np.asarray(policy_matrix, dtype=float)

    if idx_pilot is None or idx_infer is None:
        idx_pilot, idx_infer = chronological_folds(len(A))
    idx_pilot = np.asarray(idx_pilot, dtype=int)
    idx_infer = np.asarray(idx_infer, dtype=int)

    if outcome_gamma is None and outcome_kernel == "rbf":
        outcome_gamma = outcome_rbf_gamma(Y, A)
    if covariate_gamma is None and covariate_kernel == "rbf":
        covariate_gamma = median_rbf_gamma(X)
    if clip_sd is None:
        clip_sd = len(idx_infer) ** (-clip_exponent)

    pilot = _projected_fold_statistic(
        Y=Y[idx_pilot],
        X=X[idx_pilot],
        A=A[idx_pilot],
        pi1_matrix=policy_matrix[np.ix_(idx_pilot, idx_pilot)],
        outcome_kernel=outcome_kernel,
        covariate_kernel=covariate_kernel,
        outcome_gamma=outcome_gamma,
        covariate_gamma=covariate_gamma,
        ridge=ridge,
    )

    infer = _inferential_fold_statistic(
        Y_pilot=Y[idx_pilot],
        Y_infer=Y[idx_infer],
        X_infer=X[idx_infer],
        A_infer=A[idx_infer],
        pi1_matrix=policy_matrix[np.ix_(idx_infer, idx_infer)],
        witness_coefficients=pilot["witness_coefficients"],
        outcome_kernel=outcome_kernel,
        covariate_kernel=covariate_kernel,
        outcome_gamma=outcome_gamma,
        covariate_gamma=covariate_gamma,
        ridge=ridge,
        clip_sd=clip_sd,
    )

    statistic = float(np.sum(infer["normalized_scores"]) / np.sqrt(len(idx_infer)))
    return {
        "statistic": statistic,
        "p_value": 0.5 * erfc(statistic / sqrt(2.0)),
        "pilot_witness_norm": float(pilot["witness_norm"]),
        "witness_coefficients": pilot["witness_coefficients"],
        "projected_outcomes": infer["projected_outcomes"],
        "projected_scores": infer["projected_scores"],
        "sigma_hat": infer["sigma_hat"],
        "sigma_tilde": infer["sigma_tilde"],
        "idx_pilot": idx_pilot,
        "idx_infer": idx_infer,
        "outcome_gamma": outcome_gamma,
        "covariate_gamma": covariate_gamma,
        "clip_sd": float(clip_sd),
    }


def projected_adaptive_kte_zstat(*args, **kwargs):
    return projected_adaptive_kte_test(*args, **kwargs)["statistic"]


def _projected_fold_statistic(
    Y,
    X,
    A,
    pi1_matrix,
    outcome_kernel,
    covariate_kernel,
    outcome_gamma,
    covariate_gamma,
    ridge,
):
    n = len(A)
    ky = _kernel(Y, Y, outcome_kernel, outcome_gamma)
    kx = _kernel(X, X, covariate_kernel, covariate_gamma)

    tracker = {
        0: _SequentialKernelInverse(kx, ridge),
        1: _SequentialKernelInverse(kx, ridge),
    }
    average_coeffs = np.zeros(n, dtype=float)
    eps = 1e-10

    for u in range(n):
        pi1_u = np.clip(pi1_matrix[u, u], eps, 1.0 - eps)
        pi0_u = 1.0 - pi1_u

        idx1, alpha1 = tracker[1].predict_coefficients(u)
        idx0, alpha0 = tracker[0].predict_coefficients(u)

        if idx1.size:
            average_coeffs[idx1] += alpha1 / n
        if idx0.size:
            average_coeffs[idx0] -= alpha0 / n

        if A[u] == 1:
            average_coeffs[u] += 1.0 / (n * pi1_u)
            if idx1.size:
                average_coeffs[idx1] -= alpha1 / (n * pi1_u)
        else:
            average_coeffs[u] -= 1.0 / (n * pi0_u)
            if idx0.size:
                average_coeffs[idx0] += alpha0 / (n * pi0_u)

        tracker[A[u]].update(u)

    witness_norm_sq = float(average_coeffs @ ky @ average_coeffs)
    if witness_norm_sq > 1e-12:
        witness_coefficients = average_coeffs / np.sqrt(witness_norm_sq)
        witness_norm = np.sqrt(witness_norm_sq)
    else:
        witness_coefficients = None
        witness_norm = 0.0

    return {
        "witness_coefficients": witness_coefficients,
        "witness_norm": witness_norm,
        "outcome_kernel_block": ky,
    }


def _inferential_fold_statistic(
    Y_pilot,
    Y_infer,
    X_infer,
    A_infer,
    pi1_matrix,
    witness_coefficients,
    outcome_kernel,
    covariate_kernel,
    outcome_gamma,
    covariate_gamma,
    ridge,
    clip_sd,
):
    n = len(A_infer)
    kx = _kernel(X_infer, X_infer, covariate_kernel, covariate_gamma)
    if witness_coefficients is None:
        projected = _reference_projection(Y_infer, outcome_kernel, outcome_gamma)
    else:
        ky_cross = _kernel(Y_infer, Y_pilot, outcome_kernel, outcome_gamma)
        projected = ky_cross @ witness_coefficients

    tracker = {
        0: _SequentialKernelInverse(kx, ridge),
        1: _SequentialKernelInverse(kx, ridge),
    }

    m0 = np.zeros(n, dtype=float)
    m1 = np.zeros(n, dtype=float)
    scores = np.zeros(n, dtype=float)
    sigma_hat = np.ones(n, dtype=float)
    sigma_tilde = np.ones(n, dtype=float)
    eps = 1e-10

    pi1_diag = np.clip(np.diag(pi1_matrix), eps, 1.0 - eps)

    for u in range(n):
        idx1, alpha1 = tracker[1].predict_coefficients(u)
        idx0, alpha0 = tracker[0].predict_coefficients(u)

        m1[u] = float(projected[idx1] @ alpha1) if idx1.size else 0.0
        m0[u] = float(projected[idx0] @ alpha0) if idx0.size else 0.0

        pi1_u = pi1_diag[u]
        pi0_u = 1.0 - pi1_u
        if A_infer[u] == 1:
            scores[u] = (projected[u] - m1[u]) / pi1_u + m1[u] - m0[u]
        else:
            scores[u] = -(projected[u] - m0[u]) / pi0_u + m1[u] - m0[u]

        if u > 0:
            eval_pi1 = np.clip(pi1_matrix[u, :u], eps, 1.0 - eps)
            logged_realized = np.where(A_infer[:u] == 1, pi1_diag[:u], 1.0 - pi1_diag[:u])
            eval_realized = np.where(A_infer[:u] == 1, eval_pi1, 1.0 - eval_pi1)
            rho = eval_realized / (logged_realized + eps)

            recycled = np.where(
                A_infer[:u] == 1,
                (projected[:u] - m1[:u]) / (eval_pi1 + eps),
                -(projected[:u] - m0[:u]) / (1.0 - eval_pi1 + eps),
            )
            recycled += m1[:u] - m0[:u]

            m1_hat = np.mean(rho * recycled)
            m2_hat = np.mean(rho * recycled**2)
            sigma_sq = max(float(m2_hat - m1_hat**2), 0.0)
            sigma_hat[u] = np.sqrt(sigma_sq)

        sigma_tilde[u] = max(sigma_hat[u], clip_sd)
        tracker[A_infer[u]].update(u)

    return {
        "projected_outcomes": projected,
        "m0": m0,
        "m1": m1,
        "projected_scores": scores,
        "sigma_hat": sigma_hat,
        "sigma_tilde": sigma_tilde,
        "normalized_scores": scores / sigma_tilde,
    }


def _ensure_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    return array


def _kernel(XA, XB, kernel_name, gamma):
    kwargs = {"metric": kernel_name}
    if gamma is not None and kernel_name == "rbf":
        kwargs["gamma"] = gamma
    return pairwise_kernels(XA, XB, **kwargs)


def _reference_projection(Y, outcome_kernel, outcome_gamma):
    reference = np.zeros((1, Y.shape[1]), dtype=float)
    values = _kernel(Y, reference, outcome_kernel, outcome_gamma).ravel()
    norm_sq = float(_kernel(reference, reference, outcome_kernel, outcome_gamma)[0, 0])
    return values / np.sqrt(max(norm_sq, 1e-12))
