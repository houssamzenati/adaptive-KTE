import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances


def chronological_folds(N):
    cut = N // 2
    return np.arange(cut), np.arange(cut, N)


def alternating_folds(N, one_indexed=True):
    all_idx = np.arange(N)
    if one_indexed:
        return all_idx[0::2], all_idx[1::2]
    return all_idx[::2], all_idx[1::2]


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


def _safe_median_sq(X):
    X = _ensure_2d(X)
    if X.shape[0] < 2:
        return float(np.var(X)) + 1e-6
    D2 = pairwise_distances(X, X, metric="euclidean") ** 2
    upper = D2[np.triu_indices_from(D2, k=1)]
    med2 = np.median(upper)
    if not np.isfinite(med2) or med2 <= 0:
        med2 = float(np.var(X)) + 1e-6
    return med2


def _reference_projection(Y, outcome_kernel, outcome_gamma):
    reference = np.zeros((1, Y.shape[1]), dtype=float)
    values = _kernel(Y, reference, outcome_kernel, outcome_gamma).ravel()
    norm_sq = float(_kernel(reference, reference, outcome_kernel, outcome_gamma)[0, 0])
    return values / np.sqrt(max(norm_sq, 1e-12))

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

        self.inverse = np.block([[top_left, top_right], [bottom_left, bottom_right]])
        self.past_idx.append(obs_idx)


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


def xMMD2_projected_fold_generic(
    Y, w, X, A, kernel_function, Pi_0_on_0, Pi_1_on_1, idx0, idx1, lam=None, **kwargs
):
    """
    Same signature as xMMD2_vsdr_fold_generic.
    Returns one float statistic.

    Convention:
    - idx0 / Pi_0_on_0 define the pilot fold
    - idx1 / Pi_1_on_1 define the inferential fold
    """
    Y = np.asarray(Y)
    Y = Y[:, None] if Y.ndim == 1 else Y
    X = np.asarray(X)
    X = X[:, None] if X.ndim == 1 else X
    A = np.asarray(A, int).ravel()

    idx0 = np.asarray(idx0, dtype=int)
    idx1 = np.asarray(idx1, dtype=int)

    Y0, Y1 = Y[idx0], Y[idx1]
    X0, X1 = X[idx0], X[idx1]
    A0, A1 = A[idx0], A[idx1]

    outcome_gamma = kwargs.get("gamma", None) if kernel_function == "rbf" else None

    x_scale = _safe_median_sq(X1)
    covariate_gamma = (1.0 / x_scale) if kernel_function == "rbf" else None
    ridge = x_scale if lam is None else lam

    clip_exponent = kwargs.get("clip_exponent", 0.25)
    clip_sd = kwargs.get("clip_sd", len(idx1) ** (-clip_exponent))

    pilot = _projected_fold_statistic(
        Y=Y0,
        X=X0,
        A=A0,
        pi1_matrix=np.asarray(Pi_0_on_0, dtype=float),
        outcome_kernel=kernel_function,
        covariate_kernel=kernel_function,
        outcome_gamma=outcome_gamma,
        covariate_gamma=covariate_gamma,
        ridge=ridge,
    )

    infer = _inferential_fold_statistic(
        Y_pilot=Y0,
        Y_infer=Y1,
        X_infer=X1,
        A_infer=A1,
        pi1_matrix=np.asarray(Pi_1_on_1, dtype=float),
        witness_coefficients=pilot["witness_coefficients"],
        outcome_kernel=kernel_function,
        covariate_kernel=kernel_function,
        outcome_gamma=outcome_gamma,
        covariate_gamma=covariate_gamma,
        ridge=ridge,
        clip_sd=clip_sd,
    )

    return float(np.sum(infer["normalized_scores"]) / np.sqrt(len(idx1)))