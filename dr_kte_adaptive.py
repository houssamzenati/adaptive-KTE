import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances


def alternating_blocks(n: int):
    """
    A = indices 0,2,4,... ; B = indices 1,3,5,...
    (un nombre sur deux)
    """
    return np.arange(0, n, 2, dtype=int), np.arange(1, n, 2, dtype=int)


def xMMD2dr_stabilized_adaptive(
    XY, w, Xcov, T, kernel_function, idxA, idxB,
    var_floor: float = 1e-8, clip_eps: float = 1e-6, **parameters
):
    """
    Always-STABILIZED DR-xKTE^2 with explicit blocks (idxA, idxB).

    Changes vs your DR-KPE core:
      • Uses explicit A/B blocks (here provided as idxA/idxB).
      • ALWAYS applies variance renormalization (no flag).

    Parameters
    ----------
    XY : (N, d_y) array
        Outcomes Y (can be scalar or vector).
    w : (N,) or (N,1) array
        Logging probabilities / propensities π(X).
    Xcov : (N, d_x) array
        Covariates X for CME construction.
    T : (N,) array of {0,1}
        Treatments.
    kernel_function : str or callable
        Passed to sklearn.metrics.pairwise_kernels for the Y-kernel (e.g., 'rbf').
    idxA, idxB : 1D integer arrays
        Disjoint index sets for the two cross-fitting blocks.
    var_floor : float
        Floor for the stabilization denominators.
    clip_eps : float
        Clipping for π and 1−π before inversion (IPW stability).
    **parameters :
        Extra kernel parameters for pairwise_kernels on Y (e.g., gamma for RBF).

    Returns
    -------
    t_stat : float
        Studentized cross-U statistic (z-style).
    """
    # ---- inputs --------------------------------------------------------------
    XY   = np.asarray(XY)
    Xcov = np.asarray(Xcov)
    T    = np.asarray(T, dtype=int).ravel()
    w    = np.asarray(w).squeeze()

    idxA = np.asarray(idxA, dtype=int)
    idxB = np.asarray(idxB, dtype=int)

    # ---- split by blocks / treatment ----------------------------------------
    Ta, Tb = T[idxA], T[idxB]
    Ya, Yb = XY[idxA], XY[idxB]

    Y0a, Y1a = Ya[Ta == 0], Ya[Ta == 1]
    Y0b, Y1b = Yb[Tb == 0], Yb[Tb == 1]
    Y = np.vstack((Y0a, Y1a, Y0b, Y1b))

    m1, n1 = len(Y0a), len(Y1a)
    m2, n2 = len(Y0b), len(Y1b)
    m, n   = m1 + m2, n1 + n2
    N2A    = m1 + n1
    N2B    = m2 + n2

    Xa, Xb = Xcov[idxA, :], Xcov[idxB, :]
    X0a, X1a = Xa[Ta == 0, :], Xa[Ta == 1, :]
    X0b, X1b = Xb[Tb == 0, :], Xb[Tb == 1, :]
    X = np.vstack((X0a, X1a, X0b, X1b))

    # ---- IPW multipliers s_i (IPW part of φ) -------------------------------
    wa = np.clip(w[idxA].squeeze(), clip_eps, 1 - clip_eps)
    wb = np.clip(w[idxB].squeeze(), clip_eps, 1 - clip_eps)
    w0a = 1.0 / (1.0 - wa[Ta == 0])
    w1a = 1.0 / (wa[Ta == 1])
    w0b = 1.0 / (1.0 - wb[Tb == 0])
    w1b = 1.0 / (wb[Tb == 1])
    ww  = np.concatenate((-w0a, w1a, -w0b, w1b))  

    # ---- CME blocks --------
    if len(Xb) >= 2:
        sigmaKX = np.median(pairwise_distances(Xb, Xb, metric='euclidean')) ** 2
        if not np.isfinite(sigmaKX) or sigmaKX <= 0:
            sigmaKX = np.var(Xb) + 1e-6
    else:
        sigmaKX = np.var(X) + 1e-6

    KX = pairwise_kernels(X, metric='rbf', gamma=1.0 / sigmaKX)
    gamma = sigmaKX  # same naming as your original

    # A-block CMEs
    mu0a = np.linalg.solve(KX[:m1, :m1] + gamma * np.eye(m1), KX[:m1, :m1 + n1])
    zeroed_mu0a = np.vstack((mu0a, np.zeros((n1, m1 + n1))))
    mu1a = np.linalg.solve(KX[m1:m1 + n1, m1:m1 + n1] + gamma * np.eye(n1),
                           KX[m1:m1 + n1, :m1 + n1])
    zeroed_mu1a = np.vstack((np.zeros((m1, m1 + n1)), mu1a))
    muAa = np.hstack((zeroed_mu0a[:, :m1], zeroed_mu1a[:, m1:m1 + n1]))      # (N2A x N2A)

    # B-block CMEs
    mu0b = np.linalg.solve(KX[m1 + n1:m + n1, m1 + n1:m + n1] + gamma * np.eye(m2),
                           KX[m1 + n1:m + n1, m1 + n1:])
    zeroed_mu0b = np.vstack((mu0b, np.zeros((n2, m2 + n2))))
    mu1b = np.linalg.solve(KX[m + n1:, m + n1:] + gamma * np.eye(n2),
                           KX[m + n1:, m1 + n1:])
    zeroed_mu1b = np.vstack((np.zeros((m2, m2 + n2)), mu1b))
    muAb = np.hstack((zeroed_mu0b[:, :m2], zeroed_mu1b[:, m2:]))            # (N2B x N2B)

    # ---- Build φ blocks (matrix form of φ = s(k(·,Y)−μ_A) + Δμ) -------------
    # Column-wise broadcast by ww.
    left_side  = (zeroed_mu1a - zeroed_mu0a) + (np.eye(N2A) - muAa) * ww[:N2A][None, :]
    right_side = (zeroed_mu1b - zeroed_mu0b) + (np.eye(N2B) - muAb) * ww[N2A:][None, :]

    # ---- All cross inner products ⟨φ_i, φ_j⟩ via Y-kernel -------------------
    KY = pairwise_kernels(Y[:N2A], Y[N2A:], metric=kernel_function, **parameters)  # (N2A x N2B)
    G  = left_side.T @ KY @ right_side                                            # (N2A x N2B)

    # ---- Variance renormalization (row/col second moments) --------
    row_var = np.mean(G * G, axis=1)              # σ_i^2 proxy, i ∈ A
    col_var = np.mean(G * G, axis=0)              # σ_j^2 proxy, j ∈ B
    row_std = np.sqrt(np.maximum(row_var, var_floor))
    col_std = np.sqrt(np.maximum(col_var, var_floor))
    G = G / (row_std[:, None] * col_std[None, :])

    # ---- Cross-U aggregation + studentization --------------------------------
    U = G.mean(axis=1).ravel()                    # U_i = mean_j G_{ij}
    t_stat = np.sqrt(len(U)) * (U.mean() / (U.std() + 1e-12))
    return t_stat


def xMMD2dr_stabilized_alternating(
    XY, w, Xcov, T, kernel_function, var_floor: float = 1e-8, clip_eps: float = 1e-6, **parameters
):
    """
    Backward-compatible wrapper using alternating indices:
      A = 0,2,4,... ; B = 1,3,5,...
    Always stabilized (no flag).
    """
    N = len(XY)
    idxA, idxB = alternating_blocks(N)
    return xMMD2dr_stabilized_adaptive(
        XY, w, Xcov, T, kernel_function, idxA, idxB,
        var_floor=var_floor, clip_eps=clip_eps, **parameters
    )
