import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances

def cmpe_kte_test(
    Y,
    X,
    A,
    p,
    kernel_function='rbf',
    reg_lambda=1e-2,
    **kwargs
):
    """
    Adapts the CMPE xMMD2dr statistic for binary Kernel Treatment Effects.
    
    Parameters:
    - Y: Outcomes (N, 1)
    - X: Contexts (N, d)
    - A: Actions (N, ) or (N, 1), binary 0/1
    - p: Propensities pi(1|X) (N, )
    - reg_lambda: Regularization for KRR
    """
    
    # Ensure shapes
    Y = np.asarray(Y).reshape(-1, 1)
    A = np.asarray(A).reshape(-1, 1)
    X = np.asarray(X)
    p = np.asarray(p).reshape(-1, 1)
    N = len(Y)

    # 1. Define Target "Samples" (Deterministic policies)
    # pi_1: Always Action 1
    # pi_0: Always Action 0
    A_1 = np.ones_like(A)
    A_0 = np.zeros_like(A)

    # 2. Define Importance Weights for deterministic targets
    # w_1 = I(A=1) / pi(1|x)
    # w_0 = I(A=0) / pi(0|x)
    # Note: p is probability of 1.
    
    # Avoid division by zero for numerical stability
    p_safe = np.clip(p, 1e-6, 1.0 - 1e-6)
    
    w_pi_1 = (A == 1).astype(float) / p_safe
    w_pi_0 = (A == 0).astype(float) / (1.0 - p_safe)

    # 3. Kernel Setup
    # Kernel for X (Median Heuristic)
    dist_X = pairwise_distances(X, metric='euclidean')**2
    median_X = np.median(dist_X)
    if median_X == 0: median_X = np.mean(dist_X) # Fallback
    sigmaKX = median_X if median_X > 0 else 1.0
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / sigmaKX)

    # Kernel for T (Actions)
    # Since A is binary {0,1}, Euclidean dist sq is either 0 or 1.
    # We can fix gamma=1.0 or compute median (which might be 0 or 1).
    # For safety in binary settings, we force a reasonable bandwidth or use linear.
    # Here we stick to RBF with gamma=1.0 for simplicity on 0/1 data.
    sigmaKT = 1.0 
    KT = pairwise_kernels(A, metric="rbf", gamma=1.0 / sigmaKT)
    
    # Cross kernels between observed actions A and targets A_1/A_0
    KT_pi_1 = pairwise_kernels(A, A_1, metric="rbf", gamma=1.0 / sigmaKT)
    KT_pi_0 = pairwise_kernels(A, A_0, metric="rbf", gamma=1.0 / sigmaKT)

    # 4. Compute Conditional Mean Operators (Ridge Regression)
    # We solve: (K_X * K_T + lambda I) * Beta = (K_X * K_T_target)
    
    # Common regularization term
    K_joint = np.multiply(KX, KT)
    Reg = K_joint + reg_lambda * np.eye(N)
    
    # Solve for logging (in-sample reconstruction)
    # This estimates E[phi(Y) | X, A] evaluated at observed A
    mu_logging = np.linalg.solve(Reg, K_joint)
    
    # Solve for target pi_1 (Action 1)
    # This estimates E[phi(Y) | X, A=1]
    K_joint_1 = np.multiply(KX, KT_pi_1)
    mu_pi_1 = np.linalg.solve(Reg, K_joint_1)
    
    # Solve for target pi_0 (Action 0)
    # This estimates E[phi(Y) | X, A=0]
    K_joint_0 = np.multiply(KX, KT_pi_0)
    mu_pi_0 = np.linalg.solve(Reg, K_joint_0)

    # 5. Doubly Robust Correction Term
    # Formula: (mu_1 - mu_0) + (w_1 - w_0) * (I - mu_logging)
    # This operates on the feature map phi(Y) implicitly via the kernel matrix K_Y later.
    
    # Note: w_pi_1 and w_pi_0 must be broadcastable to (N, N) for elementwise mult with (I - mu)
    # The weights apply to the residuals (I - mu_logging) row-wise.
    
    # The algebraic form in CMPE code:
    # dr_term = mu_pi_prime - mu_pi + (w_pi_prime - w_pi) * (np.eye(N) - mu_logging)
    # Here pi' is 1, pi is 0.
    
    W_diff = (w_pi_1 - w_pi_0) # Shape (N, 1)
    
    # Broadcasting W_diff across columns of the residual operator
    residuals = np.eye(N) - mu_logging
    weighted_residuals = W_diff * residuals 
    
    dr_term = mu_pi_1 - mu_pi_0 + weighted_residuals

    # 6. Compute U-Statistic
    # The test statistic is || \Psi ||^2_H. 
    # In the empirical approximation: u^T K_Y u
    
    # Get Outcome Kernel
    # Re-estimate sigma for Y if needed or use kwargs
    if 'gamma' not in kwargs:
        dist_Y = pairwise_distances(Y, metric='euclidean')**2
        median_Y = np.median(dist_Y)
        sigmaKY = median_Y if median_Y > 0 else 1.0
        gamma_Y = 1.0 / sigmaKY
    else:
        gamma_Y = kwargs['gamma']
        
    KY = pairwise_kernels(Y, metric=kernel_function, gamma=gamma_Y)
    
    # Calculate Quadratic Form: dr_term^T * KY * dr_term
    # Note: dr_term is an operator matrix (N x N). 
    # The actual estimate of the embedding is Sum_i (dr_term_ji * phi(y_i)).
    # The norm squared is: Trace(dr_term.T @ KY @ dr_term) / N^2 roughly?
    # The CMPE implementation does: prod = dr_term.T @ KY @ dr_term
    # U = prod.mean(1) -> This implies averaging over one dimension.
    
    prod = dr_term.T @ KY @ dr_term
    
    # In CMPE paper code: return np.sqrt(len(U)) * U.mean() / U.std()
    # This is a self-normalized statistic (t-stat).
    
    # Extract diagonal or full mean?
    # CMPE implementation treats 'prod' columns as samples of the inner product?
    # Let's strictly follow the provided reference implementation:
    U = prod.mean(axis=1) # Average over the 'training' samples to get evaluation scores?
    
    # Avoid division by zero
    std_U = U.std()
    if std_U == 0: std_U = 1e-9
        
    stat = np.sqrt(N) * U.mean() / std_U
    
    # Compute p-value (Standard Normal assumption)
    from math import erf
    p_value = 0.5 * (1.0 - erf(stat / np.sqrt(2.0)))
    
    return stat, p_value