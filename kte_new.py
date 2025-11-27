from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
import scipy.stats as st
from sklearn.metrics import pairwise_distances

# Code extracted from https://github.com/sorawitj/counterfactual-mean-embedding/


def MMD2u(K, w, m, n):
    if m < 2 or n < 2:
        return 0.0
    """The MMD^2_u unbiased statistic."""
    # wx = 1.0/np.array(w[:m])[:,np.newaxis]
    # wy = 1.0/np.array(1.0 - w[m:])[:,np.newaxis]
    # group 1 = controls (T=0)
    wx = 1.0 / np.array(1.0 - w[:m])[:, np.newaxis]  # 1 / (1 - e)
    # group 2 = treated (T=1)
    wy = 1.0 / np.array(w[m:])[:, np.newaxis]  # 1 / e

    Kx = np.outer(wx, wx) * K[:m, :m]
    Ky = np.outer(wy, wy) * K[m:, m:]
    Kxy = np.outer(wx, wy) * K[:m, m:]

    return (
        1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        + 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        - 2.0 / (m * n) * Kxy.sum()
    )


def compute_null_distribution(
    K, w, m, n, iterations=10000, verbose=False, random_state=None, marker_interval=1000
):
    """Compute the bootstrap null-distribution of MMD2u."""
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        w_i = w[idx]
        mmd2u_null[i] = MMD2u(K_i, w_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def compute_null_distribution(
    K, w, m, n, iterations=1000, verbose=False, random_state=None, marker_interval=1000
):
    """
    CORRECTED: Uses Propensity Bootstrap instead of Permutation.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    
    # 'w' contains the propensity pi(1|X) for every unit in K.
    # We use it to simulate valid counterfactual assignments.
    
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
            
        # 1. Re-sample assignment T based on propensity w
        # This prevents "impossible" weights (like treated unit forced to control)
        t_bootstrap = rng.binomial(1, w)
        
        # 2. Identify new split indices
        idx0 = np.where(t_bootstrap == 0)[0]
        idx1 = np.where(t_bootstrap == 1)[0]
        m_new, n_new = len(idx0), len(idx1)
        
        # Safety check for degenerate splits
        if m_new < 2 or n_new < 2:
            mmd2u_null[i] = 0.0 # or skip
            continue
            
        # 3. Re-sort K and w to match MMD2u's expectation (Control first, then Treated)
        new_order = np.concatenate([idx0, idx1])
        
        K_i = K[np.ix_(new_order, new_order)]
        w_i = w[new_order]
        
        # 4. Compute Statistic
        mmd2u_null[i] = MMD2u(K_i, w_i, m_new, n_new)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test_nonuniform(
    X,
    Y,
    T,
    Prob_vec,
    kernel_function="rbf",
    iterations=500,
    verbose=False,
    random_state=None,
    **kwargs
):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    w = np.concatenate((Prob_vec[T == 0], Prob_vec[T == 1]))
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, w, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(
        K, w, m, n, iterations, verbose=verbose, random_state=random_state
    )
    # p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
    #              float(iterations))
    p_value = np.mean(mmd2u_null > mmd2u)

    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value
