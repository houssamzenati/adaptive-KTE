# %%
import numpy as np
from numpy.random import default_rng
from filelock import FileLock
import pathlib

# dsprite_adaptive_collector.py
import numpy as np
from filelock import FileLock
import pathlib

DATA_PATH = pathlib.Path(__file__).resolve().parent.joinpath("data")


def _load_dsprite():
    with FileLock("./data.lock"):
        z = np.load(
            DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )
    imgs = z["imgs"].astype(np.float32)
    meta = z["metadata"][()]
    latents_sizes = meta[b"latents_sizes"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
    )
    return imgs, latents_bases


def _image_id(latent_bases, posX_id_arr, posY_id_arr):
    n = posX_id_arr.shape[0]
    idx = np.stack(
        [
            np.zeros(n, dtype=int),  # color
            np.full(n, 2, dtype=int),  # shape=2
            np.zeros(n, dtype=int),  # scale
            np.zeros(n, dtype=int),  # orientation
            posX_id_arr,
            posY_id_arr,
        ],
        axis=1,
    )
    return idx.dot(latent_bases)


def _render_xy(U, a, shift=(0, 0)):
    ux = U[:, 0] + (shift[0] if a == 1 else 0.0)
    uy = U[:, 1] + (shift[1] if a == 1 else 0.0)
    posX = np.clip((ux * 32).astype(int), 0, 31)
    posY = np.clip((uy * 32).astype(int), 0, 31)
    return posX, posY


def _render_image(U, A, imgs, latents_bases, shift=(0, 0)):
    posX, posY = _render_xy(U, A, shift=shift)
    ids = _image_id(latents_bases, posX, posY)
    return imgs[ids]  # (n,64,64)


def _phi_mean_pixel(Y_batch):
    # Y_batch: (m,64,64) or (64,64); returns (m,)
    if Y_batch.ndim == 2:
        return np.array([Y_batch.mean()])
    return Y_batch.mean(axis=(1, 2))


class _OnlineRidge2Arms:
    # same structure as your previous collector (bias coordinate unpenalized)
    def __init__(self, d, lam):
        self.d = d
        self.S = [np.diag([0.0] + [lam] * d), np.diag([0.0] + [lam] * d)]
        self.b = [np.zeros(d + 1), np.zeros(d + 1)]
        self.th = [np.zeros(d + 1), np.zeros(d + 1)]

    def solve(self, a):
        try:
            self.th[a] = np.linalg.solve(self.S[a], self.b[a])
        except np.linalg.LinAlgError:
            self.th[a] = np.linalg.lstsq(self.S[a], self.b[a], rcond=None)[0]

    def predict_scores(self, Z):
        # Z: (m,d+1) with bias
        return np.column_stack([Z @ self.th[0], Z @ self.th[1]])

    def update(self, z, a, y):
        self.S[a] += np.outer(z, z)
        self.b[a] += z * y
        self.solve(a)


def _eps_at(t, eps0, eps_min, power):
    return max(eps_min, eps0 / ((t + 1) ** power))


def _pi1_from_scores(q0, q1, eps_t):
    if q1 > q0:
        return 1.0 - 0.5 * eps_t
    elif q1 < q0:
        return 0.5 * eps_t
    else:
        return 0.5


def collect_adaptive_kte_dsprite(
    ns,
    d,
    scenario,
    eps0=0.2,
    eps_min=0.05,
    power=0.99,
    lam=1e-2,
    rng=None,
    split="alternating",
    one_indexed=True,
    shift_pixels=4,
):
    """
    Returns the same tuple/signature as your collect_epsilon_greedy:
    X, T, Y2d, w, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all
    Where:
      - X: (ns,d) contexts (we use d=2 from [0,1]^2 latents but keep param d for parity)
      - T: (ns,) binary actions
      - Y2d: (ns, 64*64) flattened images for compatibility with your kernels
      - w: (ns,) π_t(1|X_t) at time t
      - Pi_*: fold-wise predictable propensities for realized arms
      - P_all: (ns,ns) rows t are π_t(1|X_s) for all s
    """
    rng = rng or np.random.RandomState(0)
    # contexts: first 2 dims used, rest filler if d>2
    X2 = rng.uniform(0.0, 1.0, size=(ns, 2))
    if d > 2:
        X_rest = rng.randn(ns, d - 2)
        X = np.hstack([X2, X_rest])
    else:
        X = X2

    imgs, latents_bases = _load_dsprite()
    px = shift_pixels / 32.0
    shift = (0.0, 0.0) if scenario == "I" else (px, px)

    # design matrix with bias first
    X_aug = np.hstack(
        [np.ones((ns, 1)), X2]
    )  # only the 2 spatial features drive policy
    mdl = _OnlineRidge2Arms(d=2, lam=lam)

    T = np.zeros(ns, dtype=int)
    w = np.zeros(ns, dtype=float)
    Y_img = np.zeros((ns, 64, 64), dtype=np.float32)
    P_all = np.empty((ns, ns), dtype=np.float32)

    # tiny warm-start
    mdl.th[0] = rng.randn(3) * 1e-6
    mdl.th[1] = rng.randn(3) * 1e-6

    # store snapshots
    th0_snap = np.zeros((ns, 3))
    th1_snap = np.zeros((ns, 3))

    for t in range(ns):
        th0_snap[t] = mdl.th[0]
        th1_snap[t] = mdl.th[1]

        z_t = X_aug[t]
        q0_t = z_t @ mdl.th[0]
        q1_t = z_t @ mdl.th[1]
        eps_t = _eps_at(t, eps0, eps_min, power)
        pi1_t = _pi1_from_scores(q0_t, q1_t, eps_t)

        a_t = 1 if rng.rand() < pi1_t else 0
        y_t = _render_image(
            X2[t : t + 1], np.array([a_t]), imgs, latents_bases, shift=shift
        )[0]
        r_t = _phi_mean_pixel(y_t)

        T[t] = a_t
        Y_img[t] = y_t
        w[t] = pi1_t

        # fill full π_t(1|X_s) row using model at time t
        S_all = X_aug @ np.column_stack([mdl.th[0], mdl.th[1]])  # (ns,2)
        q0_all, q1_all = S_all[:, 0], S_all[:, 1]
        P_all[t] = np.where(
            q1_all > q0_all,
            1.0 - 0.5 * eps_t,
            np.where(q1_all < q0_all, 0.5 * eps_t, 0.5),
        )

        # update with realized arm
        a = a_t
        mdl.update(z_t, a, r_t)

    # choose folds
    if split == "alternating":
        idx0 = np.arange(0, ns, 2)
        idx1 = np.arange(1, ns, 2)
    elif split == "chronological":
        cut = ns // 2
        idx0 = np.arange(0, cut)
        idx1 = np.arange(cut, ns)
    else:
        raise ValueError("split must be 'alternating' or 'chronological'.")

    # fold-wise predictable propensities for realized arms, using snapshots at t
    Z0, Z1 = X_aug[idx0], X_aug[idx1]
    N0, N1 = idx0.size, idx1.size

    Pi_0_on_0 = np.empty((N0, N0), dtype=float)
    for r, t in enumerate(idx0):
        eps_t = _eps_at(t, eps0, eps_min, power)
        q0 = Z0 @ th0_snap[t]
        q1 = Z0 @ th1_snap[t]
        pi1_vec = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )
        Pi_0_on_0[r] = np.where(T[idx0] == 1, pi1_vec, 1.0 - pi1_vec)

    Pi_1_on_1 = np.empty((N1, N1), dtype=float)
    for r, t in enumerate(idx1):
        eps_t = _eps_at(t, eps0, eps_min, power)
        q0 = Z1 @ th0_snap[t]
        q1 = Z1 @ th1_snap[t]
        pi1_vec = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )
        Pi_1_on_1[r] = np.where(T[idx1] == 1, pi1_vec, 1.0 - pi1_vec)

    # flatten images to (ns, 64*64) for your kernel code path
    Y2d = Y_img.reshape(ns, -1)
    return X, T, Y2d, w, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all
