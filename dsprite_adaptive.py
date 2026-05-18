import numpy as np
from filelock import FileLock
import pathlib

# -------------------------
# dataset I/O
# -------------------------
DATA_PATH = pathlib.Path(__file__).resolve().parent.joinpath("data")


def _load_dsprite():
    with FileLock(str(DATA_PATH.joinpath("data.lock"))):
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
            np.full(n, 2, dtype=int),  # shape = 2 (heart)
            np.zeros(n, dtype=int),  # scale
            np.zeros(n, dtype=int),  # orientation
            posX_id_arr,
            posY_id_arr,
        ],
        axis=1,
    )
    return idx.dot(latent_bases)


# -------------------------
# renderer (U,A) -> image
# -------------------------
def _render_xy(U, A, shift=(0.0, 0.0), mode="shift"):
    """
    Vectorized. U: (n,2) in [0,1]. A: (n,) in {0,1}.
    mode='shift'    : small translation when A=1
    mode='quadrant' : A=0 → bottom-left 16x16; A=1 → top-right 16x16
    """
    U = np.asarray(U)
    A = np.asarray(A).reshape(-1)
    if mode == "shift":
        shx = (A == 1).astype(float) * shift[0]
        shy = (A == 1).astype(float) * shift[1]
        ux = U[:, 0] + shx
        uy = U[:, 1] + shy
        posX = np.clip((ux * 32).astype(int), 0, 31)
        posY = np.clip((uy * 32).astype(int), 0, 31)
        return posX, posY
    elif mode == "quadrant":
        x16 = np.clip((U[:, 0] * 16).astype(int), 0, 15)
        y16 = np.clip((U[:, 1] * 16).astype(int), 0, 15)
        posX = x16 + (A * 16)  # 0..15 vs 16..31
        posY = y16 + (A * 16)
        return posX, posY
    else:
        raise ValueError("mode must be 'shift' or 'quadrant'")


def _render_image(U, A, imgs, latents_bases, shift=(0.0, 0.0), mode="shift"):
    posX, posY = _render_xy(U, A, shift=shift, mode=mode)
    ids = _image_id(latents_bases, posX, posY)
    return imgs[ids].astype(np.float32)  # (n,64,64)


def _get_base_heart(imgs, latents_bases, px=16, py=16):
    # one canonical heart; same shape/scale/orientation, central-ish location
    idx = _image_id(latents_bases, np.array([px]), np.array([py]))[0]
    return imgs[idx].astype(np.float32)  # (64,64)


def _render_image_roll_quadrant(U, A, base, scenario):
    """
    Mean-preserving renderer:
      - compute coarse positions x,y in {0,...,31} from U∈[0,1]^2
      - Scenario I: both arms use (x,y)
      - Scenario IV: A=0 uses (x,y), A=1 uses (x+32,y+32)
    All placements use np.roll → permutation of pixels → exact same mean.
    """
    U = np.asarray(U, dtype=np.float32)
    A = np.asarray(A).reshape(-1)

    # coarse coordinates 0..31
    x = np.clip((U[:, 0] * 32).astype(int), 0, 31)
    y = np.clip((U[:, 1] * 32).astype(int), 0, 31)

    if scenario == "IV":
        offx = (A == 1).astype(int) * 32
        offy = (A == 1).astype(int) * 32
    else:  # Scenario I (null): no arm-dependent offset
        offx = np.zeros_like(x)
        offy = np.zeros_like(y)

    rx = x + offx  # A=0: 0..31; A=1: 32..63   (no wrap-around)
    ry = y + offy

    out = np.empty((U.shape[0], 64, 64), dtype=np.float32)
    for i in range(U.shape[0]):
        out[i] = np.roll(np.roll(base, ry[i], axis=0), rx[i], axis=1)
    return out


# -------------------------
# simple bandit (ε-greedy)
# -------------------------
def _phi_mean_pixel(Y_batch):
    # reward proxy (keeps bandit near 50/50; translation-invariant)
    if Y_batch.ndim == 2:
        return float(Y_batch.mean())
    return Y_batch.mean(axis=(1, 2))


class _OnlineRidge2Arms:
    """Per-arm online ridge with unpenalized bias (first coord)."""

    def __init__(self, d, lam):
        self.d = d
        self.S = [np.diag([0.0] + [lam] * d), np.diag([0.0] + [lam] * d)]
        self.b = [np.zeros(d + 1), np.zeros(d + 1)]
        self.th = [np.zeros(d + 1), np.zeros(d + 1)]

    def _solve(self, a):
        try:
            self.th[a] = np.linalg.solve(self.S[a], self.b[a])
        except np.linalg.LinAlgError:
            self.th[a] = np.linalg.lstsq(self.S[a], self.b[a], rcond=None)[0]

    def predict_scores(self, Z):  # Z: (m,d+1) with bias
        return np.column_stack([Z @ self.th[0], Z @ self.th[1]])

    def update(self, z, a, y):
        self.S[a] += np.outer(z, z)
        self.b[a] += z * y
        self._solve(a)


def _eps_at(t, eps0, eps_min, power):
    return max(eps_min, eps0 / ((t + 1) ** power))


def _pi1_from_scores(q0, q1, eps_t):
    if q1 > q0:
        return 1.0 - 0.5 * eps_t
    if q1 < q0:
        return 0.5 * eps_t
    return 0.5


# -------------------------
# main collector
# -------------------------
def collect_adaptive_kte_dsprite(
    ns,
    d,
    scenario,
    eps0=0.5,
    eps_min=0.2,
    power=0.5,
    lam=1e-4,
    rng=None,
    split="alternating",
    one_indexed=True,  # kept for API parity (unused)
    renderer_mode=None,
    shift_pixels=8,
):
    """
    Returns: X, T, Y2d, w, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all
    - X: (ns,d) contexts (first 2 used for image rendering)
    - T: (ns,) actions in {0,1}
    - Y2d: (ns,4096) flattened images (float32)
    - w: (ns,) π_t(1|X_t) at decision t
    - Pi_*: fold-wise predictable propensities for the realized arms
    - idx0, idx1: fold indices
    - P_all: (ns,ns) π_t(1|X_s) rows for all s using parameters at time t
    """
    rng = rng or np.random.RandomState(0)

    # contexts: first 2 ∈ [0,1], remaining (if any) are noise features
    X2 = rng.uniform(0.0, 1.0, size=(ns, 2)).astype(np.float32)
    if d > 2:
        X = np.hstack([X2, rng.randn(ns, d - 2).astype(np.float32)])
    else:
        X = X2

    imgs, latents_bases = _load_dsprite()
    base = _get_base_heart(imgs, latents_bases)

    # Scenario I: no effect; Scenario IV: strong quadrant effect by default
    px = float(shift_pixels) / 32.0
    shift = (0.0, 0.0) if scenario == "I" else (px, px)
    mode = "shift" if scenario == "I" else (renderer_mode or "quadrant")

    # model that learns propensities (only the 2 spatial features drive it)
    X_aug = np.hstack([np.ones((ns, 1), dtype=np.float32), X2])
    mdl = _OnlineRidge2Arms(d=2, lam=lam)
    mdl.th[0] = rng.randn(3) * 1e-6
    mdl.th[1] = rng.randn(3) * 1e-6

    T = np.zeros(ns, dtype=int)
    w = np.zeros(ns, dtype=np.float32)  # π_t(1|X_t)
    Y_img = np.zeros((ns, 64, 64), dtype=np.float32)
    P_all = np.empty((ns, ns), dtype=np.float32)

    th0_snap = np.zeros((ns, 3), dtype=np.float32)
    th1_snap = np.zeros((ns, 3), dtype=np.float32)

    for t in range(ns):
        th0_snap[t] = mdl.th[0]
        th1_snap[t] = mdl.th[1]

        z_t = X_aug[t]
        q0_t = z_t @ mdl.th[0]
        q1_t = z_t @ mdl.th[1]
        eps_t = _eps_at(t, eps0, eps_min, power)
        pi1_t = _pi1_from_scores(q0_t, q1_t, eps_t)

        a_t = 1 if rng.rand() < pi1_t else 0
        y_t = _render_image_roll_quadrant(
            X2[t : t + 1], np.array([a_t]), base, scenario
        )[0]

        r_t = _phi_mean_pixel(y_t)

        T[t] = a_t
        Y_img[t] = y_t
        w[t] = pi1_t

        # full row π_t(1|X_s) using snapshot at time t
        S_all = X_aug @ np.column_stack([mdl.th[0], mdl.th[1]])
        q0_all, q1_all = S_all[:, 0], S_all[:, 1]
        P_all[t] = np.where(
            q1_all > q0_all,
            1.0 - 0.5 * eps_t,
            np.where(q1_all < q0_all, 0.5 * eps_t, 0.5),
        )

        mdl.update(z_t, a_t, r_t)

    # folds for cross-fitting
    if split == "alternating":
        idx0 = np.arange(0, ns, 2)
        idx1 = np.arange(1, ns, 2)
    elif split == "chronological":
        cut = ns // 2
        idx0 = np.arange(0, cut)
        idx1 = np.arange(cut, ns)
    else:
        raise ValueError("split must be 'alternating' or 'chronological'.")

    Z0, Z1 = X_aug[idx0], X_aug[idx1]
    Pi_0_on_0 = np.empty((idx0.size, idx0.size), dtype=np.float32)
    for r, t in enumerate(idx0):
        eps_t = _eps_at(t, eps0, eps_min, power)
        q0 = Z0 @ th0_snap[t]
        q1 = Z0 @ th1_snap[t]
        pi1_vec = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )
        Pi_0_on_0[r] = np.where(T[idx0] == 1, pi1_vec, 1.0 - pi1_vec)

    Pi_1_on_1 = np.empty((idx1.size, idx1.size), dtype=np.float32)
    for r, t in enumerate(idx1):
        eps_t = _eps_at(t, eps0, eps_min, power)
        q0 = Z1 @ th0_snap[t]
        q1 = Z1 @ th1_snap[t]
        pi1_vec = np.where(
            q1 > q0, 1.0 - 0.5 * eps_t, np.where(q1 < q0, 0.5 * eps_t, 0.5)
        )
        Pi_1_on_1[r] = np.where(T[idx1] == 1, pi1_vec, 1.0 - pi1_vec)

    Y2d = Y_img.reshape(ns, -1)  # (ns,4096)
    return X, T, Y2d, w, Pi_0_on_0, Pi_1_on_1, idx0, idx1, P_all

# --- plotting & sanity checks that match the roll-based renderer ---


def _base_heart():
    imgs, latents_bases = _load_dsprite()
    # pick a canonical center-ish heart once
    idx = _image_id(latents_bases, np.array([16]), np.array([16]))[0]
    return imgs[idx].astype(np.float32)


def plot_observational_samples(scenario="IV", n_per_arm=6, seed=0):
    import matplotlib.pyplot as plt

    # Uses the collector, which already renders with _render_image_roll_quadrant
    X, T, Y2d, w, *_ = collect_adaptive_kte_dsprite(
        ns=200, d=2, scenario=scenario, rng=np.random.RandomState(seed)
    )
    Y = Y2d.reshape(-1, 64, 64)

    idx0 = np.where(T == 0)[0][:n_per_arm]
    idx1 = np.where(T == 1)[0][:n_per_arm]
    k = min(len(idx0), len(idx1))
    idx0, idx1 = idx0[:k], idx1[:k]

    fig, axes = plt.subplots(2, k, figsize=(1.8 * k, 3.8))
    for j, i in enumerate(idx0):
        axes[0, j].imshow(Y[i], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"A=0 (i={i})", fontsize=9)
        axes[0, j].axis("off")
    for j, i in enumerate(idx1):
        axes[1, j].imshow(Y[i], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"A=1 (i={i})", fontsize=9)
        axes[1, j].axis("off")

    fig.suptitle(f"Observational samples • Scenario {scenario} (roll renderer)", y=0.98)
    plt.tight_layout()
    pathlib.Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/observational_dsprite.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_counterfactual_pairs(scenario="IV", n=6, seed=1):
    import matplotlib.pyplot as plt

    # Render A=0 and A=1 for the same contexts with the roll renderer
    rng = np.random.RandomState(seed)
    X, *_ = collect_adaptive_kte_dsprite(
        ns=max(2 * n, 32), d=2, scenario=scenario, rng=rng
    )
    U = X[:n, :2]

    base = _base_heart()
    Y0 = _render_image_roll_quadrant(U, np.zeros(n, dtype=int), base, scenario)
    Y1 = _render_image_roll_quadrant(U, np.ones(n, dtype=int), base, scenario)

    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 3.8))
    for j in range(n):
        axes[0, j].imshow(Y0[j], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title("A=0", fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(Y1[j], cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title("A=1", fontsize=9)
        axes[1, j].axis("off")
    fig.suptitle(f"Counterfactual pairs • Scenario {scenario} (roll renderer)", y=0.98)
    plt.tight_layout()
    pathlib.Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/counterfactual_pairs_dsprite.png", dpi=300, bbox_inches="tight")
    plt.show()

    # mean invariance printout
    dmean = Y1.mean(axis=(1, 2)) - Y0.mean(axis=(1, 2))
    print("Δ mean pixel (A1 − A0):", np.round(dmean, 6))


def sanity_check_mean_invariance(scenario="IV", n=2000, seed=0):
    """
    Verifies that the roll-based renderer preserves the image mean across arms.
    """
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 1, size=(n, 2))
    base = _base_heart()
    Y0 = _render_image_roll_quadrant(U, np.zeros(n, dtype=int), base, scenario)
    Y1 = _render_image_roll_quadrant(U, np.ones(n, dtype=int), base, scenario)

    m0 = Y0.mean(axis=(1, 2))
    m1 = Y1.mean(axis=(1, 2))
    diff = m1 - m0
    out = {
        "scenario": scenario,
        "n": n,
        "mean_A0": float(m0.mean()),
        "mean_A1": float(m1.mean()),
        "mean_diff": float(diff.mean()),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "allclose": bool(np.allclose(m0, m1)),
    }
    print("[Mean invariance check]", out)
    return out

if __name__ == "__main__":
    plot_observational_samples(scenario="IV", n_per_arm=6, seed=0)
    plot_counterfactual_pairs(scenario="IV", n=6, seed=1)
    sanity_check_mean_invariance(scenario="IV", n=5000, seed=3)
