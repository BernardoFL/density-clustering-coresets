import numpy as np


# Fixed means for Experiment 1 — same across all 100 runs (grid mixture)
EXP1_MEANS = np.array([[i * 5.0, j * 5.0] for i in range(3) for j in range(2)])


def generate_curved_mixture_hd(
    N: int = 10_000,
    d: int = 20,
    n_common: int = 3,
    n_rare: int = 5,
    rare_weight: float = 0.01,
    spread: float = 8.0,
    common_std: float = 1.2,
    shape_scale: float = 1.8,
    noise_std: float = 0.12,
    nested: bool = False,
    rare_offset_frac: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Misspecified mixture: common Gaussian blobs + rare non-Gaussian curved
    components embedded in R^d.

    Common (n_common): isotropic Gaussian blobs, std = common_std in all dims.
    Rare   (n_rare):   banana / half-moon / ring / arc / comet shapes in 2
                       random active dimensions; noise_std elsewhere.

    nested=False (default): rare cluster centers are randomly placed in 20D
        (well-separated from common clusters, easy to identify from 1 point).
    nested=True: rare cluster centers are placed INSIDE common cluster regions
        (offset = rare_offset_frac * common_std from a parent common center).
        In this regime the 18 non-active dims have noise std = noise_std, so
        rare points are hard to distinguish from common-cluster outliers without
        diverse shape coverage in the 2 active dims.

    Returns (data, assignments, weights, component_types):
      data           : (N, d)
      assignments    : (N,)  component index in [0, K)
      weights        : (K,)  mixture weights
      component_types: list[str] of length K
    """
    if rng is None:
        rng = np.random.default_rng()

    K = n_common + n_rare
    w_common = (1.0 - n_rare * rare_weight) / max(n_common, 1)
    weights = np.array([w_common] * n_common + [rare_weight] * n_rare, dtype=float)
    weights /= weights.sum()

    if nested:
        # Common centers: well-separated in 20D
        common_centers = rng.normal(0.0, spread, size=(n_common, d))
        # Rare centers: offset from a random parent common center
        parents = rng.choice(n_common, size=n_rare)
        offsets = rng.normal(0.0, rare_offset_frac * common_std, size=(n_rare, d))
        rare_centers = common_centers[parents] + offsets
        centers = np.vstack([common_centers, rare_centers])
    else:
        centers = rng.normal(0.0, spread, size=(K, d))
    assignments = rng.choice(K, size=N, p=weights)

    _shapes = ['banana', 'half_moon', 'ring', 'arc', 'comet']
    component_types = ['gaussian'] * n_common + [
        _shapes[(k - n_common) % len(_shapes)] for k in range(n_common, K)
    ]

    data = np.zeros((N, d))

    for k in range(K):
        mask = assignments == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue

        pts = rng.normal(0.0, noise_std, size=(n_k, d))

        if k < n_common:
            pts = rng.normal(0.0, common_std, size=(n_k, d))

        else:
            k_rare = k - n_common
            shape = _shapes[k_rare % len(_shapes)]
            active = rng.choice(d, size=2, replace=False)
            d1, d2 = int(active[0]), int(active[1])

            if shape == 'banana':
                # Parabolic arc; zero-mean by construction (E[u²-1]=0)
                u = rng.normal(0.0, 1.0, size=n_k)
                pts[:, d1] = (shape_scale / 2.5) * u
                pts[:, d2] = (shape_scale / 4.0) * (u ** 2 - 1.0) + rng.normal(0, noise_std, size=n_k)

            elif shape == 'half_moon':
                # Upper semicircle; d2 centered by subtracting E[r·sin θ]
                theta = rng.uniform(0.0, np.pi, size=n_k)
                r = shape_scale + rng.normal(0.0, noise_std * 1.5, size=n_k)
                pts[:, d1] = r * np.cos(theta)
                pts[:, d2] = r * np.sin(theta) - 2.0 * shape_scale / np.pi

            elif shape == 'ring':
                theta = rng.uniform(0.0, 2.0 * np.pi, size=n_k)
                r = shape_scale * 0.75 + rng.normal(0.0, noise_std * 1.5, size=n_k)
                pts[:, d1] = r * np.cos(theta)
                pts[:, d2] = r * np.sin(theta)

            elif shape == 'arc':
                # 300-degree arc (C-shape); d1 centered for mean of cos on U(π/6, 11π/6)
                theta = rng.uniform(np.pi / 6.0, 11.0 * np.pi / 6.0, size=n_k)
                r = shape_scale * 0.75 + rng.normal(0.0, noise_std * 1.5, size=n_k)
                pts[:, d1] = r * np.cos(theta) + shape_scale * 0.75 * 3.0 / (5.0 * np.pi)
                pts[:, d2] = r * np.sin(theta)

            elif shape == 'comet':
                # Dense core (2/3) + one-sided exponential tail (1/3); centered
                n_core = max(n_k * 2 // 3, 1)
                n_tail = n_k - n_core
                pts[:n_core, d1] = rng.normal(0.0, noise_std * 2, size=n_core)
                pts[:n_core, d2] = rng.normal(0.0, noise_std * 2, size=n_core)
                if n_tail > 0:
                    pts[n_core:, d1] = rng.exponential(shape_scale / 2, size=n_tail)
                    pts[n_core:, d2] = rng.normal(0.0, noise_std * 2, size=n_tail)
                pts[:, d1] -= shape_scale / 6.0  # subtract E[tail offset] * (1/3)

        data[mask] = centers[k] + pts

    return data, assignments, weights, component_types
