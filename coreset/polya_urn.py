import numpy as np


def polya_urn(
    data: np.ndarray,
    sample_base,
    alpha: float = 1.0,
    L: int = 500,
    rng: np.random.Generator = None,
    reinforce: bool = True,
) -> np.ndarray:
    """
    Blackwell–MacQueen Pólya urn (paper, Eq. (urn)): given y_{1:m} and the previous draws
    Ỹ_{1:j−1}, the next draw comes from F_phi with probability alpha / (alpha + m + j − 1), and
    otherwise repeats one of the m observations or j − 1 previous draws, chosen uniformly.

    data        : (m, d) conditioning dataset
    sample_base : callable () -> (d,) draw from the parametric base measure F_phi
    alpha       : DP precision parameter
    L           : trajectory length
    rng         : numpy Generator for reproducibility
    reinforce   : True (default) is the reinforced urn above; False draws L i.i.d. points from
                  the posterior centring measure F_phi^m (no reinforcement).

    Returns (L, d) trajectory array.
    """
    return polya_urn_weighted(data, None, sample_base, alpha=alpha, L=L, rng=rng, reinforce=reinforce)


def polya_urn_weighted(
    data: np.ndarray,
    weights,
    sample_base,
    alpha: float = 1.0,
    L: int = 500,
    rng: np.random.Generator = None,
    reinforce: bool = True,
) -> np.ndarray:
    """
    Pólya urn for a weighted conditioning sample (e.g. a likelihood-weighted coreset): observation i
    carries mass weights[i] (None = unit masses), F_phi carries mass alpha and each previous draw
    mass one.  With unit weights this is polya_urn.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    m, d = data.shape
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    W = float(w.sum())
    cdf = np.cumsum(w) / W
    unit = weights is None
    trajectory = np.empty((L, d))
    for j in range(L):
        prev = j if reinforce else 0                      # number of previous draws in the urn
        u = rng.uniform() * (alpha + W + prev)
        if u < alpha:
            trajectory[j] = sample_base()
        elif u < alpha + W:
            i = rng.integers(0, m) if unit else min(int(np.searchsorted(cdf, rng.uniform(), side='right')), m - 1)
            trajectory[j] = data[i]
        else:
            trajectory[j] = trajectory[rng.integers(0, prev)]
    return trajectory
