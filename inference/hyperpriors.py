import numpy as np
from scipy.stats import invwishart


def sample_niw_hyperprior(
    data: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """
    Sample (mu_t, Sigma_t) from the Normal-InvWishart hyperprior.

    mu_t    ~ N(y_bar, 3 * I_d)
    Sigma_t ~ InvWishart(nu = d+2, Psi = S)

    where y_bar and S are the empirical mean and covariance of `data`.

    Returns (mu_t, Sigma_t) as (d,) and (d, d) arrays.
    """
    if data.ndim == 1:
        data = data[:, None]

    n, d = data.shape
    y_bar = data.mean(axis=0)

    S = np.cov(data.T, ddof=1) if n > 1 else np.eye(d)
    if d == 1:
        S = np.atleast_2d(S)

    # Ensure positive definiteness
    S = S + 1e-6 * np.eye(d)

    nu = d + 2
    Sigma_t = invwishart.rvs(df=nu, scale=S, random_state=rng)
    Sigma_t = np.atleast_2d(Sigma_t) + 1e-6 * np.eye(d)

    mu_t = rng.multivariate_normal(y_bar, 3.0 * np.eye(d))

    return mu_t, Sigma_t


def build_gaussian_base(mu: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator):
    """
    Return a callable () -> (d,) that samples from N(mu, Sigma).
    """
    def sample():
        return rng.multivariate_normal(mu, Sigma)

    return sample
