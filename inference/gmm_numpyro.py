"""
Bayesian K-component Gaussian mixture inference via NumPyro (NUTS/JAX).

NumPyro JIT-compiles the entire NUTS step via JAX, making it 10–50x faster
than PyMC on CPU for the dataset sizes used here.

Model
-----
    w         ~ Dirichlet(1/K, ..., 1/K)
    mu_k      ~ MvNormal(mu0, Psi/kappa)
    L_corr_k  ~ LKJCholesky(eta=2)          Cholesky of correlation matrix
    sigma_k   ~ HalfNormal(prior_scale)     per-dimension scales
    Sigma_k    = diag(sigma_k) L_corr_k L_corr_k^T diag(sigma_k)
    y_i       ~ sum_k w_k N(mu_k, Sigma_k)  (marginalised via log-sum-exp)

Public API
----------
fit_gmm(data, K, draws, tune, chains, target_accept, hyperprior_params, rng)
    Returns dict: 'weights' (S,K), 'mu' (S,K,d), 'chol' (S,K,d,d).
    prior_cov_mu Cholesky is computed on CPU to avoid cuSolver dispatch on GPU.

classify(data, posterior)
    label_i = argmax_k E_s[r_k^(s)(y_i)], averaged over all posterior samples.
"""

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update('jax_enable_x64', True)
K_MAX = 8


def _model_diag(data, K, mu0, prior_chol_mu, prior_scale):
    """Diagonal-covariance GMM — no LKJCholesky, compatible with gradient-free SA."""
    w = numpyro.sample('w', dist.Dirichlet(jnp.ones(K) / K))
    with numpyro.plate('components', K):
        mu = numpyro.sample('mu', dist.MultivariateNormal(
            loc=mu0, scale_tril=prior_chol_mu,
        ))
        sigma = numpyro.sample('sigma', dist.HalfNormal(prior_scale).to_event(1))
    log_w = jnp.log(w + 1e-30)
    # Per-dim Normal: log p(y|k) = sum_j log N(y_j; mu_kj, sigma_kj)
    log_mix = jax.vmap(
        lambda y: jax.scipy.special.logsumexp(
            dist.Normal(mu, sigma).log_prob(y).sum(-1) + log_w
        )
    )(data)
    numpyro.factor('obs', log_mix.sum())


def fit_gmm_nuts_diag(
    data: np.ndarray,
    K: int = K_MAX,
    draws: int = 200,
    tune: int = 500,
    chains: int = 1,
    target_accept: float = 0.8,
    hyperprior_params: dict = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    Fit a diagonal-covariance GMM using NUTS (gradient-guided HMC).

    Uses _model_diag (327 params for K=8, d=20) instead of the full LKJCholesky
    model (1847 params), making NUTS tractable on n=50 data without degeneracy.

    Returns same format as fit_gmm: 'weights' (S,K), 'mu' (S,K,d), 'chol' (S,K,d,d).
    'chol' is diagonal: chol[s,k] = diag(sigma[s,k]).
    """
    n, d = data.shape
    seed = int(rng.integers(0, 2**31)) if rng is not None else 0

    if hyperprior_params is None:
        mu0   = data.mean(axis=0)
        kappa = 1.0
        nu    = float(d + 2)
        Psi   = (np.cov(data.T, ddof=1) if n > 1 else np.eye(d)) + 1e-4 * np.eye(d)
    else:
        mu0   = np.asarray(hyperprior_params['mu0'])
        kappa = float(hyperprior_params.get('kappa', 1.0))
        nu    = float(hyperprior_params.get('nu', d + 2))
        Psi   = np.asarray(hyperprior_params['Psi'])

    prior_cov_mu  = Psi / kappa
    prior_scale   = np.sqrt(np.diag(Psi / max(nu - d - 1, 1.0)))
    prior_chol_mu = np.linalg.cholesky(prior_cov_mu)

    jdata          = jnp.array(data,          dtype=jnp.float64)
    jmu0           = jnp.array(mu0,           dtype=jnp.float64)
    jprior_chol_mu = jnp.array(prior_chol_mu, dtype=jnp.float64)
    jprior_scale   = jnp.array(prior_scale,   dtype=jnp.float64)

    nuts = NUTS(_model_diag, target_accept_prob=target_accept, max_tree_depth=6)
    mcmc = MCMC(
        nuts,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method='sequential',
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jdata, K, jmu0, jprior_chol_mu, jprior_scale,
    )
    samples = mcmc.get_samples(group_by_chain=False)

    weights_s = np.array(samples['w'])      # (S, K)
    mu_s      = np.array(samples['mu'])     # (S, K, d)
    sigma_s   = np.array(samples['sigma'])  # (S, K, d)
    chol_s    = sigma_s[:, :, :, None] * np.eye(d)[None, None]  # (S, K, d, d)

    return {'weights': weights_s, 'mu': mu_s, 'chol': chol_s}


def _model_diag_weighted(data, obs_weights, K, mu0, prior_chol_mu, prior_scale):
    """Diagonal GMM with per-point weights: sum_j w_j log p(x_j | theta)."""
    w = numpyro.sample('w', dist.Dirichlet(jnp.ones(K) / K))
    with numpyro.plate('components', K):
        mu = numpyro.sample('mu', dist.MultivariateNormal(
            loc=mu0, scale_tril=prior_chol_mu,
        ))
        sigma = numpyro.sample('sigma', dist.HalfNormal(prior_scale).to_event(1))
    log_w = jnp.log(w + 1e-30)
    log_mix = jax.vmap(
        lambda y: jax.scipy.special.logsumexp(
            dist.Normal(mu, sigma).log_prob(y).sum(-1) + log_w
        )
    )(data)
    numpyro.factor('obs', (obs_weights * log_mix).sum())


def fit_gmm_nuts_diag_weighted(
    data: np.ndarray,
    obs_weights: np.ndarray,
    K: int = K_MAX,
    draws: int = 200,
    tune: int = 500,
    chains: int = 1,
    target_accept: float = 0.8,
    hyperprior_params: dict = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    NUTS on diagonal-covariance GMM with per-observation weights.

    Posterior: p(theta | {x_j, w_j}) ∝ p(theta) exp(sum_j w_j log p(x_j|theta)).
    Used for QNC coresets where the optimised weights w_j replace equal weighting.

    Parameters
    ----------
    data        : (n, d) coreset support points
    obs_weights : (n,) non-negative weights (should sum to ~N for proper scaling)
    """
    n, d = data.shape
    seed = int(rng.integers(0, 2**31)) if rng is not None else 0

    if hyperprior_params is None:
        mu0   = data.mean(axis=0)
        kappa = 1.0
        nu    = float(d + 2)
        Psi   = (np.cov(data.T, ddof=1) if n > 1 else np.eye(d)) + 1e-4 * np.eye(d)
    else:
        mu0   = np.asarray(hyperprior_params['mu0'])
        kappa = float(hyperprior_params.get('kappa', 1.0))
        nu    = float(hyperprior_params.get('nu', d + 2))
        Psi   = np.asarray(hyperprior_params['Psi'])

    prior_cov_mu  = Psi / kappa
    prior_scale   = np.sqrt(np.diag(Psi / max(nu - d - 1, 1.0)))
    prior_chol_mu = np.linalg.cholesky(prior_cov_mu)

    jdata          = jnp.array(data,          dtype=jnp.float64)
    jobs_weights   = jnp.array(obs_weights,   dtype=jnp.float64)
    jmu0           = jnp.array(mu0,           dtype=jnp.float64)
    jprior_chol_mu = jnp.array(prior_chol_mu, dtype=jnp.float64)
    jprior_scale   = jnp.array(prior_scale,   dtype=jnp.float64)

    nuts = NUTS(_model_diag_weighted, target_accept_prob=target_accept, max_tree_depth=6)
    mcmc = MCMC(
        nuts,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method='sequential',
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jdata, jobs_weights, K, jmu0, jprior_chol_mu, jprior_scale,
    )
    samples = mcmc.get_samples(group_by_chain=False)

    weights_s = np.array(samples['w'])
    mu_s      = np.array(samples['mu'])
    sigma_s   = np.array(samples['sigma'])
    chol_s    = sigma_s[:, :, :, None] * np.eye(d)[None, None]

    return {'weights': weights_s, 'mu': mu_s, 'chol': chol_s}


def fit_gmm_svi(
    data: np.ndarray,
    K: int = K_MAX,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-3,
    n_samples: int = 200,
    hyperprior_params: dict = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    Fit a K-component GMM via SVI (AutoNormal mean-field ELBO) with minibatch
    subsampling.  Scales to large N where NUTS would be prohibitively slow.

    Each minibatch of size B contributes a log-likelihood scaled by N/B so
    the ELBO gradient is an unbiased estimate of the full-data ELBO.

    Returns the same dict format as fit_gmm — 'weights' (S,K), 'mu' (S,K,d),
    'chol' (S,K,d,d) — where S=n_samples draws from the fitted variational
    distribution.  Drop-in replacement for cluster_posterior.
    """
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam

    n, d = data.shape
    seed = int(rng.integers(0, 2**31)) if rng is not None else 0
    np_rng = np.random.default_rng(seed)

    if hyperprior_params is None:
        mu0 = data.mean(axis=0)
        kappa = 1.0
        nu = float(d + 2)
        Psi = (np.cov(data.T, ddof=1) if n > 1 else np.eye(d)) + 1e-4 * np.eye(d)
    else:
        mu0 = np.asarray(hyperprior_params['mu0'])
        kappa = float(hyperprior_params.get('kappa', 1.0))
        nu = float(hyperprior_params.get('nu', d + 2))
        Psi = np.asarray(hyperprior_params['Psi'])

    prior_cov_mu = Psi / kappa
    prior_scale = np.sqrt(np.diag(Psi / max(nu - d - 1, 1.0)))
    prior_chol_mu = np.linalg.cholesky(prior_cov_mu)

    jdata = jnp.array(data, dtype=jnp.float64)
    jmu0 = jnp.array(mu0, dtype=jnp.float64)
    jprior_chol_mu = jnp.array(prior_chol_mu, dtype=jnp.float64)
    jprior_scale = jnp.array(prior_scale, dtype=jnp.float64)
    N_total = n

    def _svi_model(batch, K, mu0_, prior_chol_mu_, prior_scale_, N_total_):
        n_batch, d_ = batch.shape
        w = numpyro.sample('w', dist.Dirichlet(jnp.ones(K) / K))
        with numpyro.plate('components', K):
            mu = numpyro.sample('mu', dist.MultivariateNormal(
                loc=mu0_, scale_tril=prior_chol_mu_,
            ))
            L_corr = numpyro.sample('L_corr', dist.LKJCholesky(d_, concentration=2.0))
            sigma = numpyro.sample('sigma', dist.HalfNormal(prior_scale_).to_event(1))
        L_full = sigma[..., None] * L_corr
        log_w = jnp.log(w + 1e-30)
        comp_dist = dist.MultivariateNormal(loc=mu, scale_tril=L_full)
        log_mix = jax.vmap(
            lambda y: jax.scipy.special.logsumexp(comp_dist.log_prob(y) + log_w)
        )(batch)
        # Scale minibatch likelihood to approximate full-data ELBO
        numpyro.factor('obs', log_mix.sum() * (N_total_ / n_batch))

    guide = AutoNormal(_svi_model)
    svi = SVI(_svi_model, guide, Adam(step_size=lr), loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    svi_state = svi.init(
        init_key,
        jdata[:batch_size], K, jmu0, jprior_chol_mu, jprior_scale, N_total,
    )

    for _ in range(n_steps):
        idx = np_rng.choice(N_total, size=batch_size, replace=False)
        svi_state, _ = svi.update(
            svi_state,
            jdata[idx], K, jmu0, jprior_chol_mu, jprior_scale, N_total,
        )

    params = svi.get_params(svi_state)
    rng_key, sample_key = jax.random.split(rng_key)
    samples = guide.sample_posterior(sample_key, params, sample_shape=(n_samples,))

    weights_s = np.array(samples['w'])           # (S, K)
    mu_s      = np.array(samples['mu'])          # (S, K, d)
    L_corr_s  = np.array(samples['L_corr'])      # (S, K, d, d)
    sigma_s   = np.array(samples['sigma'])       # (S, K, d)
    chol_s    = sigma_s[..., None] * L_corr_s   # (S, K, d, d)

    return {'weights': weights_s, 'mu': mu_s, 'chol': chol_s}


def _log_mvn_chol(data: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Log-density of N(mu, L L^T) at each row of data."""
    d    = mu.shape[0]
    diff = data - mu
    z    = scipy.linalg.solve_triangular(L, diff.T, lower=True)
    maha = np.sum(z ** 2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-10)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


def posterior_mean_log_density(
    points: np.ndarray,
    posterior: dict,
    max_samples: int = 200,
) -> np.ndarray:
    """
    Mean posterior predictive log-density at each point.

    log p̄(y) ≈ log [ (1/S') Σ_s  Σ_k w_k^(s) N(y; μ_k^(s), Σ_k^(s)) ]

    Averages the mixture density over posterior draws, which is immune to
    label switching because the density is a sum over all components.

    Parameters
    ----------
    points      : (N, d) evaluation points
    posterior   : dict from fit_gmm — 'weights' (S,K), 'mu' (S,K,d), 'chol' (S,K,d,d)
    max_samples : max posterior draws to use (evenly spaced; speeds up evaluation)

    Returns
    -------
    (N,) log-density array.
    """
    N    = len(points)
    S, K = posterior['weights'].shape
    weights = posterior['weights']   # (S, K)
    mu      = posterior['mu']        # (S, K, d)
    chol    = posterior['chol']      # (S, K, d, d)

    if S > max_samples:
        idx     = np.round(np.linspace(0, S - 1, max_samples)).astype(int)
        weights = weights[idx]
        mu      = mu[idx]
        chol    = chol[idx]
        S       = max_samples

    dens = np.zeros(N)
    for s in range(S):
        log_comp = np.stack([
            np.log(weights[s, k] + 1e-300) + _log_mvn_chol(points, mu[s, k], chol[s, k])
            for k in range(K)
        ], axis=1)                                           # (N, K)
        shift    = log_comp.max(axis=1, keepdims=True)
        log_mix  = shift[:, 0] + np.log(np.exp(log_comp - shift).sum(axis=1))
        dens    += np.exp(log_mix)

    return np.log(dens / S + 1e-300)


def cluster_posterior(
    data: np.ndarray,
    posterior: dict,
    K_cl: int,
    max_samples: int = 200,
    n_tau: int = 100,
) -> np.ndarray:
    """
    Cluster data via mean posterior density + upper-level-set algorithm.

    Replaces classify() when component assignments are degenerate (label
    switching, redundant components).  The mean posterior density is smooth
    and label-switching-invariant; the τ-search finds the K_cl most prominent
    density modes.

    Parameters
    ----------
    data        : (N, d) points to cluster
    posterior   : dict from fit_gmm
    K_cl        : target number of clusters
    max_samples : posterior draws to average for density estimation
    n_tau       : τ grid resolution for the level-set search

    Returns
    -------
    (N,) integer label array.
    """
    from coreset.clustering import cluster_from_log_density
    log_dens = posterior_mean_log_density(data, posterior, max_samples=max_samples)
    return cluster_from_log_density(data, log_dens, K_cl, n_tau=n_tau)
