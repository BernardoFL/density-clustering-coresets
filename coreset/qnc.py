"""
Quasi-Newton Coreset (QNC) — Campbell & Beronov (2021).

Algorithm
---------
1. Select n data points uniformly at random (with replacement; deduplicate).
2. Compute centered log-likelihood features phi[i,s] = log p(x_i|theta_s) - mean
   at posterior samples theta_s drawn from an initial inference run on the subsample.
3. Optimize non-negative weights w via quasi-Newton to minimise
       || sum_i phi_all[i]  -  w @ phi_sub ||^2
   initialised at the count-based uniform weights (w_j = N * count_j / n).

This is a single-shot version (projector not updated iteratively), which avoids
running NUTS at every Newton step.  In practice the feature map from the initial
NUTS run is rich enough for the GMM setting.
"""

import numpy as np


# ── Feature computation ───────────────────────────────────────────────────────

def _gmm_log_lik_features(data, posterior, proj_dim=None, rng=None):
    """
    Centered log-likelihood feature matrix  phi  of shape (N, S).

    phi[i, s] = log p(x_i | theta_s) − mean_j  log p(x_j | theta_s)

    Parameters
    ----------
    data      : (N, d)
    posterior : dict from fit_gmm_nuts_diag — 'weights'(S,K), 'mu'(S,K,d), 'chol'(S,K,d,d)
    proj_dim  : max number of posterior samples to use (None = all)
    rng       : numpy Generator for subsampling posterior
    """
    from inference.gmm_numpyro import _log_mvn_chol

    N = len(data)
    w_samp = posterior['weights']   # (S, K)
    mu_samp = posterior['mu']       # (S, K, d)
    ch_samp = posterior['chol']     # (S, K, d, d)
    S, K = w_samp.shape

    if proj_dim is not None and S > proj_dim:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(S, size=proj_dim, replace=False)
        w_samp, mu_samp, ch_samp = w_samp[idx], mu_samp[idx], ch_samp[idx]
        S = proj_dim

    phi = np.empty((N, S))
    for s in range(S):
        log_w_s = np.log(w_samp[s] + 1e-30)           # (K,)
        log_comp = np.stack([
            log_w_s[k] + _log_mvn_chol(data, mu_samp[s, k], ch_samp[s, k])
            for k in range(K)
        ], axis=1)                                     # (N, K)
        shift = log_comp.max(axis=1, keepdims=True)
        phi[:, s] = shift[:, 0] + np.log(np.exp(log_comp - shift).sum(axis=1))

    phi -= phi.mean(axis=1, keepdims=True)             # centre per data point across theta samples
    return phi                                         # (N, S)


# ── Public API ────────────────────────────────────────────────────────────────

def build_qnc_coreset(
    data,
    n,
    posterior_init,
    proj_dim=None,
    tau=0.01,
    rng=None,
):
    """
    Build a Quasi-Newton Coreset of size n.

    Selects exactly n distinct data points without replacement, then solves the
    unconstrained least-squares problem

        min_w  || phi_all.sum(0) - w @ phi_sub ||^2

    via a single regularised normal-equation solve.  All n points keep a weight;
    no sparsity is introduced beyond the committed subsample.

    Parameters
    ----------
    data          : (N, d) full dataset
    n             : coreset size — exactly n points selected
    posterior_init: dict from fit_gmm_nuts_diag (used to build log-lik features)
    proj_dim      : posterior samples to use for projection (None = all)
    tau           : ridge regularisation on the Gram matrix
    rng           : numpy Generator
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(data)

    # 1. Exactly n unique points, no replacement
    idcs = rng.choice(N, size=n, replace=False)
    pts  = data[idcs]

    # 2. Log-likelihood feature matrix
    phi_all = _gmm_log_lik_features(data, posterior_init, proj_dim=proj_dim, rng=rng)
    phi_sub = phi_all[idcs]                    # (n, S)

    # 3. Unconstrained least-squares: w* = (phi_sub @ phi_sub.T + τI)^{-1} phi_sub @ target
    #    All n weights are non-zero; no projection onto the non-negative orthant.
    target = phi_all.sum(0)                    # (S,) — full-data sum of features
    S      = phi_sub.shape[1]
    H      = (phi_sub @ phi_sub.T) / S        # (n, n)
    np.fill_diagonal(H, H.diagonal() + tau)
    g      = (phi_sub @ target) / S           # (n,)
    wts = np.linalg.solve(H, g)               # (n,) unconstrained optimal weights

    # Floor near-zero / negative weights to a small positive value so all n
    # points remain active (no sparsity beyond the committed subsample).
    floor = 1e-3 * (float(N) / n)
    wts   = np.maximum(wts, floor)

    return pts, wts, idcs
