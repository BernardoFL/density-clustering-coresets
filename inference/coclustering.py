"""
Posterior co-clustering and partition-level summaries.

Given posterior draws of a K-component Gaussian mixture (the dict returned by
fit_gmm_nuts_diag / fit_gmm_svi: 'weights' (S,K), 'mu' (S,K,d), 'chol'
(S,K,d,d)) and M evaluation points:

  responsibilities      π_ik^(s)                         (S, M, K)
  coclustering          P_ij = (1/S) Σ_s Σ_k π_ik^(s) π_jk^(s)   (M, M)
                        Rao–Blackwellised, invariant to label switching
  sample_partitions     one allocation z^(s) ~ Cat(π^(s)) per draw
  voi_point_estimate    draws-based VoI-optimal partition (Wade & Ghahramani 2018):
                        the sampled partition minimising mean VoI to all draws
"""

import numpy as np

from coreset.distances import voi


def _is_diagonal(chol: np.ndarray) -> bool:
    d = chol.shape[-1]
    off = chol * (1.0 - np.eye(d))
    return not np.any(off)


def responsibilities(points: np.ndarray, posterior: dict) -> np.ndarray:
    """Component responsibilities π_ik^(s) for every draw s: shape (S, M, K)."""
    from inference.gmm_numpyro import _log_mvn_chol

    weights = posterior['weights']   # (S, K)
    mu      = posterior['mu']        # (S, K, d)
    chol    = posterior['chol']      # (S, K, d, d)
    S, K    = weights.shape
    M, d    = points.shape

    R = np.empty((S, M, K))
    diag = _is_diagonal(chol)
    for s in range(S):
        if diag:
            sig  = np.maximum(np.diagonal(chol[s], axis1=-2, axis2=-1), 1e-10)   # (K, d)
            z    = (points[:, None, :] - mu[s][None]) / sig[None]                 # (M, K, d)
            logp = -0.5 * np.sum(z ** 2, axis=-1) - np.sum(np.log(sig), axis=-1) \
                   - 0.5 * d * np.log(2.0 * np.pi)                                # (M, K)
        else:
            logp = np.stack([_log_mvn_chol(points, mu[s, k], chol[s, k])
                             for k in range(K)], axis=1)
        logp += np.log(weights[s] + 1e-300)
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        R[s] = r / r.sum(axis=1, keepdims=True)
    return R


def coclustering(R: np.ndarray) -> np.ndarray:
    """Rao–Blackwellised co-clustering matrix (M, M) from responsibilities (S, M, K)."""
    S, M, _ = R.shape
    P = np.zeros((M, M))
    for s in range(S):
        P += R[s] @ R[s].T
    return P / S


def sample_partitions(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One partition per draw, z_i^(s) ~ Cat(π_i^(s)): shape (S, M) integer labels."""
    S, M, K = R.shape
    cum = np.cumsum(R, axis=-1)
    u   = rng.uniform(size=(S, M, 1))
    return np.minimum((u > cum).sum(axis=-1), K - 1)


def voi_point_estimate(parts: np.ndarray) -> tuple:
    """
    Draws-based VoI-optimal partition: among the sampled partitions, the one
    minimising the mean VoI to all draws.  Returns (s_hat, expected_voi), where
    expected_voi = (1/S) Σ_s VoI(parts[s], s_hat).
    """
    S = len(parts)
    C = np.zeros((S, S))
    for a in range(S):
        for b in range(a + 1, S):
            C[a, b] = C[b, a] = voi(parts[a], parts[b])
    risk = C.mean(axis=1)
    best = int(np.argmin(risk))
    return parts[best].copy(), float(risk[best])


def coclustering_metrics(P: np.ndarray, P_ref: np.ndarray) -> dict:
    """‖P − P_ref‖_F / M and max_ij |P − P_ref|."""
    M = P.shape[0]
    D = np.abs(P - P_ref)
    return {'frob_over_M': float(np.sqrt(np.sum(D ** 2)) / M), 'max_abs': float(D.max())}


def summarise_posterior(points: np.ndarray, posterior: dict, rng: np.random.Generator) -> dict:
    """Co-clustering matrix, VoI-optimal point estimate and its posterior VoI spread."""
    R     = responsibilities(points, posterior)
    P     = coclustering(R)
    parts = sample_partitions(R, rng)
    s_hat, spread = voi_point_estimate(parts)
    return {'P': P, 's_hat': s_hat, 'voi_spread': spread}
