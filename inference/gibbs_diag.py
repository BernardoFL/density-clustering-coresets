"""
Overfitted Bayesian mixture with diagonal covariances, fitted by blocked Gibbs.

Model (K_max components, conjugate Normal–inverse-gamma per coordinate):
    π ~ Dirichlet(α, …, α)                      α small (sparse: Rousseau & Mengersen 2011)
    σ²_kj ~ InvGamma(a0, b0_j)
    μ_kj | σ²_kj ~ N(m0_j, σ²_kj / κ0)
    z_i | π ~ Cat(π),   x_ij | z_i = k ~ N(μ_kj, σ²_kj)
Unused components empty out, so the number of occupied clusters is inferred.

Hyperparameters are set from the data being fitted: m0 = mean, b0_j = (a0 − 1)·s_w·var_j
(prior mean of σ²_kj is a fraction s_w of the marginal variance).
"""

import time

import numpy as np


def default_prior(X, alpha=0.01, kappa0=0.01, a0=3.0, s_w=0.1):
    var = X.var(axis=0) + 1e-8
    return {'alpha': alpha, 'kappa0': kappa0, 'a0': a0, 'm0': X.mean(axis=0), 'b0': (a0 - 1.0) * s_w * var}


def _loglik_matrix(X, w, mu, s2):
    """log π_k + log N(x_i | μ_k, diag σ²_k): (n, K)."""
    ls = np.log(s2)
    quad = ((X ** 2) @ (1.0 / s2).T) - 2.0 * X @ (mu / s2).T + np.sum(mu ** 2 / s2, axis=1)[None]
    return np.log(w + 1e-300)[None] - 0.5 * (quad + ls.sum(1)[None] + X.shape[1] * np.log(2 * np.pi))


def gibbs_diag_mixture(X, K_max=30, n_iter=2000, burn=1000, thin=5, prior=None, seed=0, init='kmeans',
                       weights=None):
    """
    weights (n,), optional: likelihood weights (weighted coreset posterior); None = unit weights.
    Returns dict with draws of z (S, n), weights (S, K), mu (S, K, d), s2 (S, K, d),
    the full log-likelihood trace (n_iter,) and the sampler wall time.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n, d = X.shape
    if weights is not None:
        weights = np.asarray(weights, float)
    pr = prior or default_prior(X)
    a, k0, a0, m0, b0 = pr['alpha'], pr['kappa0'], pr['a0'], pr['m0'], pr['b0']
    K = K_max
    if init == 'kmeans' and n > K:
        from sklearn.cluster import KMeans
        z = KMeans(n_clusters=K, n_init=1, random_state=int(rng.integers(2 ** 31))).fit_predict(X)
    else:
        z = rng.integers(K, size=n)
    out = {k: [] for k in ('z', 'weights', 'mu', 's2')}
    trace = np.empty(n_iter)
    t0 = time.perf_counter()
    for it in range(n_iter):
        # parameters | z
        if weights is None:
            cnt = np.bincount(z, minlength=K).astype(float)
            sx = np.zeros((K, d)); np.add.at(sx, z, X)
            sxx = np.zeros((K, d)); np.add.at(sxx, z, X ** 2)
        else:
            cnt = np.bincount(z, weights=weights, minlength=K)
            sx = np.zeros((K, d)); np.add.at(sx, z, weights[:, None] * X)
            sxx = np.zeros((K, d)); np.add.at(sxx, z, weights[:, None] * X ** 2)
        xbar = np.where(cnt[:, None] > 0, sx / np.maximum(cnt[:, None], 1), 0.0)
        ss = np.maximum(sxx - cnt[:, None] * xbar ** 2, 0.0)
        kn = k0 + cnt[:, None]
        mn = (k0 * m0[None] + sx) / kn
        an = a0 + 0.5 * cnt[:, None]
        bn = b0[None] + 0.5 * ss + 0.5 * (k0 * cnt[:, None] / kn) * (xbar - m0[None]) ** 2
        s2 = bn / rng.gamma(an, 1.0)
        mu = mn + np.sqrt(s2 / kn) * rng.standard_normal((K, d))
        w = rng.dirichlet(a + cnt)
        # z | parameters
        lp = _loglik_matrix(X, w, mu, s2)
        m = lp.max(1, keepdims=True)
        p = np.exp(lp - m)
        trace[it] = float(np.sum(m[:, 0] + np.log(p.sum(1))))
        p /= p.sum(1, keepdims=True)
        u = rng.random(n)[:, None]
        z = np.minimum((np.cumsum(p, axis=1) < u).sum(1), K - 1)
        if it >= burn and (it - burn) % thin == 0:
            out['z'].append(z.copy()); out['weights'].append(w); out['mu'].append(mu); out['s2'].append(s2)
    res = {k: np.array(v) for k, v in out.items()}
    res.update({'loglik_trace': trace, 'time': time.perf_counter() - t0, 'burn': burn, 'thin': thin, 'prior': pr})
    return res


def ess(x):
    """Effective sample size of a 1-D chain (Geyer initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.var(x) * n)
    s, k = 0.0, 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        s += pair
        k += 2
    return float(n / (1 + 2 * s))


def point_partition(z_draws, max_draws=100):
    """Partition of the fitted points minimising the posterior expected VoI among the sampled ones."""
    from inference.coclustering import voi_point_estimate
    z = np.asarray(z_draws)
    z = z[np.unique(np.linspace(0, len(z) - 1, min(max_draws, len(z))).astype(int))]
    return voi_point_estimate(z)[0]


def partition_params(X_fit, z_hat, prior):
    """Posterior-mean parameters of each cluster of z_hat (conjugate NIG given the partition); weights ∝ sizes."""
    k0, a0, m0, b0 = prior['kappa0'], prior['a0'], prior['m0'], prior['b0']
    mu, s2, w = [], [], []
    for k in np.unique(z_hat):
        Y = X_fit[z_hat == k]
        c = len(Y)
        xbar = Y.mean(0)
        ss = ((Y - xbar) ** 2).sum(0)
        kn = k0 + c
        mu.append((k0 * m0 + c * xbar) / kn)
        bn = b0 + 0.5 * ss + 0.5 * (k0 * c / kn) * (xbar - m0) ** 2
        s2.append(bn / (a0 + 0.5 * c - 1.0))
        w.append(c)
    return np.array(w, float) / np.sum(w), np.array(mu), np.array(s2)


def _climb(x, w, mu, s2, max_iter=1000):
    for _ in range(max_iter):
        lp = _loglik_matrix(x[None], w, mu, s2)[0]
        r = np.exp(lp - lp.max()); r /= r.sum()
        x_new = (r[:, None] * mu / s2).sum(0) / (r[:, None] / s2).sum(0)
        if np.max(np.abs(x_new - x)) < 1e-9:
            return x_new
        x = x_new
    return x


def _log_mix(P, w, mu, s2):
    lp = _loglik_matrix(P, w, mu, s2)
    m = lp.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(lp - m).sum(1)))


def merge_shallow_modes(modes, w, mu, s2, cutoff=0.2, n_grid=50):
    """
    Ridgeline-ratio merging (Hennig 2010, as in fpc::mergenormals): two modes are one cluster unless
    the density along the segment joining them falls below cutoff × the lower of the two peaks.
    The straight segment approximates the ridgeline.  Returns a map mode -> merged cluster.
    """
    M = len(modes)
    parent = list(range(M))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    peaks = _log_mix(modes, w, mu, s2)
    t = np.linspace(0, 1, n_grid)[:, None]
    for a in range(M):
        for b in range(a + 1, M):
            seg = modes[a] + t * (modes[b] - modes[a])
            ratio = np.exp(_log_mix(seg, w, mu, s2).min() - min(peaks[a], peaks[b]))
            if ratio > cutoff:
                parent[find(a)] = find(b)
    roots = [find(i) for i in range(M)]
    _, out = np.unique(roots, return_inverse=True)
    return out


def modal_map(w, mu, s2, min_weight=0.01, tol=1e-2, ridge_cutoff=0.2):
    """
    Modal clustering of a diagonal Gaussian mixture (modal EM; Li, Ray & Lindsay 2007): a cluster is
    a mode of the fitted density, not a component.  From each component mean, iterate
    x <- (Σ_k r_k(x) μ_k / σ²_k) / (Σ_k r_k(x) / σ²_k) to a stationary point; components reaching
    the same mode (within tol × RMS component sd) form one cluster.  Components with weight
    < min_weight are negligible: they are left out of the density being climbed (so a near-empty,
    very sharp component cannot create a mode of its own) and are attached to the mode their
    mean climbs to.  Returns (component -> cluster map, modes).
    """
    big = w >= min_weight
    if not big.any():
        big = w == w.max()
    wb = w[big] / w[big].sum()
    mub, s2b = mu[big], s2[big]
    ends = np.array([_climb(mu[k].copy(), wb, mub, s2b) for k in range(len(w))])
    scale = tol * np.sqrt(np.mean(s2b))
    reps, cmap = [], np.empty(len(w), dtype=int)
    for k in np.argsort(~big, kind='stable'):          # modes of big components first
        for j, r in enumerate(reps):
            if np.max(np.abs(ends[k] - r)) < scale:
                cmap[k] = j
                break
        else:
            if big[k]:
                reps.append(ends[k]); cmap[k] = len(reps) - 1
            else:                                       # no matching mode: nearest one
                cmap[k] = int(np.argmin([np.sum((ends[k] - r) ** 2) for r in reps]))
    reps = np.array(reps)
    if ridge_cutoff is not None and len(reps) > 1:
        mm = merge_shallow_modes(reps, wb, mub, s2b, ridge_cutoff)
        cmap = mm[cmap]
        reps = np.array([reps[mm == j].mean(0) for j in range(mm.max() + 1)])
    return cmap, reps


def project(X_new, X_fit, z_hat, prior, modal=True, return_params=False):
    """
    Assign X_new with the point partition's posterior-mean mixture: each point goes to its most
    likely component; with modal=True (default) components are then merged by the modes of the
    fitted density, so a cluster is a mode, not a component.
    """
    w, mu, s2 = partition_params(X_fit, z_hat, prior)
    out = np.empty(len(X_new), dtype=int)
    for i in range(0, len(X_new), 50000):
        out[i:i + 50000] = _loglik_matrix(X_new[i:i + 50000], w, mu, s2).argmax(1)
    cmap = None
    if modal:
        cmap, _ = modal_map(w, mu, s2)
        out = cmap[out]
    if return_params:
        return out, {'w': w, 'mu': mu, 's2': s2, 'component_to_cluster': cmap}
    return out


def coclustering_rb(X_eval, draws, max_draws=100, modal=True, min_weight=1e-3):
    """
    Rao–Blackwellised posterior similarity of X_eval: mean over draws of R Rᵀ, with R the
    responsibilities of each draw's clusters (components merged by modes when modal=True;
    components with weight < min_weight dropped).  Also returns the number of clusters per draw.
    """
    S = len(draws['weights'])
    idx = np.unique(np.linspace(0, S - 1, min(max_draws, S)).astype(int))
    P = np.zeros((len(X_eval), len(X_eval)))
    n_clusters = []
    for s in idx:
        w, mu, s2 = draws['weights'][s], draws['mu'][s], draws['s2'][s]
        keep = w >= min_weight
        w, mu, s2 = w[keep] / w[keep].sum(), mu[keep], s2[keep]
        lp = _loglik_matrix(X_eval, w, mu, s2)
        lp -= lp.max(1, keepdims=True)
        R = np.exp(lp); R /= R.sum(1, keepdims=True)
        if modal:
            cmap, modes = modal_map(w, mu, s2)
            M = np.zeros((len(w), len(modes))); M[np.arange(len(w)), cmap] = 1.0
            R = R @ M
        n_clusters.append(R.shape[1])
        P += R @ R.T
    return P / len(idx), n_clusters
