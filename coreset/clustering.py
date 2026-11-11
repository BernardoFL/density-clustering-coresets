import numpy as np
from scipy.spatial import cKDTree


def _assign_labels_from_density(
    traj: np.ndarray,
    log_dens: np.ndarray,
    tau: float,
    h: float,
    s_min: int = 1,
) -> np.ndarray:
    """
    Upper-level-set label assignment given pre-computed log densities and bandwidth.

    High-density points (log_dens >= tau) are grouped via a radius-2h graph.
    Small components are merged into the nearest large one; low-density points
    are assigned to the nearest large-component centroid.
    """
    L         = traj.shape[0]
    high_mask = log_dens >= tau
    high_idx  = np.where(high_mask)[0]
    labels    = np.full(L, -1, dtype=int)

    if len(high_idx) == 0:
        return np.zeros(L, dtype=int)

    high_points = traj[high_idx]
    tree        = cKDTree(high_points)
    parent      = list(range(len(high_idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in tree.query_pairs(2.0 * h):
        union(a, b)

    comp_map = {}
    comp_id  = 0
    for i in range(len(high_idx)):
        root = find(i)
        if root not in comp_map:
            comp_map[root] = comp_id
            comp_id += 1
        labels[high_idx[i]] = comp_map[root]

    comp_sizes  = np.bincount(labels[labels >= 0], minlength=comp_id)
    large_comps = np.where(comp_sizes >= s_min)[0]

    if len(large_comps) == 0:
        large_comps = np.arange(comp_id)

    large_centroids = np.array([traj[labels == c].mean(axis=0) for c in large_comps])

    for c in set(range(comp_id)) - set(large_comps.tolist()):
        for idx in np.where(labels == c)[0]:
            labels[idx] = large_comps[np.argmin(np.linalg.norm(large_centroids - traj[idx], axis=1))]

    for idx in np.where(~high_mask)[0]:
        labels[idx] = large_comps[np.argmin(np.linalg.norm(large_centroids - traj[idx], axis=1))]

    unique_labels = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_labels)}
    return np.array([remap[l] for l in labels], dtype=int)


class ClustModel:
    """
    A fitted Clust map that can label points other than the trajectory.

    Level-set Clust at every dimension, with dimension-adapted hyperparameters
    (see fit_clust).  label(x) = −1 if the KDE at x is below τ, else the
    cluster of the nearest core trajectory point within the graph radius
    (−1 if none), after the same minsize merge applied to the trajectory.
    Backend 'constant' is a degenerate trajectory: everything gets label 0.

    `labels` holds the trajectory's own labels; label(trajectory) reproduces it.
    """

    def __init__(self, backend, labels, mean=None, W=None, Z=None, h=None,
                 radius=None, tau=None, core_Z=None, core_labels=None):
        self.backend     = backend
        self.labels      = labels
        self.mean        = mean          # (d,)   centre
        self.W           = W             # (d, d) isotropic scaling, z = (x − mean) @ W
        self.Z           = Z             # (L, d) scaled trajectory (KDE centres)
        self.h           = h             # KDE bandwidth in scaled units
        self.radius      = radius        # graph radius in scaled units
        self.tau         = tau           # log-density threshold
        self.core_Z      = core_Z
        self.core_labels = core_labels
        self._tree       = cKDTree(core_Z) if core_Z is not None and len(core_Z) else None

    def whiten(self, points: np.ndarray) -> np.ndarray:
        return (points - self.mean) @ self.W

    def log_density(self, points: np.ndarray) -> np.ndarray:
        return _kde_logpdf(self.whiten(points), self.Z, self.h)

    def label(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(points)
        if points.ndim == 1:
            pts = pts.T
        if self.backend == 'constant':
            return np.zeros(len(pts), dtype=int)
        out   = np.full(len(pts), -1, dtype=int)
        z     = self.whiten(pts)
        above = np.where(_kde_logpdf(z, self.Z, self.h) >= self.tau)[0]
        if len(above) and self._tree is not None:
            dist, nn = self._tree.query(z[above], k=1)
            ok = dist <= self.radius
            out[above[ok]] = self.core_labels[nn[ok]]
        return out


def _isotropic_scaling(traj: np.ndarray):
    """
    Mean and map W = I/s (z = (x − mean) @ W), s = RMS coordinate std.

    Isotropic on purpose: whitening by the full trajectory covariance (scipy's
    gaussian_kde convention) shrinks exactly the between-cluster directions,
    which dominate that covariance, and destroys cluster separation in high d.
    """
    mean = traj.mean(axis=0)
    s    = np.sqrt(np.mean(np.var(traj, axis=0, ddof=1)))
    return mean, np.eye(traj.shape[1]) / max(s, 1e-12)


def _kde_logpdf(z: np.ndarray, Z: np.ndarray, h: float, chunk: int = 2048) -> np.ndarray:
    """
    Log-density at z of the isotropic Gaussian KDE with centres Z and bandwidth h,
    leaving out centres that coincide with the query point.

    The Pólya urn copies data points, so a trajectory point sits on its own
    kernel and on those of its copies.  Counting them inflates its density far
    above that of any point not in the trajectory (in high d the self-term
    dominates), and τ would be calibrated to a level no reference point can
    reach.  Leaving coincident centres out (normalising by the remaining
    count) puts trajectory points and reference points on the same footing.
    Reference points are never in the trajectory, so for them nothing is left out.
    """
    from scipy.special import logsumexp
    L, d  = Z.shape
    const = -d * np.log(h) - 0.5 * d * np.log(2.0 * np.pi)
    zz    = np.sum(Z ** 2, axis=1)
    out   = np.empty(len(z))
    for i in range(0, len(z), chunk):
        q    = z[i:i + chunk]
        sq   = np.maximum(np.sum(q ** 2, axis=1)[:, None] + zz[None] - 2.0 * q @ Z.T, 0.0)
        same = sq <= 1e-12 * max(1.0, float(np.max(zz)))
        expo = np.where(same, -np.inf, -0.5 * sq / h ** 2)
        kept = L - same.sum(axis=1)
        with np.errstate(divide='ignore'):
            out[i:i + chunk] = logsumexp(expo, axis=1) - np.log(np.maximum(kept, 1)) + const
        out[i:i + chunk][kept == 0] = -np.inf
    return out


def silverman_factor(L: int, d: int) -> float:
    """Multivariate Silverman bandwidth factor (whitened units)."""
    return (4.0 / (d + 2)) ** (1.0 / (d + 4)) * L ** (-1.0 / (d + 4))


def graph_radius(h: float, d: int) -> float:
    """Radius-graph connectivity scale: 2h·√d, the reach of a d-dim Gaussian kernel."""
    return 2.0 * h * np.sqrt(d)


def _components_at_tau(D: np.ndarray, log_dens: np.ndarray, tau: float, radius: float):
    """Connected components of {log_dens ≥ τ} under the radius graph; returns (K, labels)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    core = np.where(log_dens >= tau)[0]
    labels = np.full(len(log_dens), -1, dtype=int)
    if len(core) == 0:
        return 0, labels
    K, comp = connected_components(csr_matrix(D[np.ix_(core, core)] <= radius), directed=False)
    labels[core] = comp
    return K, labels


def _components_over_taus(D: np.ndarray, log_dens: np.ndarray, taus: np.ndarray,
                          radius: float) -> np.ndarray:
    """
    Number of connected components of {log_dens ≥ τ} under the radius graph,
    for every τ in `taus`, in one incremental union-find sweep.

    Points are activated in decreasing density and an edge joins as soon as
    both endpoints are active (its activation level is the smaller endpoint
    density).  Equal to calling _components_at_tau once per τ, at the cost of
    a single pass.
    """
    L = len(log_dens)
    ii, jj = np.nonzero(np.triu(D <= radius, k=1))
    e_lvl  = np.minimum(log_dens[ii], log_dens[jj])
    e_ord  = np.argsort(-e_lvl, kind='stable')
    ii, jj, e_lvl = ii[e_ord].tolist(), jj[e_ord].tolist(), e_lvl[e_ord].tolist()
    p_lvl  = np.sort(log_dens)[::-1].tolist()

    parent = list(range(L))
    K, pp, ep, M = 0, 0, 0, len(ii)
    out = np.zeros(len(taus), dtype=int)
    for t_idx in np.argsort(-np.asarray(taus), kind='stable'):
        tau = taus[t_idx]
        while pp < L and p_lvl[pp] >= tau:
            K += 1
            pp += 1
        while ep < M and e_lvl[ep] >= tau:
            a, b = ii[ep], jj[ep]
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[a] = b
                K -= 1
            ep += 1
        out[t_idx] = K
    return out


def _adaptive_level_set_search(Z: np.ndarray, K_cl: int, n_tau: int = 50, max_steps: int = 20,
                               min_size: int = 0):
    """
    τ-search with bidirectional bandwidth adaptation, in scaled coordinates.

    Starts from the multivariate Silverman factor; for each bandwidth h the
    KDE log-density and the radius graph (radius 2h√d) are formed and τ is
    swept over the 5–95% quantiles of the log-density.  If no τ gives K_cl
    components, h shrinks (too few) or grows (too many) and the search repeats.
    Once K_cl is reached, h keeps shrinking while K_cl is still attainable, and
    the attempt with the largest core set (lowest such τ) is kept, so that few
    points fall in the sub-threshold cell.
    With min_size > 1 only components with at least min_size points count toward K_cl, so
    isolated specks cannot meet the target and tau rises until real regions separate.
    Returns (log_dens, tau, h, radius) of the best attempt.
    """
    from scipy.spatial.distance import squareform, pdist
    L, d = Z.shape
    D    = squareform(pdist(Z))
    h    = silverman_factor(L, d)
    best, hit = None, False
    for _ in range(max_steps + 1):
        log_dens = _kde_logpdf(Z, Z, h)
        radius   = graph_radius(h, d)
        fin      = log_dens[np.isfinite(log_dens)]
        taus     = np.linspace(np.percentile(fin, 5), np.percentile(fin, 95), n_tau)
        if min_size > 1:
            Ks = np.array([int(np.sum(np.bincount(lab[lab >= 0]) >= min_size)) if (lab >= 0).any() else 0
                           for lab in (_components_at_tau(D, log_dens, t, radius)[1] for t in taus)])
        else:
            Ks = _components_over_taus(D, log_dens, taus, radius)
        i        = int(np.argmin(np.abs(Ks - K_cl)))       # lowest τ attaining the best K
        diff     = abs(int(Ks[i]) - K_cl)
        core     = float(np.mean(log_dens >= taus[i]))
        if best is None or (diff, -core) < (best[0], -best[1]):
            best = (diff, core, log_dens, float(taus[i]), h, radius)
        if diff == 0:
            hit = True
        elif hit:
            break                                          # shrinking past the K_cl regime
        h *= 0.75 if (hit or Ks.max() < K_cl) else 1.3
    return best[2], best[3], best[4], best[5]


def fit_clust(
    trajectory: np.ndarray,
    K_cl: int,
    minsize: int = 1,
    n_tau: int = 50,
    max_steps: int = 20,
    min_share: float = 0.0,
) -> ClustModel:
    """
    Fit the Clust map of Algorithm 1 to a trajectory and return a ClustModel.

    Upper-level-set clustering of a Gaussian KDE at every dimension, with the
    hyperparameters adapted to d:
      * coordinates centred and scaled isotropically by the RMS coordinate
        std (not whitened: see _isotropic_scaling);
      * h starts at the multivariate Silverman factor (4/(d+2))^{1/(d+4)} L^{-1/(d+4)};
      * the radius graph connects core points within 2h√d;
      * τ and h are searched so the level set has K_cl components.
    Sub-threshold trajectory points keep the shared label −1, so the fitted
    map labels reference points consistently.
    min_share > 0: only components holding at least that share of the trajectory count toward
    K_cl, and smaller components are treated as sub-threshold (−1).
    """
    traj = np.atleast_2d(trajectory)
    if trajectory.ndim == 1:
        traj = traj.T
    L, d = traj.shape
    if np.mean(np.std(traj, axis=0, ddof=1)) < 1e-10:
        return ClustModel('constant', np.zeros(L, dtype=int))

    mean, W = _isotropic_scaling(traj)
    Z = (traj - mean) @ W
    m_min = int(np.ceil(min_share * L)) if min_share > 0 else 0
    log_dens, tau, h, radius = _adaptive_level_set_search(Z, K_cl, n_tau, max_steps, min_size=m_min)
    labels = _core_components(Z, log_dens, tau, radius, minsize)
    if m_min > 1 and (labels >= 0).any():
        cnt = np.bincount(labels[labels >= 0])
        labels = np.where((labels >= 0) & (cnt[np.maximum(labels, 0)] >= m_min), labels, -1)

    # Canonicalize core clusters by mean log-density (highest = label 0); −1 stays.
    ks = np.unique(labels[labels >= 0])
    mean_dens = np.array([log_dens[labels == k].mean() for k in ks])
    remap = {int(ks[i]): r for r, i in enumerate(np.argsort(-mean_dens))}
    labels = np.array([remap.get(int(l), -1) for l in labels], dtype=int)

    core = labels >= 0
    return ClustModel(
        'levelset', labels, mean=mean, W=W, Z=Z, h=h, radius=radius, tau=tau,
        core_Z=Z[core], core_labels=labels[core],
    )


def _core_components(
    traj: np.ndarray,
    log_dens: np.ndarray,
    tau: float,
    radius: float,
    minsize: int = 1,
) -> np.ndarray:
    """
    Connected components of the core points (log_dens ≥ τ) under the radius
    graph, with components smaller than `minsize` merged into the nearest large
    component (by centroid).  Sub-threshold points get −1.
    """
    L         = traj.shape[0]
    labels    = np.full(L, -1, dtype=int)
    high_idx  = np.where(log_dens >= tau)[0]
    if len(high_idx) == 0:
        return labels

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pairs = cKDTree(traj[high_idx]).query_pairs(radius, output_type='ndarray')
    m     = len(high_idx)
    adj   = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    _, comp = connected_components(adj, directed=False)
    labels[high_idx] = comp

    sizes = np.bincount(comp)
    large = np.where(sizes >= minsize)[0]
    if len(large) == 0:
        large = np.arange(len(sizes))
    if len(large) < len(sizes):
        centroids = np.array([traj[labels == c].mean(axis=0) for c in large])
        for c in np.setdiff1d(np.arange(len(sizes)), large):
            members = np.where(labels == c)[0]
            target  = large[np.argmin(np.linalg.norm(
                centroids - traj[members].mean(axis=0), axis=1))]
            labels[members] = target
    return labels


def cluster_from_log_density(
    points: np.ndarray,
    log_dens: np.ndarray,
    K_cl: int,
    h: float = None,
    n_tau: int = 100,
    max_shrink: int = 20,
) -> np.ndarray:
    """
    Upper-level-set τ-search clustering on pre-computed log-densities.

    Same τ-search and label assignment as cluster_trajectory, but takes a
    pre-computed log-density instead of estimating one with KDE.

    When h is not supplied the connectivity radius is initialised from a
    k-means run rather than Silverman's rule.  Silverman is calibrated for
    KDE smoothing, not graph connectivity: at small n (e.g. n=100 coreset
    points) it produces a radius so wide that all points collapse into one
    component before the τ-search can separate them.  The k-means estimate
    uses half the median nearest-neighbour distance between the K_cl cluster
    centres, which sits between the within- and between-cluster scales.

    After finding the initial h an adaptive shrinkage loop (identical in
    spirit to cluster_trajectory) adjusts h until the τ-search finds
    exactly K_cl components, or until max_shrink attempts are exhausted.

    Parameters
    ----------
    points     : (L, d) or (L,) points at which log_dens was evaluated
    log_dens   : (L,) log-density values at those points
    K_cl       : target number of upper-level-set clusters
    h          : graph connectivity radius; if None, estimated via k-means
    n_tau      : number of τ levels to probe
    max_shrink : maximum bandwidth adjustment steps

    Returns
    -------
    (L,) integer label array, compacted to 0 … K-1, sorted by mean log-density.
    """
    from sklearn.cluster import KMeans

    pts = np.atleast_2d(points)
    if points.ndim == 1:
        pts = pts.T
    L = pts.shape[0]
    log_dens = np.asarray(log_dens, dtype=float)

    # ── Initial bandwidth: k-means NN distance between cluster centres ────────
    # Half the median nearest-neighbour distance between K_cl centres sits
    # between the within-cluster and between-cluster scales, giving the τ-search
    # room to separate components even when n is small.
    if h is None:
        K_init = min(K_cl, L)
        km = KMeans(n_clusters=K_init, n_init=10, random_state=0).fit(pts)
        c = km.cluster_centers_                                   # (K_init, d)
        c_sq = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(c_sq, np.inf)
        h = float(np.sqrt(c_sq.min(axis=1)).mean()) / 2.0
        h = max(h, 1e-10)

    # τ grid is fixed (depends only on log_dens, not h)
    tau_lo   = np.percentile(log_dens, 5)
    tau_hi   = np.percentile(log_dens, 95)
    tau_grid = np.linspace(tau_lo, tau_hi, n_tau)

    pt_order = np.argsort(-log_dens)
    pt_dens  = log_dens[pt_order].tolist()

    best_tau  = float(tau_grid[0])
    best_diff = float('inf')
    best_h    = h

    for _ in range(max_shrink + 1):
        pairs_raw = cKDTree(pts).query_pairs(2.0 * h)
        if pairs_raw:
            pairs_np = np.array(sorted(pairs_raw), dtype=np.intp)
            pair_min = np.minimum(log_dens[pairs_np[:, 0]], log_dens[pairs_np[:, 1]])
            pm_order = np.argsort(-pair_min)
            pa_list  = pairs_np[pm_order, 0].tolist()
            pb_list  = pairs_np[pm_order, 1].tolist()
            pm_list  = pair_min[pm_order].tolist()
        else:
            pa_list = pb_list = pm_list = []
        M = len(pa_list)

        parent   = list(range(L))
        pt_ptr   = 0
        pair_ptr = 0
        K_cur    = 0
        K_at_tau = [0] * n_tau

        for i_rev, tau in enumerate(reversed(tau_grid.tolist())):
            while pt_ptr < L and pt_dens[pt_ptr] >= tau:
                K_cur += 1
                pt_ptr += 1
            while pair_ptr < M and pm_list[pair_ptr] >= tau:
                a, b = pa_list[pair_ptr], pb_list[pair_ptr]
                x = a
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                ra = x
                x = b
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                rb = x
                if ra != rb:
                    parent[ra] = rb
                    K_cur -= 1
                pair_ptr += 1
            K_at_tau[n_tau - 1 - i_rev] = K_cur

        iter_best_tau  = float(tau_grid[0])
        iter_best_diff = abs(K_at_tau[0] - K_cl)
        iter_best_K    = K_at_tau[0]
        for i in range(1, n_tau):
            K    = K_at_tau[i]
            diff = abs(K - K_cl)
            if diff < iter_best_diff:
                iter_best_diff = diff
                iter_best_tau  = float(tau_grid[i])
                iter_best_K    = K
            if iter_best_diff == 0:
                break

        if iter_best_diff < best_diff:
            best_diff = iter_best_diff
            best_tau  = iter_best_tau
            best_h    = h

        if best_diff == 0:
            break

        # Too few components → h too large (clusters merge) → shrink.
        # Too many components → h too small (clusters split) → expand.
        h = h * 0.65 if iter_best_K < K_cl else h * 1.4

    labels = _assign_labels_from_density(pts, log_dens, best_tau, best_h)

    K_found   = int(labels.max()) + 1
    mean_dens = np.array([
        log_dens[labels == k].mean() if (labels == k).any() else -np.inf
        for k in range(K_found)
    ])
    order = np.argsort(-mean_dens)
    remap = np.empty(K_found, dtype=int)
    remap[order] = np.arange(K_found)
    return remap[labels]
