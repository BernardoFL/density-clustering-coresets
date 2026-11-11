"""
Soft level-set VoI on a reference set: the differentiable surrogate for the
transform phase of Algorithm 1.

Both partitions label the same held-out reference points, so a real confusion
matrix exists.  Only the coreset labels are softened:
  (i)  a sigmoid gate on log f̂(x) − log τ   (above vs below the level set);
  (ii) a tempered softmax over per-component KDE contributions (which component).
Gradients flow through the KDE, which is smooth in (A, b).  Which trajectory
points form which component (the topology) is recomputed from the current
transform but treated as a constant.

The KDE hyperparameters (isotropic scaling, h, τ, graph radius, minsize) of
iteration t come from fit_clust on the full-data trajectory of that iteration
and are shared, unchanged, by its coreset trajectory.  Everything is computed
in that scaled space; τ is a log-density on the same scale as _kde_logpdf.

Sub-threshold reference points (label −1) follow the VoI_ref convention of
coreset.distances.voi_ref: each one is its own singleton ('singletons').  The
alternative 'cell' puts them all in one shared cell.
"""

import numpy as np

from coreset.clustering import _kde_logpdf, _core_components, fit_clust
from coreset.distances import voi, voi_ref


# ── Hard pieces (numpy) ──────────────────────────────────────────────────────

def hp_from_model(model, minsize: int = 1, min_share: float = 0.0) -> dict:
    """KDE hyperparameters of a fitted full-data Clust, to share with the coreset."""
    assert model.backend == 'levelset', 'degenerate (constant) full-data trajectory'
    return {'mean': model.mean, 's': float(1.0 / model.W[0, 0]), 'h': float(model.h),
            'tau': float(model.tau), 'radius': float(model.radius), 'minsize': int(minsize),
            'min_share': float(min_share)}


def scale(hp: dict, X: np.ndarray) -> np.ndarray:
    return (X - hp['mean']) / hp['s']


def _relabel(labels: np.ndarray):
    """Map component ids to 0..m−1 (−1 stays); returns (labels, m)."""
    ks  = np.unique(labels[labels >= 0])
    out = np.full(len(labels), -1, dtype=int)
    for r, k in enumerate(ks):
        out[labels == k] = r
    return out, len(ks)


def clust_shared(hp: dict, traj: np.ndarray):
    """Clust(traj) with fixed hyperparameters hp; returns (labels in {−1, 0..m−1}, m)."""
    Z  = scale(hp, traj)
    ld = _kde_logpdf(Z, Z, hp['h'])
    lab = _core_components(Z, ld, hp['tau'], hp['radius'], hp['minsize'])
    m_min = int(np.ceil(hp.get('min_share', 0.0) * len(Z)))
    if m_min > 1 and (lab >= 0).any():                 # components below the minimum share are sub-threshold
        cnt = np.bincount(lab[lab >= 0])
        lab = np.where((lab >= 0) & (cnt[np.maximum(lab, 0)] >= m_min), lab, -1)
    return _relabel(lab)


def log_kde_components(z: np.ndarray, Z: np.ndarray, labels: np.ndarray, m: int,
                       h: float, chunk: int = 2048):
    """
    Scaled coordinates.  Returns lf (B,) = log f̂(z) (same formula as _kde_logpdf,
    coincident centres left out) and lfk (B, m) = log of component k's share of the
    mixture, (1/L) Σ_{l∈C_k} K_h(z − z_l).  Sub-threshold points count in lf only.
    """
    from scipy.special import logsumexp
    L, d  = Z.shape
    const = -d * np.log(h) - 0.5 * d * np.log(2.0 * np.pi)
    zz    = np.sum(Z ** 2, axis=1)
    lf    = np.empty(len(z))
    lfk   = np.empty((len(z), m))
    for i in range(0, len(z), chunk):
        q    = z[i:i + chunk]
        sq   = np.maximum(np.sum(q ** 2, axis=1)[:, None] + zz[None] - 2.0 * q @ Z.T, 0.0)
        same = sq <= 1e-12 * max(1.0, float(np.max(zz)))
        expo = np.where(same, -np.inf, -0.5 * sq / h ** 2)
        norm = np.log(np.maximum(L - same.sum(axis=1), 1)) - const
        with np.errstate(divide='ignore'):
            lf[i:i + chunk] = logsumexp(expo, axis=1) - norm
            for k in range(m):
                lfk[i:i + chunk, k] = logsumexp(expo[:, labels == k], axis=1) - norm
    return lf, lfk


def hard_ref_labels(hp: dict, X: np.ndarray, traj: np.ndarray, labels: np.ndarray, m: int,
                    return_lf: bool = False):
    """
    Hard labels of the points X under the Clust of `traj` (labels, m):
    argmax_k of the per-component KDE share, −1 where log f̂(x) < τ.
    The soft labels converge to these as κ, β → 0.
    """
    lf, lfk = log_kde_components(scale(hp, X), scale(hp, traj), labels, m, hp['h'])
    lab = np.argmax(lfk, axis=1) if m > 0 else np.full(len(X), -1, dtype=int)
    lab[lf < hp['tau']] = -1
    return (lab, lf) if return_lf else lab


def hard_criterion(s_full: np.ndarray, s_core: np.ndarray, unassigned: str = 'singletons',
                   w_merge: float = 1.0) -> float:
    """
    VoI_ref = H(full | coreset) + H(coreset | full).  With w_merge != 1 the first term,
    which grows when the coreset merges full-data clusters, is weighted by w_merge.
    """
    if w_merge == 1.0:
        return voi_ref(s_full, s_core) if unassigned == 'singletons' else voi(s_full, s_core)
    from coreset.distances import singletons_for_unassigned
    a, b = np.asarray(s_full), np.asarray(s_core)
    if unassigned == 'singletons':
        a, b = singletons_for_unassigned(a), singletons_for_unassigned(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    J = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(J, (ai, bi), 1.0 / len(a))
    H = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    HJ, Hf, Hc = H(J), H(J.sum(1)), H(J.sum(0))
    return w_merge * (HJ - Hc) + (HJ - Hf)


# ── Soft pieces (JAX) ────────────────────────────────────────────────────────

def _xlogx(p):
    import jax.numpy as jnp
    pos = p > 0
    return jnp.where(pos, p * jnp.log(jnp.where(pos, p, 1.0)), 0.0)


def _ent(p):
    import jax.numpy as jnp
    return -jnp.sum(_xlogx(p))


def voi_from_joint(M, w_merge=1.0):
    """
    VoI (nats) of a joint table M (rows = full data, columns = coreset; entries ≥ 0, sum 1)
    = H(rows | cols) + H(cols | rows); the first term is weighted by w_merge.
    """
    HM = _ent(M)
    return w_merge * (HM - _ent(M.sum(0))) + (HM - _ent(M.sum(1)))


def soft_ref_labels(A, b, Yn, mean, s, h, tau, lab, X, kappa, beta, M_pad: int):
    """
    Soft coreset labels of the reference points X for one iteration.

    Yn (L, d) raw coreset trajectory, transformed here as Yn Aᵀ + b; lab (L,) its
    component ids under the CURRENT transform, constants (sub-threshold = M_pad).
    Returns g (B,) = P(above τ) and w (B, M_pad) = P(component | above), zero on
    empty / padded components; g = 0 when there is no component at all.
    """
    import jax
    import jax.numpy as jnp
    L, d  = Yn.shape
    Z     = ((Yn @ A.T + b) - mean) / s
    Xs    = (X - mean) / s
    sq    = jnp.maximum(jnp.sum(Xs ** 2, 1)[:, None] + jnp.sum(Z ** 2, 1)[None] - 2.0 * Xs @ Z.T, 0.0)
    e     = -0.5 * sq / h ** 2                                              # (B, L)
    const = -d * jnp.log(h) - 0.5 * d * jnp.log(2.0 * jnp.pi) - jnp.log(L)
    lf    = jax.scipy.special.logsumexp(e, axis=1) + const
    g     = jax.nn.sigmoid((lf - tau) / kappa)

    # per-component log-sum-exp via segment ops (exact, stable; −1 points → dummy segment)
    eT    = e.T                                                             # (L, B)
    emax  = jax.ops.segment_max(eT, lab, num_segments=M_pad + 1)
    emax  = jax.lax.stop_gradient(jnp.where(jnp.isfinite(emax), emax, 0.0))
    ssum  = jax.ops.segment_sum(jnp.exp(eT - emax[lab]), lab, num_segments=M_pad + 1)[:M_pad]
    valid = ssum[:, 0] > 0                                                  # non-empty components
    lfk   = (jnp.log(jnp.where(valid[:, None], ssum, 1.0)) + emax[:M_pad]).T  # (B, M_pad), up to const
    w     = jax.nn.softmax(jnp.where(valid[None], lfk / beta, -1e30), axis=1) * valid[None]
    g     = jnp.where(jnp.any(valid), g, 0.0)
    return g, w


def soft_voi_one(A, b, Yn, mean, s, h, tau, lab, sN, X, kappa, beta, M_pad: int, KN_pad: int,
                 singletons: bool = True, w_merge=1.0):
    """
    Soft VoI between the full-data labels sN (B,) of the reference points and the
    soft coreset labels, for one iteration.  Equals the hard VoI_ref as κ, β → 0.
    """
    import jax
    import jax.numpy as jnp
    g, w = soft_ref_labels(A, b, Yn, mean, s, h, tau, lab, X, kappa, beta, M_pad)
    B    = X.shape[0]
    qc   = g[:, None] * w                                       # (B, M_pad)  in a component
    q0   = 1.0 - g                                              # (B,)        sub-threshold
    if not singletons:
        P = jax.nn.one_hot(sN + 1, KN_pad + 1)                  # −1 → column 0
        M = P.T @ jnp.concatenate([q0[:, None], qc], axis=1) / B
        return voi_from_joint(M, w_merge)
    # sub-threshold points on either side are singletons: the joint table has the
    # (component × component) block plus one entry per point for each singleton side
    core = sN >= 0
    P    = jax.nn.one_hot(sN, KN_pad)                           # rows of −1 points are zero
    J11  = P.T @ qc / B
    e_cs = jnp.where(core, q0, 0.0) / B                         # full comp × coreset singleton
    e_sc = jnp.where(core[:, None], 0.0, qc) / B                # full singleton × coreset comp
    e_ss = jnp.where(core, 0.0, q0) / B                         # singleton × singleton
    H_M  = _ent(J11) + _ent(e_cs) + _ent(e_sc) + _ent(e_ss)
    rows = jnp.concatenate([P.sum(0) / B, jnp.where(core, 0.0, 1.0 / B)])
    cols = jnp.concatenate([qc.sum(0) / B, q0 / B])
    return w_merge * (H_M - _ent(cols)) + (H_M - _ent(rows))


_JIT_CACHE = {}


def soft_voi_batch_fn(M_pad: int, KN_pad: int, singletons: bool):
    """jit(value_and_grad) of the mean soft VoI over a batch of iterations (vmapped)."""
    key = (M_pad, KN_pad, singletons)
    if key not in _JIT_CACHE:
        import jax
        import jax.numpy as jnp

        def one(A, b, Yn, mean, s, h, tau, lab, sN, X, kappa, beta, w_merge):
            return soft_voi_one(A, b, Yn, mean, s, h, tau, lab, sN, X, kappa, beta,
                                M_pad, KN_pad, singletons, w_merge)

        def batch(A, b, Yn, mean, s, h, tau, lab, sN, X, kappa, beta, w_merge=1.0):
            v = jax.vmap(one, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, None, None, None, None))(
                A, b, Yn, mean, s, h, tau, lab, sN, X, kappa, beta, w_merge)
            return jnp.mean(v)

        _JIT_CACHE[key] = (jax.jit(jax.value_and_grad(batch, argnums=(0, 1))), jax.jit(batch))
    return _JIT_CACHE[key]


# ── Optimisation loop ────────────────────────────────────────────────────────

class _Iteration:
    """Everything of collection iteration t that does not depend on (A, b)."""

    def __init__(self, traj_full, traj_core, data_ref, K_cl, minsize, min_share=0.0):
        mf          = fit_clust(traj_full, K_cl, minsize=minsize, min_share=min_share)
        self.hp     = hp_from_model(mf, minsize, min_share)
        self.kN     = int(mf.labels.max() + 1) if (mf.labels >= 0).any() else 0
        self.sN     = hard_ref_labels(self.hp, data_ref, traj_full, mf.labels, self.kN)
        self.agree  = float(np.mean(hard_ref_labels(self.hp, traj_full, traj_full, mf.labels, self.kN)
                                    == mf.labels))
        self.Yn     = traj_core
        self.cache  = None                 # (labels, m, step) of the coreset topology


def _anneal(v0, v1, step, n_steps):
    return float(v0 * (v1 / v0) ** (step / max(n_steps - 1, 1)))


def _project(A, b, s_min, s_max, b_max):
    U, S, Vt = np.linalg.svd(A)
    A = (U * np.clip(S, s_min, s_max)) @ Vt
    nb = np.linalg.norm(b)
    return A, (b * (b_max / nb) if nb > b_max else b)


def optimize_soft_levelset(
    trajs_full: list,
    trajs_core: list,
    data_ref: np.ndarray,
    K_cl: int,
    rng: np.random.Generator,
    minsize: int = 1,
    n_steps: int = 500,
    lr: float = 3e-3,
    B: int = 256,
    iter_batch: int = 8,
    kappa0: float = 1.0,
    kappa_min: float = 0.05,
    beta0: float = 1.0,
    beta_min: float = 0.1,
    R: int = 10,
    eval_every: int = 25,
    s_min: float = 0.2,
    s_max: float = 5.0,
    b_max: float = None,
    unassigned: str = 'singletons',
    w_merge: float = 1.0,
    min_share: float = 0.0,
    progress: bool = True,
) -> dict:
    """
    Adam on the soft level-set VoI over affine T(y) = A y + b, projected onto
    K = {singular values of A in [s_min, s_max], ‖b‖ ≤ b_max} (contains I).

    Every eval_every steps (and at step 0 = identity, and at the end) the hard
    objective F = mean_t VoI_ref over all iterations and all reference points is
    computed; T̄ is the checkpoint with the lowest F.  Returns A_bar, b_bar and a log.
    """
    from jax.experimental import enable_x64

    assert unassigned in ('singletons', 'cell')
    d     = data_ref.shape[1]
    niter = len(trajs_full)
    N_r   = len(data_ref)
    B     = min(B, N_r)
    iter_batch = min(iter_batch, niter)
    if b_max is None:
        b_max = 3.0 * float(np.sqrt(np.sum(np.var(np.vstack(trajs_full), axis=0))))

    its    = [_Iteration(tf, tc, data_ref, K_cl, minsize, min_share) for tf, tc in zip(trajs_full, trajs_core)]
    KN_pad = max(1, max(it.kN for it in its))
    agree  = np.array([it.agree for it in its])
    if agree.min() < 0.95:
        print(f'  [soft-levelset] WARNING: hard_ref_labels vs Clust agreement on full-data '
              f'trajectories min {agree.min():.3f} (< 0.95), mean {agree.mean():.3f}')

    def topology(it, A, b, step):
        if it.cache is None or step - it.cache[2] >= R:
            lab, m = clust_shared(it.hp, it.Yn @ A.T + b)
            it.cache = (lab, m, step)
        return it.cache[0], it.cache[1]

    def hard_F(A, b, kappa):
        F, band, ms = [], [], []
        for it in its:
            Y = it.Yn @ A.T + b
            lab, m = clust_shared(it.hp, Y)
            s, lf  = hard_ref_labels(it.hp, data_ref, Y, lab, m, return_lf=True)
            F.append(hard_criterion(it.sN, s, unassigned, w_merge))
            band.append(np.abs(lf - it.hp['tau']) <= kappa * np.log(99.0))   # g ∈ [0.01, 0.99]
            ms.append(m)
        return float(np.mean(F)), float(np.mean(band)), ms

    A, b   = np.eye(d), np.zeros(d)
    mA, vA = np.zeros_like(A), np.zeros_like(A)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    M_pad  = 16
    log    = {'step': [], 'soft_loss': [], 'kappa': [], 'beta': [], 'm_sampled': [],
              'eval_step': [], 'hard_F': [], 'band_fraction': [], 'eval_m': []}
    best   = None

    def evaluate(step, kappa):
        nonlocal best
        F, band, ms = hard_F(A, b, kappa)
        log['eval_step'].append(step)
        log['hard_F'].append(F)
        log['band_fraction'].append(band)
        log['eval_m'].append(ms)
        if best is None or F < best[0]:
            best = (F, step, A.copy(), b.copy())

    steps = range(n_steps)
    if progress:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc='  soft-levelset', leave=False, unit='step', ncols=72)
        except ImportError:
            pass

    with enable_x64():
        X_all = np.asarray(data_ref, dtype=np.float64)
        for step in steps:
            kappa = _anneal(kappa0, kappa_min, step, n_steps)
            beta  = _anneal(beta0, beta_min, step, n_steps)
            if step % eval_every == 0:
                evaluate(step, kappa)
            T_b  = rng.choice(niter, size=iter_batch, replace=False)
            idx  = rng.choice(N_r, size=B, replace=False)
            tops = [topology(its[t], A, b, step) for t in T_b]
            ms   = [m for _, m in tops]
            while max(ms) > M_pad:
                M_pad *= 2
            vg, _ = soft_voi_batch_fn(M_pad, KN_pad, unassigned == 'singletons')
            lab_b = np.stack([np.where(lab >= 0, lab, M_pad) for lab, _ in tops])
            loss, (gA, gb) = vg(
                A, b,
                np.stack([its[t].Yn for t in T_b]),
                np.stack([its[t].hp['mean'] for t in T_b]),
                np.array([its[t].hp['s'] for t in T_b]),
                np.array([its[t].hp['h'] for t in T_b]),
                np.array([its[t].hp['tau'] for t in T_b]),
                lab_b,
                np.stack([its[t].sN[idx] for t in T_b]),
                X_all[idx], kappa, beta, w_merge,
            )
            gA, gb = np.asarray(gA), np.asarray(gb)
            k = step + 1
            mA = b1 * mA + (1 - b1) * gA;  vA = b2 * vA + (1 - b2) * gA ** 2
            mb = b1 * mb + (1 - b1) * gb;  vb = b2 * vb + (1 - b2) * gb ** 2
            A  = A - lr * (mA / (1 - b1 ** k)) / (np.sqrt(vA / (1 - b2 ** k)) + eps)
            b  = b - lr * (mb / (1 - b1 ** k)) / (np.sqrt(vb / (1 - b2 ** k)) + eps)
            A, b = _project(A, b, s_min, s_max, b_max)
            log['step'].append(step)
            log['soft_loss'].append(float(loss))
            log['kappa'].append(kappa)
            log['beta'].append(beta)
            log['m_sampled'].append(ms)
        evaluate(n_steps, _anneal(kappa0, kappa_min, n_steps - 1, n_steps))

    F_best, step_best, A_bar, b_bar = best
    log.update({
        'agreement_full_mean': float(agree.mean()), 'agreement_full_min': float(agree.min()),
        'kN': [it.kN for it in its], 'checkpoint_step': int(step_best), 'checkpoint_F': float(F_best),
        'F_identity': float(log['hard_F'][0]),
    })
    return {'A_bar': A_bar, 'b_bar': b_bar, 'log': log,
            'config': {'B': B, 'iter_batch': iter_batch, 'kappa0': kappa0, 'kappa_min': kappa_min,
                       'beta0': beta0, 'beta_min': beta_min, 'R': R, 'eval_every': eval_every,
                       'n_steps': n_steps, 'lr': lr, 's_min': s_min, 's_max': s_max,
                       'b_max': float(b_max), 'unassigned': unassigned, 'w_merge': w_merge, 'min_share': min_share}}
