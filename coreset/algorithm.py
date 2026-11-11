import time

import numpy as np

from coreset.polya_urn import polya_urn
from coreset.clustering import fit_clust


def _iter_progress(n, desc):
    try:
        from tqdm import tqdm as _tqdm
        return _tqdm(range(n), desc=desc, leave=False, unit='iter', ncols=72)
    except ImportError:
        return range(n)


def build_coreset_selected(
    data: np.ndarray,
    n: int,
    niter: int,
    L: int,
    sample_hyperprior,
    build_base_measure,
    N_r: int,
    K_cl: int,
    alpha: float = 1.0,
    nval: int = 100,
    minsize: int = 1,
    lr: float = 3e-3,
    n_steps: int = 500,
    B: int = 256,
    iter_batch: int = 8,
    kappa0: float = 1.0,
    kappa_min: float = 0.05,
    beta0: float = 1.0,
    beta_min: float = 0.1,
    R: int = 10,
    eval_every: int = 25,
    unassigned: str = 'singletons',
    w_merge: float = 1.0,
    min_share: float = 0.0,
    lambda_grid=tuple(np.round(np.linspace(0.0, 1.0, 11), 2)),
    seed=None,
    urn_reinforce: bool = True,
    A_bar_override: np.ndarray = None,
    b_bar_override: np.ndarray = None,
) -> dict:
    """
    Algorithm 1 of the paper: coreset construction with reference split and selection step.

    1. Split y_{1:N} uniformly at random into a reference set (size N_r) and
       the remainder y_A.  Both Pólya urns run on y_A only, and the coreset
       support is drawn uniformly from y_A, independently of the data values.
    2. Collection phase (synchronous pairs of full-data / coreset trajectories
       under the same φ_t), then Adam on the soft level-set VoI on the reference
       set (coreset.soft_levelset) over affine T(y) = A y + b.  T̄ is the
       checkpoint with the lowest hard VoI_ref on the training draws.
    3. Selection step along the path T_λ = I + λ(T̄ − I), λ in lambda_grid
       (λ = 0 is the uniform subsample, λ = 1 the full transform).  The nval
       fresh hyperparameter draws are split in two independent halves:
         * selection draws: λ* minimises the mean VoI_ref between the full-data
           partition and the T_λ-coreset partition of the reference set
           (ties go to the smallest λ, i.e. towards the uniform subsample);
         * evaluation draws: T_λ* and I are compared on fresh draws, so the
           reported paired differences are not biased by the choice of λ*.
       The output transform is T_out = T_λ*.

    Clust of a coreset trajectory uses the KDE hyperparameters (scaling, h, τ,
    radius) fitted on the paired full-data trajectory; reference points are
    labelled by hard_ref_labels (argmax per-component KDE share, −1 below τ).

    seed : int or np.random.SeedSequence; split into independent child streams
           for split / support / collect / transform / select.
    urn_reinforce : passed to polya_urn (see its docstring).
    A_bar_override, b_bar_override : skip the Adam phase (tests).

    Returns a dict; see the keys at the end of this function.
    """
    from coreset.soft_levelset import optimize_soft_levelset

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    ss_split, ss_support, ss_collect, ss_transform, ss_select = ss.spawn(5)

    N, d = data.shape
    assert N_r <= N / 2, f'N_r={N_r} must be at most N/2={N / 2}'

    # Step 0: reference split
    perm    = np.random.default_rng(ss_split).permutation(N)
    ref_idx = np.sort(perm[:N_r])
    A_idx   = np.sort(perm[N_r:])
    data_A  = data[A_idx]
    data_ref = data[ref_idx]

    # Step 1: coreset support, uniform from y_A
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    y_star      = data[support_idx].copy()

    # Step 2: collection phase
    t0  = time.perf_counter()
    rng = np.random.default_rng(ss_collect)
    raw_trajs_full, raw_trajs_core = [], []
    for _ in _iter_progress(niter, '  collect'):
        phi_t = sample_hyperprior(rng)
        base  = build_base_measure(phi_t, rng)
        raw_trajs_full.append(polya_urn(data_A, base, alpha=alpha, L=L, rng=rng, reinforce=urn_reinforce))
        raw_trajs_core.append(polya_urn(y_star, base, alpha=alpha, L=L, rng=rng, reinforce=urn_reinforce))
    t_collect = time.perf_counter() - t0

    # Step 3: transform phase (Adam on the soft level-set VoI)
    t0 = time.perf_counter()
    surrogate_log = surrogate_config = None
    if A_bar_override is not None:
        A_bar, b_bar = np.asarray(A_bar_override, float), np.asarray(b_bar_override, float)
    else:
        res = optimize_soft_levelset(
            raw_trajs_full, raw_trajs_core, data_ref, K_cl,
            rng=np.random.default_rng(ss_transform), minsize=minsize,
            n_steps=n_steps, lr=lr, B=B, iter_batch=iter_batch,
            kappa0=kappa0, kappa_min=kappa_min, beta0=beta0, beta_min=beta_min,
            R=R, eval_every=eval_every, unassigned=unassigned, w_merge=w_merge, min_share=min_share,
        )
        A_bar, b_bar = res['A_bar'], res['b_bar']
        surrogate_log, surrogate_config = res['log'], res['config']
    t_transform = time.perf_counter() - t0

    def apply_path(X, lam):
        return ((np.eye(d) + lam * (A_bar - np.eye(d))) @ X.T).T + lam * b_bar

    # Step 4: selection along the path from I to T̄, on fresh draws
    t0 = time.perf_counter()
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    assert lambda_grid[0] == 0.0 and np.all(np.diff(lambda_grid) > 0), 'grid must start at 0 and increase'
    nval_select = nval // 2
    nval_eval   = nval - nval_select
    rng_sel     = np.random.default_rng(ss_select)
    sel_kw = dict(alpha=alpha, L=L, K_cl=K_cl, minsize=minsize, unassigned=unassigned, w_merge=w_merge, min_share=min_share, rng=rng_sel, urn_reinforce=urn_reinforce)
    sel = _select_lambda(data_A, y_star, data_ref, apply_path, lambda_grid,
                         sample_hyperprior, build_base_measure, nval=nval_select, **sel_kw)
    select_curve = sel['V'].mean(axis=0)               # mean VoI_ref per λ
    lam_out      = float(lambda_grid[int(np.argmin(select_curve))])   # first = smallest λ on ties
    A_out = np.eye(d) + lam_out * (A_bar - np.eye(d))
    b_out = lam_out * b_bar
    ev = _select_lambda(data_A, y_star, data_ref, apply_path, np.array([0.0, lam_out]),
                        sample_hyperprior, build_base_measure, nval=nval_eval, **sel_kw)
    t_select = time.perf_counter() - t0

    V_I, V_out = ev['V'][:, 0], ev['V'][:, 1]
    J_hat_bar  = float(np.mean(V_out))
    J_hat_I    = float(np.mean(V_I))
    selected   = 'T_bar' if lam_out > 0 else 'I'      # 'T_bar' = a (possibly partial) transform is used

    return {
        'family': 'affine',
        'A_bar': A_bar, 'b_bar': b_bar,
        'A_out': A_out, 'b_out': b_out,
        'selected': selected,
        'y_star': y_star,
        'y_coreset': apply_path(y_star, lam_out),
        'support_idx': support_idx,
        'ref_idx': ref_idx,
        'lambda_out': lam_out,
        'lambda_grid': lambda_grid,
        'select_curve': select_curve,                  # selection draws, mean VoI_ref per λ
        'J_hat_bar': J_hat_bar,                        # evaluation draws, T_out
        'J_hat_I': J_hat_I,                            # evaluation draws, I
        'V_bar': V_out,                                # evaluation draws, per draw, T_out
        'V_I': V_I,                                    # evaluation draws, per draw, I
        'clust_params': ev['clust_params'],
        'surrogate_log': surrogate_log,
        'timings': {
            'collect': t_collect,
            'transform': t_transform,
            'select': t_select,
        },
        'config': {
            'N': N, 'd': d, 'n': n, 'N_r': N_r, 'niter': niter, 'L': L,
            'nval': nval, 'nval_select': nval_select, 'nval_eval': nval_eval,
            'selection': 'lambda path, split draws',
            'family': 'affine',
            'surrogate': 'soft level-set VoI on the reference set',
            'surrogate_config': surrogate_config,
            'minsize': minsize, 'K_cl': K_cl, 'alpha': alpha,
            'clust_backend': 'levelset (dimension-adapted), coreset shares full-data h/tau',
            'ref_labels': 'argmax per-component KDE share',
            'w_merge': w_merge,
            'min_share': min_share,          # components below this share of a trajectory are sub-threshold
            'voi_ref': unassigned,            # 'singletons': sub-threshold reference points are singletons
            'seed_entropy': str(ss.entropy),
            'seed_spawn_key': list(ss.spawn_key),
        },
    }


def _select_lambda(
    data_A: np.ndarray,
    y_star: np.ndarray,
    data_ref: np.ndarray,
    apply_path,
    lambdas: np.ndarray,
    sample_hyperprior,
    build_base_measure,
    alpha: float,
    L: int,
    K_cl: int,
    nval: int,
    minsize: int,
    unassigned: str,
    rng: np.random.Generator,
    w_merge: float = 1.0,
    min_share: float = 0.0,
    urn_reinforce: bool = True,
) -> dict:
    """
    VoI_ref of T_λ-coreset partitions along the path T_λ = I + λ(T̄ − I);
    apply_path(X, λ) applies T_λ to a coreset trajectory X.

    For each draw v: fresh φ_v, fresh full-data and coreset trajectories; Clust
    is fitted on the full-data trajectory and its hyperparameters are shared by
    the coreset trajectory under every T_λ; reference points are labelled with
    hard_ref_labels.  Values for different λ are paired on the same draws.
    Returns V of shape (nval, len(lambdas)).
    """
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels, hard_criterion

    V = np.empty((nval, len(lambdas)))
    clust_params = []
    for v in _iter_progress(nval, '  select'):
        phi_v  = sample_hyperprior(rng)
        base   = build_base_measure(phi_v, rng)
        traj_N = polya_urn(data_A, base, alpha=alpha, L=L, rng=rng, reinforce=urn_reinforce)
        traj_n = polya_urn(y_star, base, alpha=alpha, L=L, rng=rng, reinforce=urn_reinforce)

        m_full = fit_clust(traj_N, K_cl, minsize=minsize, min_share=min_share)
        hp     = hp_from_model(m_full, minsize, min_share)
        kN     = int(m_full.labels.max() + 1) if (m_full.labels >= 0).any() else 0
        s_full = hard_ref_labels(hp, data_ref, traj_N, m_full.labels, kN)
        params = {'tau': hp['tau'], 'h': hp['h'], 'k_full': kN}
        for j, lam in enumerate(lambdas):
            X       = apply_path(traj_n, lam)
            lab, m  = clust_shared(hp, X)
            V[v, j] = hard_criterion(s_full, hard_ref_labels(hp, data_ref, X, lab, m), unassigned, w_merge)
            params[f'm_lambda={lam:g}'] = m
        clust_params.append(params)
    return {'V': V, 'clust_params': clust_params}
