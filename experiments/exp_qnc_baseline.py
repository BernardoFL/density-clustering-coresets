"""
QNC baseline (Naik, Rousseau & Campbell 2022) for the valley-separated simulation and Levine_13dim.

QNC keeps the uniform support y* of each replicate and refines its likelihood weights by
quasi-Newton steps on the KL divergence between the weighted-coreset and full-data posteriors of a
diagonal Gaussian mixture (K_max = K_cl components, blocked Gibbs).  The weighted coreset enters the
same predictive construction through the weighted urn (coreset.polya_urn.polya_urn_weighted).

Each replicate is replayed from its stored seeds: reference split, support and evaluation draws are
the ones of the reported runs, so the full-data and uniform partitions (and hence the reported
uniform losses, which are checked) are reproduced exactly, and QNC is evaluated on the same paired
draws as the coreset.  The QNC trajectory uses its own random stream.

Usage:  python experiments/exp_qnc_baseline.py --study valley --n_jobs 16
        python experiments/exp_qnc_baseline.py --study levine --n_jobs 16
        python experiments/exp_qnc_baseline.py --summarise
"""
import os, sys, glob, json, itertools, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'qnc_baseline')
QNC = dict(n_qn=10, step=0.3, gibbs_iter=600, gibbs_burn=300, gibbs_thin=3, ridge=1e-2)
QNC_TAG = 7                                   # extra spawn-key entry for the QNC random streams


# ── QNC weights ──────────────────────────────────────────────────────────────

def _mix_loglik(X, w, mu, s2, chunk=20000):
    """log sum_k w_k N(x | mu_k, diag s2_k) for each row of X."""
    from scipy.special import logsumexp
    from inference.gibbs_diag import _loglik_matrix
    return np.concatenate([logsumexp(_loglik_matrix(X[i:i + chunk], w, mu, s2), axis=1)
                           for i in range(0, len(X), chunk)])


def qnc_weights(y, X_full, K, seed, cfg=QNC):
    """
    Quasi-Newton coreset weights on the fixed support y (Naik et al. 2022, Algorithm 1):
    starting from w = (N / n) 1, each step draws S samples from the w-weighted coreset posterior,
    forms centred log-likelihood features g_s(x) = log p(x | theta_s) - mean_s, and takes a damped
    quasi-Newton step towards argmin_w || sum_full g - w^T g_y ||^2 (ridge-regularised), projected
    onto w >= 0.  Returns the iterate with the smallest relative residual (QNC's own objective,
    never the predictive loss) and the log.
    """
    from inference.gibbs_diag import gibbs_diag_mixture, default_prior
    rng = np.random.default_rng(seed)
    n = len(y); N = len(X_full)
    w = np.full(n, N / n)
    pr = default_prior(y)
    log, best = [], None
    for k in range(cfg['n_qn'] + 1):
        fit = gibbs_diag_mixture(y, K, cfg['gibbs_iter'], cfg['gibbs_burn'], cfg['gibbs_thin'],
                                 prior=pr, seed=int(rng.integers(2 ** 31)), weights=w)
        S = len(fit['weights'])
        Gy = np.empty((n, S)); tf = np.empty(S)
        for s in range(S):
            Gy[:, s] = _mix_loglik(y, fit['weights'][s], fit['mu'][s], fit['s2'][s])
            tf[s] = _mix_loglik(X_full, fit['weights'][s], fit['mu'][s], fit['s2'][s]).sum()
        Gy -= Gy.mean(1, keepdims=True); tf -= tf.mean()
        # QNC's own objective at the current weights: relative residual of the log-likelihood match
        # under samples from the current weighted posterior (an estimate of its KL gradient norm)
        resid = float(np.linalg.norm(tf - w @ Gy) / max(np.linalg.norm(tf), 1e-300))
        log.append({'step': k, 'resid': resid, 'sum_w': float(w.sum()), 'n_zero': int((w == 0).sum())})
        if best is None or resid < best[0]:
            best = (resid, w.copy())
        if k == cfg['n_qn']:
            break
        H = Gy @ Gy.T / S
        H[np.diag_indices(n)] += cfg['ridge'] * np.trace(H) / n + 1e-12
        w_ls = np.linalg.solve(H, Gy @ tf / S)                # undamped quasi-Newton target
        w = np.maximum(w + cfg['step'] * (w_ls - w), 0.0)     # damped step, projected onto w >= 0
        if w.sum() <= 0:
            break
    return best[1], log


# ── Shared evaluation of one paired draw ─────────────────────────────────────

def _partitions(tN, trajs, ref, K_cl):
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    mf = fit_clust(tN, K_cl); hp = hp_from_model(mf)
    sN = hard_ref_labels(hp, ref, tN, mf.labels, int(mf.labels.max() + 1) if (mf.labels >= 0).any() else 0)
    out = []
    for t in trajs:
        lab, m = clust_shared(hp, t)
        out.append(hard_ref_labels(hp, ref, t, lab, m))
    return sN, out


# ── Valley-separated simulation ──────────────────────────────────────────────

def run_valley(sep, p_min, n, rep):
    from experiments.exp_sim_valleys import generate, CFG, SEPS, PMINS, NS, MASTER
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from sklearn.metrics import adjusted_rand_score as ari
    cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(n)
    ss = np.random.SeedSequence([MASTER, cell, rep])
    s_data, s_core, s_eval = ss.spawn(3)
    X, lab, *_ = generate(sep, p_min, np.random.default_rng(s_data))
    ss_split, ss_support, *_ = s_core.spawn(5)
    perm = np.random.default_rng(ss_split).permutation(len(X))
    ref_idx, A_idx = np.sort(perm[:CFG['N_r']]), np.sort(perm[CFG['N_r']:])
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]

    t0 = time.perf_counter()
    w, qlog = qnc_weights(y, XA, CFG['K'], np.random.SeedSequence([MASTER, cell, rep, QNC_TAG, 0]))
    t_qnc = time.perf_counter() - t0

    stored = json.load(open(os.path.join(RES, 'sim_valleys', f's{sep:g}_p{p_min:g}_n{n:04d}_rep{rep:02d}.json')))
    rng = np.random.default_rng(s_eval)
    rng_q = np.random.default_rng(np.random.SeedSequence([MASTER, cell, rep, QNC_TAG, 1]))
    ev = {k: [] for k in ('voi_unif', 'voi_qnc', 'ari_unif', 'ari_qnc')}
    for _ in range(CFG['V_eval']):
        phi = sample_niw_hyperprior(X, rng); base = build_gaussian_base(*phi, rng)
        tN = polya_urn(XA, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        tq = _weighted_urn(y, w, phi, rng_q, 1.0, CFG['L'])
        sN, (su, sq) = _partitions(tN, [tn, tq], R, CFG['K'])
        ev['voi_unif'].append(voi_ref(sN, su)); ev['voi_qnc'].append(voi_ref(sN, sq))
        ev['ari_unif'].append(ari(lab[ref_idx], su)); ev['ari_qnc'].append(ari(lab[ref_idx], sq))
    check = float(np.max(np.abs(np.array(ev['voi_unif']) - np.array(stored['eval']['voi_unif']))))
    return {'study': 'valley', 'sep': sep, 'p_min': p_min, 'n': n, 'rep': rep, 'eval': ev,
            'voi_core': stored['eval']['voi_core'], 'ari_core': stored['eval']['ari_core'],
            'ari_full': stored['eval']['ari_full'], 'reproduction_max_abs_diff': check,
            'qnc_weights': w, 'qnc_log': qnc_log_fix(qlog), 't_qnc': t_qnc, 'qnc_cfg': QNC}


def _weighted_urn(y, w, phi, rng_q, alpha, L):
    """Weighted urn with the same phi as the paired draw; all of its randomness comes from rng_q."""
    from coreset.polya_urn import polya_urn_weighted
    mu, Sigma = phi
    return polya_urn_weighted(y, w, lambda: rng_q.multivariate_normal(mu, Sigma), alpha=alpha, L=L, rng=rng_q,
                              reinforce=True)


def qnc_log_fix(log):
    return [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()} for d in log]


# ── Levine_13dim ─────────────────────────────────────────────────────────────

def run_levine(n, rep):
    from experiments.exp_levine13 import half, MASTER, SEED_NAMES
    from experiments.runlog import child_seeds
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    stored = json.load(open(os.path.join(RES, 'levine13_test', f'test_n{n:04d}_rep{rep:02d}.json')))
    cfg = stored['config']
    X, _, _ = half('test', cfg)
    ss = child_seeds(MASTER, rep, SEED_NAMES)['coreset']
    ss_split, ss_support, _, _, ss_select = ss.spawn(5)
    perm = np.random.default_rng(ss_split).permutation(len(X))
    ref_idx, A_idx = np.sort(perm[:cfg['N_r']]), np.sort(perm[cfg['N_r']:])
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    assert np.array_equal(ref_idx, stored['coreset']['ref_idx'])
    assert np.array_equal(support_idx, stored['coreset']['support_idx'])
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]

    t0 = time.perf_counter()
    w, qlog = qnc_weights(y, XA, cfg['K_cl'], np.random.SeedSequence([MASTER, rep, n, QNC_TAG, 0]))
    t_qnc = time.perf_counter() - t0

    rng = np.random.default_rng(ss_select)
    nsel = cfg['nval'] // 2
    for _ in range(nsel):                                   # replay the selection draws' stream
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
    rng_q = np.random.default_rng(np.random.SeedSequence([MASTER, rep, n, QNC_TAG, 1]))
    ev = {'voi_unif': [], 'voi_qnc': []}
    for _ in range(cfg['nval'] - nsel):
        phi = sample_niw_hyperprior(X, rng); base = build_gaussian_base(*phi, rng)
        tN = polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        tq = _weighted_urn(y, w, phi, rng_q, cfg['alpha'], cfg['L'])
        sN, (su, sq) = _partitions(tN, [tn, tq], R, cfg['K_cl'])
        ev['voi_unif'].append(voi_ref(sN, su)); ev['voi_qnc'].append(voi_ref(sN, sq))
    check = float(np.max(np.abs(np.array(ev['voi_unif']) - np.array(stored['coreset']['V_I']))))
    return {'study': 'levine', 'n': n, 'rep': rep, 'eval': ev, 'voi_core': stored['coreset']['V_bar'],
            'reproduction_max_abs_diff': check, 'qnc_weights': w, 'qnc_log': qnc_log_fix(qlog),
            't_qnc': t_qnc, 'qnc_cfg': QNC}


# ── Driver ───────────────────────────────────────────────────────────────────

def _path(job):
    if job[0] == 'valley':
        _, sep, p, n, rep = job
        return os.path.join(OUT, 'valley', f's{sep:g}_p{p:g}_n{n:04d}_rep{rep:02d}.json')
    _, n, rep = job
    return os.path.join(OUT, 'levine', f'n{n:04d}_rep{rep:02d}.json')


def _worker(job):
    path = _path(job)
    if os.path.exists(path):
        return path, 'skip'
    t0 = time.perf_counter()
    res = run_valley(*job[1:]) if job[0] == 'valley' else run_levine(*job[1:])
    res['t_total'] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, res)
    return path, f"{res['t_total']:.0f}s check={res['reproduction_max_abs_diff']:.2g}"


def jobs(study, reps=None):
    if study == 'valley':
        from experiments.exp_sim_valleys import SEPS, PMINS, NS
        return [('valley', s, p, n, r) for s, p, n in itertools.product(SEPS, PMINS, NS) for r in (reps or range(20))]
    return [('levine', n, r) for n in (50, 100, 150, 250, 400, 1000) for r in (reps or range(30))]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 100 * (a.mean() - b.mean()) / a.mean()


def _lo(a, b):
    from scipy import stats
    dd = np.asarray(a) - np.asarray(b)
    if dd.std() == 0:
        return 0.0
    return dd.mean() - stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))


def summarise():
    for study in ('valley', 'levine'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, study, '*.json')))]
        if not runs:
            continue
        print(f'\n{study}: {len(runs)} replicates; max |uniform loss - reported| = '
              f'{max(r["reproduction_max_abs_diff"] for r in runs):.2g}')
        keys = (('sep', 'p_min', 'n') if study == 'valley' else ('n',))
        cells = sorted({tuple(r[k] for k in keys) for r in runs}, key=lambda c: tuple(-x if i < 2 and study == 'valley' else x for i, x in enumerate(c)))
        print(' '.join(f'{k:>5s}' for k in keys) + ' | QNC vs unif: decr [IQR]  int>0 | coreset vs QNC: decr  int>0'
              + (' | ARI unif  qnc  core' if study == 'valley' else '') + ' | sum w/N  t_qnc')
        for c in cells:
            S = [r for r in runs if tuple(r[k] for k in keys) == c]
            rq = np.array([_rel(r['eval']['voi_unif'], r['eval']['voi_qnc']) for r in S])
            iq = sum(_lo(r['eval']['voi_unif'], r['eval']['voi_qnc']) > 0 for r in S)
            rc = np.array([_rel(r['eval']['voi_qnc'], r['voi_core']) for r in S])
            ic = sum(_lo(r['eval']['voi_qnc'], r['voi_core']) > 0 for r in S)
            line = (' '.join(f'{x:5g}' for x in c) + f' | {np.median(rq):6.1f}% [{np.percentile(rq, 25):4.0f},'
                    f'{np.percentile(rq, 75):4.0f}] {iq:3d}/{len(S):<3d} | {np.median(rc):6.1f}% {ic:3d}/{len(S):<3d}')
            if study == 'valley':
                line += ' | ' + ' '.join(
                    f'{np.mean([np.mean(v) for v in vals]):.3f}' for vals in (
                        [r['eval']['ari_unif'] for r in S], [r['eval']['ari_qnc'] for r in S], [r['ari_core'] for r in S]))
            Nw = [np.sum(r['qnc_weights']) for r in S]
            line += f' | {np.median(Nw):9.0f}  {np.median([r["t_qnc"] for r in S]):4.0f}s'
            print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', choices=['valley', 'levine'])
    p.add_argument('--reps', default=None, help='e.g. 0-1')
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    if a.summarise:
        return summarise()
    from experiments.runlog import parse_range
    J = jobs(a.study, parse_range(a.reps) if a.reps else None)
    if a.n_jobs == 1:
        for j in J:
            print(*_worker(j), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(a.n_jobs) as pool:
            for out in pool.imap_unordered(_worker, J):
                print(*out, flush=True)


if __name__ == '__main__':
    main()
