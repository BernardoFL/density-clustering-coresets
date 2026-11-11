"""
Observation-space transport baseline (Claici, Genevay & Solomon 2020) for the valley-separated
simulation and Levine_13dim.

The baseline keeps the uniform support y* of each replicate and fits an affine map T(y) = A y + b
minimising W_2{ P_hat[T(y*)], P_hat(x) } between the transformed support and the observations x used
by the urn (a random subsample of M_TARGET of them), by alternating an exact optimal coupling with a
least-squares update of (A, b) towards the barycentric targets.  Each step does not increase W_2^2,
so the iteration is monotone; it starts at the identity and keeps the best iterate.  The fitted map
is used as the coreset map is: it is applied to the coreset urn trajectory, with no
lambda-path selection.

Each replicate is replayed from its stored seeds (as in exp_qnc_baseline): reference split, support
and evaluation draws are those of the reported runs, so the uniform losses are reproduced exactly
(checked) and the baseline is evaluated on the same paired draws, and the same coreset trajectory,
as the coreset.  The map fit uses its own random stream (target subsample only).

Usage:  python experiments/exp_claici_baseline.py --study valley --n_jobs 14
        python experiments/exp_claici_baseline.py --study levine --n_jobs 14
        python experiments/exp_claici_baseline.py --summarise
"""
import os, sys, glob, json, itertools, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json
from experiments.exp_qnc_baseline import _partitions, _rel, _lo

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'claici_baseline')
W2 = dict(M_target=5000, max_iter=100, tol=1e-7)
TAG = 11                                      # extra spawn-key entry for the baseline's random stream


def w2_affine(y, X_target, seed, cfg=W2):
    """Affine map minimising W_2^2 between P_hat[A y + b] and P_hat(Z), Z a subsample of X_target."""
    import ot
    rng = np.random.default_rng(seed)
    n, d = y.shape
    Z = X_target[rng.choice(len(X_target), size=min(cfg['M_target'], len(X_target)), replace=False)]
    a, bz = np.full(n, 1 / n), np.full(len(Z), 1 / len(Z))
    Y1 = np.hstack([y, np.ones((n, 1))])
    A, b = np.eye(d), np.zeros(d)
    log, best = [], None
    for it in range(cfg['max_iter']):
        Ty = y @ A.T + b
        M = ot.dist(Ty, Z)                                   # squared Euclidean costs
        G = ot.emd(a, bz, M, numItermax=10 ** 7)
        cost = float((G * M).sum())
        log.append(cost)
        if best is None or cost < best[0]:
            best = (cost, A.copy(), b.copy())
        if it > 0 and log[-2] - cost <= cfg['tol'] * log[-2]:
            break
        coef, *_ = np.linalg.lstsq(Y1, n * G @ Z, rcond=None)   # barycentric targets
        A, b = coef[:d].T, coef[d]
    return best[1], best[2], {'w2sq': log, 'w2sq_identity': log[0], 'w2sq_best': best[0]}


def run_valley(sep, p_min, n, rep):
    from experiments.exp_sim_valleys import generate, CFG, SEPS, PMINS, NS, MASTER
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from sklearn.metrics import adjusted_rand_score as ari
    cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(n)
    s_data, s_core, s_eval = np.random.SeedSequence([MASTER, cell, rep]).spawn(3)
    X, lab, *_ = generate(sep, p_min, np.random.default_rng(s_data))
    ss_split, ss_support, *_ = s_core.spawn(5)
    perm = np.random.default_rng(ss_split).permutation(len(X))
    ref_idx, A_idx = np.sort(perm[:CFG['N_r']]), np.sort(perm[CFG['N_r']:])
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]

    t0 = time.perf_counter()
    A, b, wlog = w2_affine(y, XA, np.random.SeedSequence([MASTER, cell, rep, TAG]))
    t_fit = time.perf_counter() - t0

    stored = json.load(open(os.path.join(RES, 'sim_valleys', f's{sep:g}_p{p_min:g}_n{n:04d}_rep{rep:02d}.json')))
    rng = np.random.default_rng(s_eval)
    ev = {k: [] for k in ('voi_unif', 'voi_w2', 'ari_unif', 'ari_w2')}
    for _ in range(CFG['V_eval']):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        tN = polya_urn(XA, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        sN, (su, sw) = _partitions(tN, [tn, tn @ A.T + b], R, CFG['K'])
        ev['voi_unif'].append(voi_ref(sN, su)); ev['voi_w2'].append(voi_ref(sN, sw))
        ev['ari_unif'].append(ari(lab[ref_idx], su)); ev['ari_w2'].append(ari(lab[ref_idx], sw))
    check = float(np.max(np.abs(np.array(ev['voi_unif']) - np.array(stored['eval']['voi_unif']))))
    return {'study': 'valley', 'sep': sep, 'p_min': p_min, 'n': n, 'rep': rep, 'eval': ev,
            'voi_core': stored['eval']['voi_core'], 'ari_core': stored['eval']['ari_core'],
            'ari_full': stored['eval']['ari_full'], 'reproduction_max_abs_diff': check,
            'A': A, 'b': b, 'w2_log': wlog, 't_fit': t_fit, 'w2_cfg': W2}


def run_levine(n, rep):
    from experiments.exp_levine13 import half, MASTER, SEED_NAMES
    from experiments.runlog import child_seeds
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    stored = json.load(open(os.path.join(RES, 'levine13_test', f'test_n{n:04d}_rep{rep:02d}.json')))
    cfg = stored['config']
    X, _, _ = half('test', cfg)
    ss_split, ss_support, _, _, ss_select = child_seeds(MASTER, rep, SEED_NAMES)['coreset'].spawn(5)
    perm = np.random.default_rng(ss_split).permutation(len(X))
    ref_idx, A_idx = np.sort(perm[:cfg['N_r']]), np.sort(perm[cfg['N_r']:])
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    assert np.array_equal(ref_idx, stored['coreset']['ref_idx'])
    assert np.array_equal(support_idx, stored['coreset']['support_idx'])
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]

    t0 = time.perf_counter()
    A, b, wlog = w2_affine(y, XA, np.random.SeedSequence([MASTER, rep, n, TAG]))
    t_fit = time.perf_counter() - t0

    rng = np.random.default_rng(ss_select)
    nsel = cfg['nval'] // 2
    for _ in range(nsel):                                   # replay the selection draws' stream
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
    ev = {'voi_unif': [], 'voi_w2': []}
    for _ in range(cfg['nval'] - nsel):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        tN = polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        sN, (su, sw) = _partitions(tN, [tn, tn @ A.T + b], R, cfg['K_cl'])
        ev['voi_unif'].append(voi_ref(sN, su)); ev['voi_w2'].append(voi_ref(sN, sw))
    check = float(np.max(np.abs(np.array(ev['voi_unif']) - np.array(stored['coreset']['V_I']))))
    return {'study': 'levine', 'n': n, 'rep': rep, 'eval': ev, 'voi_core': stored['coreset']['V_bar'],
            'reproduction_max_abs_diff': check, 'A': A, 'b': b, 'w2_log': wlog, 't_fit': t_fit,
            'w2_cfg': W2}


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
    return path, f"{res['t_total']:.0f}s fit={res['t_fit']:.1f}s check={res['reproduction_max_abs_diff']:.2g}"


def jobs(study, reps=None):
    if study == 'valley':
        from experiments.exp_sim_valleys import SEPS, PMINS, NS
        return [('valley', s, p, n, r) for s, p, n in itertools.product(SEPS, PMINS, NS) for r in (reps or range(20))]
    return [('levine', n, r) for n in (50, 100, 150, 250, 400, 1000) for r in (reps or range(30))]


def summarise():
    for study in ('valley', 'levine'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, study, '*.json')))]
        if not runs:
            continue
        print(f'\n{study}: {len(runs)} replicates; max |uniform loss - reported| = '
              f'{max(r["reproduction_max_abs_diff"] for r in runs):.2g}')
        keys = (('sep', 'p_min', 'n') if study == 'valley' else ('n',))
        cells = sorted({tuple(r[k] for k in keys) for r in runs},
                       key=lambda c: tuple(-x if i < 2 and study == 'valley' else x for i, x in enumerate(c)))
        print(' '.join(f'{k:>5s}' for k in keys) + ' | W2 vs unif: decr [IQR]  int>0 int<0 | coreset vs W2: decr  int>0'
              + (' | ARI unif  w2  core' if study == 'valley' else '') + ' | W2^2 best/id  t_fit')
        for c in cells:
            S = [r for r in runs if tuple(r[k] for k in keys) == c]
            rw = np.array([_rel(r['eval']['voi_unif'], r['eval']['voi_w2']) for r in S])
            iw = sum(_lo(r['eval']['voi_unif'], r['eval']['voi_w2']) > 0 for r in S)
            iwn = sum(_lo(r['eval']['voi_w2'], r['eval']['voi_unif']) > 0 for r in S)
            rc = np.array([_rel(r['eval']['voi_w2'], r['voi_core']) for r in S])
            ic = sum(_lo(r['eval']['voi_w2'], r['voi_core']) > 0 for r in S)
            line = (' '.join(f'{x:5g}' for x in c) + f' | {np.median(rw):6.1f}% [{np.percentile(rw, 25):4.0f},'
                    f'{np.percentile(rw, 75):4.0f}] {iw:3d}/{len(S):<3d} {iwn:3d} | {np.median(rc):6.1f}% {ic:3d}/{len(S):<3d}')
            if study == 'valley':
                line += ' | ' + ' '.join(
                    f'{np.mean([np.mean(v) for v in vals]):.3f}' for vals in (
                        [r['eval']['ari_unif'] for r in S], [r['eval']['ari_w2'] for r in S], [r['ari_core'] for r in S]))
            ratio = [r['w2_log']['w2sq_best'] / r['w2_log']['w2sq_identity'] for r in S]
            line += f' | {np.median(ratio):.3f}  {np.median([r["t_fit"] for r in S]):4.1f}s'
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
