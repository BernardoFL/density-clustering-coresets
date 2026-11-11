"""
Full-data replication loss and equivalent uniform size (Supplement, "Absolute losses, full-data
replication and equivalent uniform size"; uniform-curve table of "Convergence of the uniform subsample").

Every reported replicate (valley simulation, Levine_13dim) is replayed from its stored seeds, so
the full-data partitions of the
evaluation draws are exactly those behind Tables 1-3 (the uniform loss is checked).  On each
evaluation draw we add
  * the full-data replication loss: VoI_ref between the full-data partition and that of a second,
    independent full-data trajectory under the same phi, clustered with the first trajectory's
    bandwidth and threshold (as the coreset is);
  * the uniform curve: VoI_ref of nested uniform subsamples of the analysis set at a grid of sizes.
Extra randomness uses its own streams (spawn key ..., TAG).

Usage:  python experiments/exp_supp_replication.py --study valley --n_jobs 14
        python experiments/exp_supp_replication.py --summarise
"""
import os, sys, glob, json, itertools, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'supp_replication')
TAG = 11
GRID = {'valley': (50, 100, 250, 500, 1000, 2000, 4000),
        'levine': (50, 100, 250, 400, 1000, 2000, 4000, 8000)}


def _labels(tN, trajs, ref, K_cl):
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    mf = fit_clust(tN, K_cl); hp = hp_from_model(mf)
    sN = hard_ref_labels(hp, ref, tN, mf.labels, int(mf.labels.max() + 1) if (mf.labels >= 0).any() else 0)
    out = []
    for t in trajs:
        lab, m = clust_shared(hp, t)
        out.append(hard_ref_labels(hp, ref, t, lab, m))
    return sN, out


def _extra(X_A, y_grid, phi, rng_x, alpha, L):
    """Second full-data trajectory and the uniform-curve trajectories for one draw (own stream)."""
    from coreset.polya_urn import polya_urn
    from inference.hyperpriors import build_gaussian_base
    base = build_gaussian_base(*phi, rng_x)
    tN2 = polya_urn(X_A, base, alpha=alpha, L=L, rng=rng_x, reinforce=True)
    return [tN2] + [polya_urn(Y, base, alpha=alpha, L=L, rng=rng_x, reinforce=True) for Y in y_grid]


def run_valley(sep, p_min, n, rep):
    from experiments.exp_sim_valleys import generate, CFG, SEPS, PMINS, NS, MASTER
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(n)
    s_data, s_core, s_eval = np.random.SeedSequence([MASTER, cell, rep]).spawn(3)
    X, lab, *_ = generate(sep, p_min, np.random.default_rng(s_data))
    ss_split, ss_support, *_ = s_core.spawn(5)
    perm = np.random.default_rng(ss_split).permutation(len(X))
    ref_idx, A_idx = np.sort(perm[:CFG['N_r']]), np.sort(perm[CFG['N_r']:])
    support_idx = np.sort(np.random.default_rng(ss_support).choice(A_idx, size=n, replace=False))
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]
    nest = np.random.default_rng(np.random.SeedSequence([MASTER, cell, rep, TAG, 0])).permutation(len(XA))
    y_grid = [XA[nest[:m]] for m in GRID['valley']]
    stored = json.load(open(os.path.join(RES, 'sim_valleys', f's{sep:g}_p{p_min:g}_n{n:04d}_rep{rep:02d}.json')))
    rng = np.random.default_rng(s_eval)
    rng_x = np.random.default_rng(np.random.SeedSequence([MASTER, cell, rep, TAG, 1]))
    ev = {'voi_unif': [], 'voi_repl': [], 'voi_grid': []}
    for _ in range(CFG['V_eval']):
        phi = sample_niw_hyperprior(X, rng); base = build_gaussian_base(*phi, rng)
        tN = polya_urn(XA, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=1.0, L=CFG['L'], rng=rng, reinforce=True)
        sN, labs = _labels(tN, [tn] + _extra(XA, y_grid, phi, rng_x, 1.0, CFG['L']), R, CFG['K'])
        ev['voi_unif'].append(voi_ref(sN, labs[0])); ev['voi_repl'].append(voi_ref(sN, labs[1]))
        ev['voi_grid'].append([voi_ref(sN, s) for s in labs[2:]])
    return {'study': 'valley', 'sep': sep, 'p_min': p_min, 'n': n, 'rep': rep, 'grid': GRID['valley'], 'eval': ev,
            'voi_core': stored['eval']['voi_core'],
            'reproduction_max_abs_diff': float(np.max(np.abs(np.subtract(ev['voi_unif'], stored['eval']['voi_unif']))))}


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
    assert np.array_equal(support_idx, stored['coreset']['support_idx'])
    R, XA, y = X[ref_idx], X[A_idx], X[support_idx]
    nest = np.random.default_rng(np.random.SeedSequence([MASTER, rep, n, TAG, 0])).permutation(len(XA))
    y_grid = [XA[nest[:m]] for m in GRID['levine']]
    rng = np.random.default_rng(ss_select)
    nsel = cfg['nval'] // 2
    for _ in range(nsel):                                   # replay the selection draws' stream
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
    rng_x = np.random.default_rng(np.random.SeedSequence([MASTER, rep, n, TAG, 1]))
    ev = {'voi_unif': [], 'voi_repl': [], 'voi_grid': []}
    for _ in range(cfg['nval'] - nsel):
        phi = sample_niw_hyperprior(X, rng); base = build_gaussian_base(*phi, rng)
        tN = polya_urn(XA, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        tn = polya_urn(y, base, alpha=cfg['alpha'], L=cfg['L'], rng=rng, reinforce=True)
        sN, labs = _labels(tN, [tn] + _extra(XA, y_grid, phi, rng_x, cfg['alpha'], cfg['L']), R, cfg['K_cl'])
        ev['voi_unif'].append(voi_ref(sN, labs[0])); ev['voi_repl'].append(voi_ref(sN, labs[1]))
        ev['voi_grid'].append([voi_ref(sN, s) for s in labs[2:]])
    return {'study': 'levine', 'n': n, 'rep': rep, 'grid': GRID['levine'], 'eval': ev,
            'voi_core': stored['coreset']['V_bar'],
            'reproduction_max_abs_diff': float(np.max(np.abs(np.subtract(ev['voi_unif'], stored['coreset']['V_I']))))}


def _path(job):
    if job[0] == 'valley':
        _, sep, p, n, rep = job
        return os.path.join(OUT, 'valley', f's{sep:g}_p{p:g}_n{n:04d}_rep{rep:02d}.json')
    return os.path.join(OUT, 'levine', f'n{job[1]:04d}_rep{job[2]:02d}.json')


def _worker(job):
    path = _path(job)
    if os.path.exists(path):
        return path, 'skip'
    t0 = time.perf_counter()
    res = run_valley(*job[1:]) if job[0] == 'valley' else run_levine(*job[1:])
    res['t_total'] = time.perf_counter() - t0
    save_json(path, res)
    return path, f"{res['t_total']:.0f}s check={res['reproduction_max_abs_diff']:.2g}"


def jobs(study, reps=None):
    if study == 'valley':
        from experiments.exp_sim_valleys import SEPS, PMINS, NS
        return [('valley', s, p, n, r) for s, p, n in itertools.product(SEPS, PMINS, NS) for r in (reps or range(20))]
    return [('levine', n, r) for n in (50, 100, 150, 250, 400, 1000) for r in (reps or range(30))]


def equivalent_size(grid, curve, target):
    """Size at which the uniform curve (interpolated linearly in log size) reaches `target`;
    returns (value, flag) with flag '<', '>' or '' when outside the grid."""
    g, c = np.log(np.asarray(grid, float)), np.asarray(curve, float)
    c = np.minimum.accumulate(c)                          # monotone envelope of the noisy curve
    if target >= c[0]:
        return float(grid[0]), '<'
    if target < c[-1]:
        return float(grid[-1]), '>'
    i = int(np.argmax(c <= target))
    t = (c[i - 1] - target) / max(c[i - 1] - c[i], 1e-12)
    return float(np.exp(g[i - 1] + t * (g[i] - g[i - 1]))), ''


def summarise():
    for study in ('valley', 'levine'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, study, '*.json')))]
        if not runs:
            continue
        print(f'\n{study}: {len(runs)} replicates; max |uniform loss - reported| = '
              f'{max(r["reproduction_max_abs_diff"] for r in runs):.2g}')
        keys = ('sep', 'p_min', 'n') if study == 'valley' else ('n',)
        cells = sorted({tuple(r[k] for k in keys) for r in runs}, key=lambda c: (-c[0], -c[1], c[2]) if len(c) == 3 else c)
        print(' '.join(f'{k:>5s}' for k in keys) + ' |  J_unif  J_core  J_repl | excess removed [IQR] | n_eq median [IQR] (>grid)')
        for c in cells:
            S = [r for r in runs if tuple(r[k] for k in keys) == c]
            ju = np.array([np.mean(r['eval']['voi_unif']) for r in S]); jc = np.array([np.mean(r['voi_core']) for r in S])
            jr = np.array([np.mean(r['eval']['voi_repl']) for r in S])
            fr = 100 * (ju - jc) / (ju - jr)
            ne = [equivalent_size(r['grid'], np.mean(r['eval']['voi_grid'], axis=0), np.mean(r['voi_core'])) for r in S]
            nv = np.array([v for v, _ in ne]); over = sum(f == '>' for _, f in ne)
            print(' '.join(f'{x:5g}' for x in c) + f' | {np.median(ju):7.2f} {np.median(jc):7.2f} {np.median(jr):7.2f} | '
                  f'{np.median(fr):5.0f}% [{np.percentile(fr, 25):3.0f},{np.percentile(fr, 75):3.0f}] | '
                  f'{np.median(nv):6.0f} [{np.percentile(nv, 25):5.0f},{np.percentile(nv, 75):5.0f}] ({over})')
        G = runs[0]['grid']
        print('  uniform curve (median over replicates of the per-replicate mean):',
              {g: round(float(np.median([np.mean(r['eval']['voi_grid'], axis=0)[i] for r in runs])), 2) for i, g in enumerate(G)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', choices=['valley', 'levine'])
    p.add_argument('--reps', default=None)
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    if a.summarise:
        return summarise()
    from experiments.runlog import parse_range
    J = jobs(a.study, parse_range(a.reps) if a.reps else None)
    from multiprocessing import Pool
    with Pool(a.n_jobs) as pool:
        for out in pool.imap_unordered(_worker, J):
            print(*out, flush=True)


if __name__ == '__main__':
    main()
