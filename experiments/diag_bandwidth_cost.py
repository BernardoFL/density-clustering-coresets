"""
Diagnostics (Supplement, "Implementation details"): selected KDE bandwidth and the cost of one predictive
clustering.

For replicates 0-9 of each valley setting (sep, p_min) and of Levine_13dim, draws V_DIAG fresh
full-data trajectories (own random stream) and records
  - the selected bandwidth h (scaled coordinates) and its ratio to the Silverman starting value,
  - the wall-clock time of the urn trajectory and of the bandwidth/threshold search (Clust),
  - for comparison, the time of a coreset trajectory at n = 100 and of its clustering under the
    shared hyperparameters.
Run single-process on an otherwise idle machine.

Usage:  python experiments/diag_bandwidth_cost.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'diag_bandwidth_cost')
V_DIAG, REPS, N_CORE, TAG = 5, range(10), 100, 17


def silverman(d, L):
    return (4 / (d + 2)) ** (1 / (d + 4)) * L ** (-1 / (d + 4))


def one(X, XA, R, K_cl, alpha, L, seed):
    from coreset.polya_urn import polya_urn
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    rng = np.random.default_rng(seed)
    y = XA[rng.choice(len(XA), N_CORE, replace=False)]
    out = []
    for _ in range(V_DIAG):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        t0 = time.perf_counter(); tN = polya_urn(XA, base, alpha=alpha, L=L, rng=rng, reinforce=True)
        t1 = time.perf_counter(); mf = fit_clust(tN, K_cl); hp = hp_from_model(mf)
        hard_ref_labels(hp, R, tN, mf.labels, int(mf.labels.max() + 1) if (mf.labels >= 0).any() else 0)
        t2 = time.perf_counter(); tn = polya_urn(y, base, alpha=alpha, L=L, rng=rng, reinforce=True)
        t3 = time.perf_counter(); lab, m = clust_shared(hp, tn); hard_ref_labels(hp, R, tn, lab, m)
        t4 = time.perf_counter()
        out.append({'h': hp['h'], 'h_ratio': hp['h'] / silverman(X.shape[1], L),
                    't_urn_full': t1 - t0, 't_clust_full': t2 - t1, 't_urn_core': t3 - t2, 't_clust_core': t4 - t3})
    return out


def main():
    from experiments.exp_sim_valleys import generate, CFG, SEPS, PMINS, NS, MASTER
    from experiments.exp_levine13 import half
    from experiments.exp_levine13 import MASTER as LMASTER
    rows = []
    for sep in SEPS:
        for p_min in PMINS:
            cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(100)
            for rep in REPS:
                s_data, s_core, _ = np.random.SeedSequence([MASTER, cell, rep]).spawn(3)
                X, *_ = generate(sep, p_min, np.random.default_rng(s_data))
                perm = np.random.default_rng(s_core.spawn(5)[0]).permutation(len(X))
                R, XA = X[np.sort(perm[:CFG['N_r']])], X[np.sort(perm[CFG['N_r']:])]
                for d in one(X, XA, R, CFG['K'], 1.0, CFG['L'], np.random.SeedSequence([MASTER, cell, rep, TAG])):
                    rows.append({'study': 'valley', 'sep': sep, 'p_min': p_min, 'rep': rep, 'N_urn': len(XA), **d})
                print('valley', sep, p_min, rep, flush=True)
    cfg = json.load(open(os.path.join(RES, 'levine13_test', 'test_n0100_rep00.json')))['config']
    X, _, _ = half('test', cfg)
    for rep in REPS:
        perm = np.random.default_rng(np.random.SeedSequence([LMASTER, rep, TAG, 0])).permutation(len(X))
        R, XA = X[np.sort(perm[:cfg['N_r']])], X[np.sort(perm[cfg['N_r']:])]
        for d in one(X, XA, R, cfg['K_cl'], cfg['alpha'], cfg['L'], np.random.SeedSequence([LMASTER, rep, TAG, 1])):
            rows.append({'study': 'levine', 'rep': rep, 'N_urn': len(XA), **d})
        print('levine', rep, flush=True)
    save_json(os.path.join(OUT, 'rows.json'), rows)
    for study in ('valley', 'levine'):
        S = [r for r in rows if r['study'] == study]
        q = lambda k: np.percentile([r[k] for r in S], [0, 25, 50, 75, 100])
        print(f'\n{study}: {len(S)} draws, N_urn = {S[0]["N_urn"]}')
        for k in ('h', 'h_ratio', 't_urn_full', 't_clust_full', 't_urn_core', 't_clust_core'):
            print(f'  {k:13s} min/q1/med/q3/max = ' + ' '.join(f'{v:.3g}' for v in q(k)))
        if study == 'valley':
            for sep in SEPS:
                for p_min in PMINS:
                    hs = [r['h_ratio'] for r in S if r['sep'] == sep and r['p_min'] == p_min]
                    print(f'    sep={sep} p={p_min}: h/h_S median {np.median(hs):.2f} range [{min(hs):.2f}, {max(hs):.2f}]')


if __name__ == '__main__':
    main()
