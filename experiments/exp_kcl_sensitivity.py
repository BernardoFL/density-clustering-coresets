"""
Sensitivity to the clustering resolution K_cl (Supplement, "Sensitivity to the clustering resolution").

Rebuilds coresets of reported replicates (same data, reference split and support: same seeds) at
other values of K_cl.  The full-data
reference partition is re-derived at each K_cl, so every K_cl is assessed against its own target.
For each replicate we also estimate the full-data replication loss at that K_cl (50 draws).

  valley: all (Delta, p_min) at n = 100, K_cl in {5, 15, 20} (reported: 10), reps 0-19
  levine: n in {100, 400}, K_cl in {12, 36} (reported: 24), reps 0-19

Usage:  python experiments/exp_kcl_sensitivity.py --n_jobs 14
        python experiments/exp_kcl_sensitivity.py --summarise
"""
import os, sys, glob, json, itertools, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'kcl_sensitivity')
TAG = 13
V_REPL = 50
VALLEY_K, LEVINE_K = (5, 15, 20), (12, 36)


def replication_loss(X, XA, R, K_cl, seed, alpha, L):
    """Mean VoI_ref between two independent full-data partitions under a common phi (V_REPL draws)."""
    from coreset.polya_urn import polya_urn
    from coreset.distances import voi_ref
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    rng = np.random.default_rng(seed); out = []
    for _ in range(V_REPL):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        t1 = polya_urn(XA, base, alpha=alpha, L=L, rng=rng, reinforce=True)
        t2 = polya_urn(XA, base, alpha=alpha, L=L, rng=rng, reinforce=True)
        mf = fit_clust(t1, K_cl); hp = hp_from_model(mf)
        s1 = hard_ref_labels(hp, R, t1, mf.labels, int(mf.labels.max() + 1) if (mf.labels >= 0).any() else 0)
        lab, m = clust_shared(hp, t2)
        out.append(voi_ref(s1, hard_ref_labels(hp, R, t2, lab, m)))
    return out


def run_valley(sep, p_min, K_cl, rep, n=100):
    from experiments.exp_sim_valleys import run, generate, CFG, SEPS, PMINS, NS, MASTER
    r = run(sep, p_min, n, rep, K_cl=K_cl, urn_reinforce=True)
    cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(n)
    s_data, s_core, _ = np.random.SeedSequence([MASTER, cell, rep]).spawn(3)
    X, *_ = generate(sep, p_min, np.random.default_rng(s_data))
    ss_split = s_core.spawn(5)[0]
    perm = np.random.default_rng(ss_split).permutation(len(X))
    R, XA = X[np.sort(perm[:CFG['N_r']])], X[np.sort(perm[CFG['N_r']:])]
    repl = replication_loss(X, XA, R, K_cl, np.random.SeedSequence([MASTER, cell, rep, TAG, K_cl]), 1.0, CFG['L'])
    return {'study': 'valley', 'sep': sep, 'p_min': p_min, 'n': n, 'K_cl': K_cl, 'rep': rep,
            'eval': r['eval'], 'lambda_out': r['lambda_out'], 't_build': r['t_build'], 'voi_repl': repl}


def run_levine(n, K_cl, rep):
    from experiments.exp_levine13 import half, MASTER, SEED_NAMES
    from experiments.runlog import child_seeds
    from coreset.algorithm import build_coreset_selected
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    cfg = json.load(open(os.path.join(RES, 'levine13_test', f'test_n{n:04d}_rep{rep:02d}.json')))['config']
    X, _, _ = half('test', cfg)
    t0 = time.perf_counter()
    cs = build_coreset_selected(
        data=X, n=n, niter=cfg['niter'], L=cfg['L'],
        sample_hyperprior=lambda g: sample_niw_hyperprior(X, g),
        build_base_measure=lambda p, g: build_gaussian_base(*p, g),
        N_r=cfg['N_r'], K_cl=K_cl, alpha=cfg['alpha'], nval=cfg['nval'], minsize=cfg['minsize'],
        lr=cfg['lr'], n_steps=cfg['n_steps'], w_merge=cfg.get('w_merge', 1.0),
        seed=child_seeds(MASTER, rep, SEED_NAMES)['coreset'], urn_reinforce=True)
    t_build = time.perf_counter() - t0
    ref = np.asarray(cs['ref_idx'])
    R, XA = X[ref], X[np.setdiff1d(np.arange(len(X)), ref)]
    repl = replication_loss(X, XA, R, K_cl, np.random.SeedSequence([MASTER, rep, n, TAG, K_cl]),
                            cfg['alpha'], cfg['L'])
    return {'study': 'levine', 'n': n, 'K_cl': K_cl, 'rep': rep, 'V_I': cs['V_I'], 'V_bar': cs['V_bar'],
            'lambda_out': cs['lambda_out'], 't_build': t_build, 'voi_repl': repl}


def _path(job):
    if job[0] == 'valley':
        _, sep, p, k, rep = job
        return os.path.join(OUT, 'valley', f's{sep:g}_p{p:g}_K{k:02d}_rep{rep:02d}.json')
    _, n, k, rep = job
    return os.path.join(OUT, 'levine', f'n{n:04d}_K{k:02d}_rep{rep:02d}.json')


def _worker(job):
    path = _path(job)
    if os.path.exists(path):
        return path, 'skip'
    res = run_valley(*job[1:]) if job[0] == 'valley' else run_levine(*job[1:])
    save_json(path, res)
    return path, f"{res['t_build']:.0f}s"


def jobs():
    from experiments.exp_sim_valleys import SEPS, PMINS
    J = [('valley', s, p, k, r) for s, p in itertools.product(SEPS, PMINS) for k in VALLEY_K for r in range(20)]
    J += [('levine', n, k, r) for n in (100, 400) for k in LEVINE_K for r in range(20)]
    return J


def _lo(a, b):
    from scipy import stats
    dd = np.asarray(a) - np.asarray(b)
    return dd.mean() - stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if dd.std() > 0 else 0.0


def summarise():
    """Rows for the new K_cl values; the reported K_cl rows come from the original results
    (sim_valleys, levine13_test) and supp_replication for their replication loss."""
    from experiments.exp_supp_replication import OUT as REPL
    rows = []
    for f in glob.glob(os.path.join(OUT, 'valley', '*.json')):
        r = json.load(open(f)); e = r['eval']
        rows.append(('valley', r['sep'], r['p_min'], r['K_cl'], r['rep'], e['voi_unif'], e['voi_core'],
                     r['voi_repl'], np.mean(e['ari_unif']), np.mean(e['ari_core']), r['lambda_out']))
    for f in glob.glob(os.path.join(RES, 'sim_valleys', 's*_n0100_rep*.json')):
        r = json.load(open(f)); e = r['eval']
        q = os.path.join(REPL, 'valley', os.path.basename(f))
        repl = json.load(open(q))['eval']['voi_repl'] if os.path.exists(q) else [np.nan]
        rows.append(('valley', r['sep'], r['p_min'], 10, r['rep'], e['voi_unif'], e['voi_core'], repl,
                     np.mean(e['ari_unif']), np.mean(e['ari_core']), r['lambda_out']))
    for f in glob.glob(os.path.join(OUT, 'levine', '*.json')):
        r = json.load(open(f))
        rows.append(('levine', r['n'], None, r['K_cl'], r['rep'], r['V_I'], r['V_bar'], r['voi_repl'],
                     np.nan, np.nan, r['lambda_out']))
    for n in (100, 400):
        for f in glob.glob(os.path.join(RES, 'levine13_test', f'test_n{n:04d}_rep*.json')):
            r = json.load(open(f))
            if r['rep'] >= 20:
                continue
            q = os.path.join(REPL, 'levine', f'n{n:04d}_rep{r["rep"]:02d}.json')
            repl = json.load(open(q))['eval']['voi_repl'] if os.path.exists(q) else [np.nan]
            rows.append(('levine', n, None, 24, r['rep'], r['coreset']['V_I'], r['coreset']['V_bar'], repl,
                         np.nan, np.nan, r['coreset']['lambda_out']))
    print('study  cell          K_cl | J_unif J_core J_repl | decrease [IQR] int>0 | excess removed | ARI unif/core | λ>0')
    for key in sorted({(r[0], r[1], r[2], r[3]) for r in rows}, key=lambda k: (k[0], -(k[1] or 0), -(k[2] or 0), k[3])):
        S = [r for r in rows if (r[0], r[1], r[2], r[3]) == key]
        ju = np.array([np.mean(r[5]) for r in S]); jc = np.array([np.mean(r[6]) for r in S])
        jr = np.array([np.mean(r[7]) for r in S]); dec = 100 * (ju - jc) / ju
        ex = 100 * (ju - jc) / (ju - jr); ip = sum(_lo(r[5], r[6]) > 0 for r in S)
        cell = f'{key[1]:g}/{key[2]:g}' if key[0] == 'valley' else f'n={key[1]}'
        print(f'{key[0]:6s} {cell:12s} {key[3]:5d} | {np.median(ju):6.2f} {np.median(jc):6.2f} {np.nanmedian(jr):6.2f} | '
              f'{np.median(dec):5.1f}% [{np.percentile(dec, 25):3.0f},{np.percentile(dec, 75):3.0f}] {ip:2d}/{len(S):<2d} | '
              f'{np.nanmedian(ex):5.0f}% | {np.nanmean([r[8] for r in S]):.2f}/{np.nanmean([r[9] for r in S]):.2f} | '
              f'{sum(r[10] > 0 for r in S)}/{len(S)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    if a.summarise:
        return summarise()
    from multiprocessing import Pool
    with Pool(a.n_jobs) as pool:
        for out in pool.imap_unordered(_worker, jobs()):
            print(*out, flush=True)


if __name__ == '__main__':
    main()
