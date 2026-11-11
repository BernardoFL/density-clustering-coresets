"""
Valley-separated simulation (main text, Section 4.2 and Table 1).

Gaussian mixtures in d = 8 (K = 10 components, two rare components of prevalence p_min) with
mean separation Delta in {2, 3}, p_min in {0.05, 0.01} and coreset size n in {50, 100, 250};
20 replicates per setting.  Each replicate builds the coreset (coreset.algorithm.
build_coreset_selected) and compares it with uniform subsampling on the same support over
V_eval = 50 fresh evaluation draws.

Usage:  python experiments/exp_sim_valleys.py --reps 0-19 --n_jobs 16
        python experiments/exp_sim_valleys.py --summarise
"""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false'}.items():
    os.environ.setdefault(k, v)
import argparse, time
import numpy as np
from experiments.runlog import save_json, load_runs, parse_range

OUT = os.path.join(os.path.dirname(__file__), 'results', 'sim_valleys')
SEPS, PMINS, NS = (2.0, 3.0), (0.05, 0.01), (50, 100, 250)
CFG = dict(N=20000, d=8, K=10, sd_lo=0.5, sd_hi=1.5, N_r=2000, niter=40, L=500, nval=100, n_steps=500, V_eval=50)
MASTER = 8080


def generate(sep, p_min, rng):
    N, d, K = CFG['N'], CFG['d'], CFG['K']
    mu = rng.normal(scale=sep, size=(K, d)); sd = rng.uniform(CFG['sd_lo'], CFG['sd_hi'], size=(K, d))
    w = np.full(K, (1 - 2 * p_min) / (K - 2)); w[:2] = p_min
    cnt = rng.multinomial(N, w)
    X = np.vstack([mu[k] + sd[k] * rng.standard_normal((c, d)) for k, c in enumerate(cnt)])
    lab = np.repeat(np.arange(K), cnt)
    perm = rng.permutation(N)
    return X[perm], lab[perm], mu, sd, w


def valley_ratios(mu, sd, w):
    from scipy.special import logsumexp
    def ld(P):
        lp = -0.5 * (((P[:, None, :] - mu[None]) / sd[None]) ** 2).sum(-1) - np.log(sd).sum(1)[None] + np.log(w)[None]
        return logsumexp(lp, axis=1)
    t = np.linspace(0, 1, 101)[:, None]; out = []
    for k in range(len(mu)):
        j = np.argsort(np.linalg.norm(mu - mu[k], axis=1))[1]
        s = ld(mu[k] + t * (mu[j] - mu[k])); out.append(float(np.exp(s.min() - min(s[0], s[-1]))))
    return out


def run(sep, p_min, n, rep, cfg=CFG, K_cl=None, urn_reinforce=True, alpha=1.0):
    """K_cl: clustering resolution (default cfg['K'], the number of generating components)."""
    from coreset.algorithm import build_coreset_selected
    from coreset.polya_urn import polya_urn
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    from coreset.distances import voi_ref
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from sklearn.metrics import adjusted_rand_score as ari
    cell = SEPS.index(sep) * 100 + PMINS.index(p_min) * 10 + NS.index(n)
    ss = np.random.SeedSequence([MASTER, cell, rep])
    s_data, s_core, s_eval = ss.spawn(3)
    X, lab, mu, sd, w = generate(sep, p_min, np.random.default_rng(s_data))
    CFG = cfg
    ms = cfg.get('clust_min_share', 0.0)
    K_cl = CFG['K'] if K_cl is None else K_cl
    t0 = time.perf_counter()
    cs = build_coreset_selected(
        data=X, n=n, niter=CFG['niter'], L=CFG['L'],
        sample_hyperprior=lambda g: sample_niw_hyperprior(X, g),
        build_base_measure=lambda p, g: build_gaussian_base(*p, g),
        N_r=CFG['N_r'], K_cl=K_cl, alpha=alpha, nval=CFG['nval'], n_steps=CFG['n_steps'], seed=s_core, min_share=ms,
        urn_reinforce=urn_reinforce)
    t_build = time.perf_counter() - t0
    ref = np.asarray(cs['ref_idx']); R = X[ref]
    XA = X[np.setdiff1d(np.arange(len(X)), ref)]
    y = cs['y_star']
    A, b = np.asarray(cs['A_out']), np.asarray(cs['b_out'])
    rng = np.random.default_rng(s_eval)
    ev = {k: [] for k in ('voi_unif', 'voi_core', 'ari_full', 'ari_unif', 'ari_core')}
    for _ in range(CFG['V_eval']):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        tN = polya_urn(XA, base, alpha=alpha, L=CFG['L'], rng=rng, reinforce=urn_reinforce)
        tn = polya_urn(y, base, alpha=alpha, L=CFG['L'], rng=rng, reinforce=urn_reinforce)
        mf = fit_clust(tN, K_cl, min_share=ms); hp = hp_from_model(mf, min_share=ms)
        sN = hard_ref_labels(hp, R, tN, mf.labels, int(mf.labels.max() + 1))
        lu, mu_ = clust_shared(hp, tn); su = hard_ref_labels(hp, R, tn, lu, mu_)
        tc = tn @ A.T + b
        lc, mc = clust_shared(hp, tc); sc = hard_ref_labels(hp, R, tc, lc, mc)
        ev['voi_unif'].append(voi_ref(sN, su)); ev['voi_core'].append(voi_ref(sN, sc))
        ev['ari_full'].append(ari(lab[ref], sN)); ev['ari_unif'].append(ari(lab[ref], su)); ev['ari_core'].append(ari(lab[ref], sc))
    return {'sep': sep, 'p_min': p_min, 'n': n, 'rep': rep, 'valley_ratio': valley_ratios(mu, sd, w),
            'lambda_out': cs['lambda_out'], 'J_hat_I': cs['J_hat_I'], 'J_hat_bar': cs['J_hat_bar'],
            'V_I': cs['V_I'], 'V_bar': cs['V_bar'], 'eval': ev, 't_build': t_build, 'cfg': CFG}


def _worker(a):
    sep, p_min, n, rep, cfg, out = a
    path = os.path.join(out, f's{sep:g}_p{p_min:g}_n{n:04d}_rep{rep:02d}.json')
    if not os.path.exists(path):
        save_json(path, run(sep, p_min, n, rep, cfg))
    return path


def summarise(OUT=OUT):
    from scipy import stats
    from scipy.stats import binomtest
    runs = load_runs(OUT, 's*_rep*.json')
    print(f'Valley-separated simulation, {len(runs)} replicates (fresh evaluation draws, V = {CFG["V_eval"]})\n')
    print(f'{"sep":>4s} {"p_min":>5s} {"n":>4s} {"reps":>4s} | {"VoI_ref decrease":>18s} {"int>0":>6s} | '
          f'{"ARI truth: full  uniform  coreset":>34s} | {"core−unif":>9s} {"W/L":>6s} {"p":>5s} | λ*>0  time')
    for sep, p, n in itertools.product(SEPS, PMINS, NS):
        S = [r for r in runs if r['sep'] == sep and r['p_min'] == p and r['n'] == n]
        if not S:
            continue
        vu = np.array([np.mean(r['eval']['voi_unif']) for r in S]); vc = np.array([np.mean(r['eval']['voi_core']) for r in S])
        rel = 100 * (vu - vc) / vu
        lo = []
        for r in S:
            dd = np.asarray(r['eval']['voi_unif']) - np.asarray(r['eval']['voi_core'])
            lo.append(dd.mean() - stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if dd.std() > 0 else 0)
        af = np.mean([np.mean(r['eval']['ari_full']) for r in S])
        au = np.array([np.mean(r['eval']['ari_unif']) for r in S]); ac = np.array([np.mean(r['eval']['ari_core']) for r in S])
        da = ac - au; nz = da[da != 0]
        pv = binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
        print(f'{sep:4g} {p:5g} {n:4d} {len(S):4d} | {np.median(rel):6.1f}% [{np.percentile(rel, 25):3.0f},{np.percentile(rel, 75):3.0f}] '
              f'{sum(l > 0 for l in lo):3d}/{len(S):<2d} | {af:10.3f} {au.mean():8.3f} {ac.mean():8.3f} | '
              f'{da.mean():+9.3f} {int((da > 0).sum()):2d}/{int((da < 0).sum()):<3d} {pv:5.2f} | '
              f'{sum(r["lambda_out"] > 0 for r in S):2d}/{len(S):<2d} {np.median([r["t_build"] for r in S]):4.0f}s')
    vr = np.concatenate([r['valley_ratio'] for r in runs])
    print(f'\nvalley ratio to nearest cluster: median {np.median(vr):.3f}; share < 0.5: {np.mean(vr < 0.5):.0%}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', default='0-19'); p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--summarise', action='store_true')
    p.add_argument('--out', default=OUT)
    p.add_argument('--cells', default=None, help='setting indices 0-11 (sep x p_min x n order); default all')
    p.add_argument('--clust_min_share', type=float, default=0.0,
                   help='Clust: components below this share of a trajectory are sub-threshold')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summarise:
        return summarise(a.out)
    cfg = dict(CFG, clust_min_share=a.clust_min_share)
    cells = list(itertools.product(SEPS, PMINS, NS))
    cells = [cells[i] for i in parse_range(a.cells)] if a.cells else cells
    jobs = [(s, pm, n, r, cfg, a.out) for r in parse_range(a.reps) for s, pm, n in cells]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.n_jobs) as pool:
        for path in pool.imap_unordered(_worker, jobs):
            print(os.path.basename(path), flush=True)
    summarise(a.out)


if __name__ == '__main__':
    main()
