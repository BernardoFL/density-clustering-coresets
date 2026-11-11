"""
Convergence of the uniform subsample (Supplement, "Convergence of the uniform subsample";
figure fig_valleys).  Predictive level-set clustering on simulated Gaussian mixtures
(d = 8, K = 10, N = 20,000) with mean separation sd in {2, 3, 4} and rare-cluster prevalence
p_min in {0.05, 0.01}.

Valley depth: for each true cluster, its nearest other cluster (by mean); ratio = min of the TRUE
mixture density on the segment between the two means / the smaller density at the two means
(ratio < 1 = a valley; small = deep valley).
Predictive clustering: Pólya-urn trajectory (L = 500) from a uniform subsample of size n, or from
all non-reference points ('full') -> fit_clust (K_cl = 10) -> labels of a 2000-point reference set;
ARI vs the true labels (−1 kept as its own label), share assigned, clusters found.
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

SEPS = (2.0, 3.0, 4.0)
PMINS = (0.05, 0.01)
NS = (50, 112, 253, 570, 1281, 2883, 'full')
REPS, V = 10, 30
CFG = dict(N=20_000, d=8, K=10, sd_lo=0.5, sd_hi=1.5)
OUT = os.path.join(os.path.dirname(__file__), 'results', 'uniform_convergence.json')


def true_logdens(P, w, mu, sd):
    from scipy.special import logsumexp
    lp = -0.5 * (((P[:, None, :] - mu[None]) / sd[None]) ** 2).sum(-1) - np.log(sd).sum(1)[None] + np.log(w)[None]
    return logsumexp(lp, axis=1)


def one(args):
    sep, p_min, rep = args
    from coreset.polya_urn import polya_urn
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, hard_ref_labels
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from sklearn.metrics import adjusted_rand_score as ari
    cfg = CFG
    rng = np.random.default_rng([161, int(sep * 10), int(p_min * 1000), rep])
    # generate, keeping the true parameters
    N, d, K = cfg['N'], cfg['d'], cfg['K']
    mu = rng.normal(scale=sep, size=(K, d)); sd = rng.uniform(cfg['sd_lo'], cfg['sd_hi'], size=(K, d))
    w = np.full(K, (1 - 2 * p_min) / (K - 2)); w[:2] = p_min
    cnt = rng.multinomial(N, w)
    X = np.vstack([mu[k] + sd[k] * rng.standard_normal((c, d)) for k, c in enumerate(cnt)])
    lab = np.repeat(np.arange(K), cnt)
    perm = rng.permutation(N); X, lab = X[perm], lab[perm]
    # valley depth to the nearest cluster
    ratios = []
    t = np.linspace(0, 1, 101)[:, None]
    for k in range(K):
        j = np.argsort(np.linalg.norm(mu - mu[k], axis=1))[1]
        seg = true_logdens(mu[k] + t * (mu[j] - mu[k]), w, mu, sd)
        ratios.append(float(np.exp(seg.min() - min(seg[0], seg[-1]))))
    R, rest = np.arange(2000), np.arange(2000, N)
    out = {'sep': sep, 'p_min': p_min, 'rep': rep, 'valley_ratio': ratios}
    for n in NS:
        src = X[rest] if n == 'full' else X[rest[:n]]
        rows = []
        for _ in range(V):
            base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
            tr = polya_urn(src, base, alpha=1.0, L=500, rng=rng)
            m = fit_clust(tr, K)
            s = hard_ref_labels(hp_from_model(m), X[R], tr, m.labels, int(m.labels.max() + 1))
            rows.append((ari(lab[R], s), np.mean(s >= 0), len(np.unique(s[s >= 0]))))
        out[str(n)] = np.mean(rows, axis=0).tolist()
    return out


if __name__ == '__main__':
    import multiprocessing as mp
    jobs = [(s, p, r) for s in SEPS for p in PMINS for r in range(REPS)]
    with mp.get_context('spawn').Pool(12) as pool:
        res = pool.map(one, jobs)
    json.dump(res, open(OUT, 'w'))
    for s, p in itertools.product(SEPS, PMINS):
        S = [x for x in res if x['sep'] == s and x['p_min'] == p]
        vr = np.concatenate([x['valley_ratio'] for x in S])
        print(f'\nseparation {s}, p_min {p}: valley ratio to nearest cluster median {np.median(vr):.2f} '
              f'(share of clusters with a valley ratio < 0.5: {np.mean(vr < 0.5):.0%})')
        print(f'  {"n":>6s} | ARI vs truth | share assigned | clusters found')
        for n in NS:
            a = np.array([x[str(n)] for x in S])
            print(f'  {str(n):>6s} | {a[:, 0].mean():.3f} ± {a[:, 0].std():.3f} | {a[:, 1].mean():14.3f} | {a[:, 2].mean():6.1f}')
