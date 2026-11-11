"""
Levine_13dim mass cytometry (main text, Section 4.3 and Table 2).

167,044 cells, 13 markers, 24 expert-gated populations (Weber & Robinson 2016 benchmark).
Cells are split once into development / test halves (data/cytof.py); settings were tuned on
the development half and results are reported on the test half.  Within a half, the coreset
is built by build_coreset_selected (uniform support from y_A, soft level-set map, lambda-path
selection).  Arms, same support and same sampler seed:
    coreset  T_out(y*)          uniform  y*
Downstream: overfitted diagonal Normal-inverse-gamma mixture (K_max = 30, Dirichlet
alpha = 0.01) by blocked Gibbs (inference/gibbs_diag.py); point partition of the fitted points
= draws-based VoI-optimal partition; every held-out cell is assigned by one E-step with that
partition's posterior-mean parameters.
Full data: the same sampler on every cell of the half (reference and timing).

Held-out cells: the half minus the reference set and the support.  Metrics on the held-out
gated cells: ARI, macro-F1 (Hungarian matching of clusters to the 24 populations), number of
clusters holding >= 1% of these cells, Brier score of the posterior similarity matrix on
2,000 of them, ESS per second of the post-burn-in log-likelihood trace.

Usage
-----
  python experiments/exp_levine13.py --split test --ns 50,100,150,250,400,1000 --reps 0-29 --n_jobs 8
  python experiments/exp_levine13.py --split test --summarise
  python experiments/exp_levine13.py --split dev --summarise --blind   # development half, coreset arm hidden
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'CUDA_VISIBLE_DEVICES': '', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
             'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false', 'OMP_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)

import argparse
import json
import time

import numpy as np

from experiments.runlog import child_seeds, seed_int, save_json, load_runs, parse_range

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MASTER = 13
DATASET = 'Levine_13dim'
CONFIG = dict(
    niter=40, L=500, alpha=1.0, N_r=2000, nval=100, minsize=1, n_steps=500, lr=3e-3, K_cl=24,
    w_merge=1.0, K_max=30, dir_alpha=0.01, kappa0=0.01, a0=3.0, s_w=0.1,
    gibbs_iter=2000, gibbs_burn=1000, gibbs_thin=5, n_brier=2000, min_share=0.01,
)
SEED_NAMES = ['coreset', 'gibbs', 'brier']


def half(split, cfg=None):
    """
    Cells of one half.  cfg['dataset'] (default Levine_13dim); if cfg['pca_dim'] is set, the
    markers are projected on the first pca_dim principal components of the DEVELOPMENT half
    (fitted there only, then applied to either half).
    """
    from data.cytof import load_cytof, split_cells
    cfg = cfg or CONFIG
    v = load_cytof(cfg.get('dataset', DATASET))
    dev, test = split_cells(len(v['X']))
    idx = dev if split == 'dev' else test
    X = v['X'][idx]
    if cfg.get('pca_dim'):
        from sklearn.decomposition import PCA
        X = PCA(cfg['pca_dim'], random_state=0).fit(v['X'][dev]).transform(X)
    return X, v['labels'][idx], v['names']


def macro_f1(pred, truth, n_pop):
    from scipy.optimize import linear_sum_assignment
    ks = np.unique(pred)
    F = np.zeros((n_pop, len(ks)))
    for j, c in enumerate(ks):
        p = pred == c
        for k in range(n_pop):
            t = truth == k
            tp = np.sum(p & t)
            F[k, j] = 2 * tp / (p.sum() + t.sum()) if tp else 0.0
    r, c = linear_sum_assignment(-F)
    f1 = np.zeros(n_pop)
    f1[r] = F[r, c]
    return float(f1.mean()), f1


def n_big(labels, share):
    c = np.bincount(np.asarray(labels) - np.min(labels))
    return int(np.sum(c >= share * len(labels)))


def evaluate(fit, X_fit, X_eval, lab_eval, X_brier, same_brier, cfg, n_pop):
    from sklearn.metrics import adjusted_rand_score
    from inference.gibbs_diag import point_partition, project, ess, coclustering_rb
    zh = point_partition(fit['z'])
    modal = cfg.get('modal', True)          # fixed K: modal=False, each component is a cluster
    pred, params = project(X_eval, X_fit, zh, fit['prior'], modal=modal, return_params=True)
    mf1, f1 = macro_f1(pred, lab_eval, n_pop)
    P, n_cl_draws = coclustering_rb(X_brier, fit, modal=modal)
    iu = np.triu_indices(len(X_brier), 1)
    post = fit['loglik_trace'][fit['burn']:]
    return {
        'ari': float(adjusted_rand_score(lab_eval, pred)), 'macro_f1': mf1, 'f1': f1,
        'K_hat': n_big(pred, cfg['min_share']),
        'K_components_draws': [int(len(np.unique(z))) for z in fit['z']],
        'K_modes_draws': n_cl_draws,
        'n_components_point': int(len(params['w'])),
        'fit_params': params,                      # point-estimate mixture, for re-scoring without refitting
        'brier': float(np.mean((P[iu] - same_brier[iu]) ** 2)),
        'ess_per_s': ess(post) / (fit['time'] * (len(post) / len(fit['loglik_trace']))),
        'sampler_time': fit['time'],
    }


def _eval_sets(X, lab, exclude, rng, cfg):
    keep = np.setdiff1d(np.arange(len(X)), exclude)
    g = keep[lab[keep] >= 0]
    b = rng.choice(g, cfg['n_brier'], replace=False)
    same = (lab[b][:, None] == lab[b][None]).astype(float)
    return g, b, same


def run_replicate(split, n, rep, cfg, out_dir):
    from coreset.algorithm import build_coreset_selected
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from inference.gibbs_diag import gibbs_diag_mixture, default_prior

    X, lab, names = half(split, cfg)
    seeds = child_seeds(MASTER, rep, SEED_NAMES)
    t0 = time.perf_counter()
    cs = build_coreset_selected(
        data=X, n=n, niter=cfg['niter'], L=cfg['L'],
        sample_hyperprior=lambda g: sample_niw_hyperprior(X, g),
        build_base_measure=lambda p, g: build_gaussian_base(*p, g),
        N_r=cfg['N_r'], K_cl=cfg['K_cl'], alpha=cfg['alpha'], nval=cfg['nval'], minsize=cfg['minsize'],
        lr=cfg['lr'], n_steps=cfg['n_steps'], w_merge=cfg.get('w_merge', 1.0), seed=seeds['coreset'],
        min_share=cfg.get('clust_min_share', 0.0),
    )
    t_coreset = time.perf_counter() - t0
    g, b, same = _eval_sets(X, lab, np.concatenate([cs['ref_idx'], cs['support_idx']]),
                            np.random.default_rng(seeds['brier']), cfg)
    K_true = n_big(lab[g], cfg['min_share'])
    held = np.setdiff1d(np.arange(len(X)), np.concatenate([cs['ref_idx'], cs['support_idx']]))
    fp = os.path.join(out_dir, f'full_{split}_labels.npy')
    full_labels = np.load(fp) if os.path.exists(fp) else None
    from sklearn.metrics import adjusted_rand_score
    from coreset.distances import voi
    arms = {}
    for arm, Z in (('coreset', cs['y_coreset']), ('uniform', cs['y_star'])):
        pr = default_prior(Z, cfg['dir_alpha'], cfg['kappa0'], cfg['a0'], cfg['s_w'])
        fit = gibbs_diag_mixture(Z, cfg['K_max'], cfg['gibbs_iter'], cfg['gibbs_burn'], cfg['gibbs_thin'],
                                 prior=pr, seed=seed_int(seeds['gibbs']))
        arms[arm] = evaluate(fit, Z, X[g], lab[g], X[b], same, cfg, len(names))
        arms[arm]['K_bias'] = abs(arms[arm]['K_hat'] - K_true)
        if full_labels is not None:
            pred_h = assign(arms[arm]['fit_params'], X[held])
            arms[arm]['ari_full'] = float(np.mean([adjusted_rand_score(fl[held], pred_h) for fl in full_labels]))
            arms[arm]['voi_full'] = float(np.mean([voi(fl[held], pred_h) for fl in full_labels]))
    return {
        'split': split, 'n': n, 'rep': rep, 'master_seed': MASTER, 'seeds': seeds, 'config': cfg,
        'dataset': cfg.get('dataset', DATASET), 'K_true': K_true, 'arms': arms,
        'coreset': {k: cs[k] for k in ('lambda_out', 'select_curve', 'J_hat_bar', 'J_hat_I', 'V_bar', 'V_I',
                                       'A_bar', 'b_bar', 'support_idx', 'ref_idx', 'surrogate_log', 'config')},
        'timings': {'coreset_build': t_coreset, **{f'coreset_{k}': v for k, v in cs['timings'].items()},
                    **{f'gibbs_{a}': arms[a]['sampler_time'] for a in arms}},
    }


def assign(params, X):
    """Labels of X under a saved point-estimate mixture (component argmax, then its cluster map)."""
    from inference.gibbs_diag import _loglik_matrix
    w, mu, s2 = (np.asarray(params[k]) for k in ('w', 'mu', 's2'))
    out = np.concatenate([_loglik_matrix(X[i:i + 50000], w, mu, s2).argmax(1) for i in range(0, len(X), 50000)])
    cmap = params.get('component_to_cluster')
    return out if cmap is None else np.asarray(cmap)[out]


def _full_fit(args):
    split, cfg, seed = args
    from inference.gibbs_diag import gibbs_diag_mixture, default_prior
    X, lab, names = half(split, cfg)
    g, b, same = _eval_sets(X, lab, np.array([], int), np.random.default_rng(0), cfg)
    pr = default_prior(X, cfg['dir_alpha'], cfg['kappa0'], cfg['a0'], cfg['s_w'])
    fit = gibbs_diag_mixture(X, cfg['K_max'], cfg['gibbs_iter'], cfg['gibbs_burn'], cfg['gibbs_thin'], prior=pr, seed=seed)
    fit['z'] = fit['z'][::4]                     # 50 draws are enough for the point partition on N cells
    ev = evaluate(fit, X, X[g], lab[g], X[b], same, cfg, len(names))
    return seed, ev, assign(ev['fit_params'], X)


def full_reference(split, cfg, out_dir):
    """Full-data fits with n_full_seeds seeds: the reference partitions, their seed-to-seed variation, timing."""
    import itertools
    from sklearn.metrics import adjusted_rand_score
    from coreset.distances import voi
    path = os.path.join(out_dir, f'full_{split}.json')
    if os.path.exists(path):
        return json.load(open(path))
    X, lab, _ = half(split, cfg)
    g, _, _ = _eval_sets(X, lab, np.array([], int), np.random.default_rng(0), cfg)
    seeds = list(range(cfg.get('n_full_seeds', 5)))
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(len(seeds)) as pool:
        res = sorted(pool.map(_full_fit, [(split, cfg, s) for s in seeds]), key=lambda r: r[0])
    labels = np.stack([r[2] for r in res])
    np.save(os.path.join(out_dir, f'full_{split}_labels.npy'), labels)
    pairs = list(itertools.combinations(range(len(seeds)), 2))
    rec = {'split': split, 'N': len(X), 'K_true': n_big(lab[g], cfg['min_share']),
           'seeds': [{k: v for k, v in r[1].items() if k != 'fit_params'} for r in res],
           'floor_ari': [adjusted_rand_score(labels[i], labels[j]) for i, j in pairs],
           'floor_voi': [voi(labels[i], labels[j]) for i, j in pairs]}
    for k in ('ari', 'macro_f1', 'K_hat', 'brier', 'ess_per_s', 'sampler_time'):
        rec[k] = float(np.mean([r[1][k] for r in res]))
    save_json(path, rec)
    return json.load(open(path))


def _worker(args):
    split, n, rep, cfg, out_dir = args
    path = os.path.join(out_dir, f'{split}_n{n:04d}_rep{rep:02d}.json')
    if os.path.exists(path):
        return path
    t0 = time.perf_counter()
    rec = run_replicate(split, n, rep, cfg, out_dir)
    rec['timings']['total'] = time.perf_counter() - t0
    save_json(path, rec)
    return path


def summarise(out_dir, split, blind):
    runs = load_runs(out_dir, f'{split}_n*_rep*.json')
    arms = ['uniform'] if blind else ['coreset', 'uniform']
    fp = os.path.join(out_dir, f'full_{split}.json')
    if os.path.exists(fp):
        f = json.load(open(fp))
        print(f"full data (N={f['N']}): ARI {f['ari']:.3f}  macro-F1 {f['macro_f1']:.3f}  K̂ {f['K_hat']} (K_true {f['K_true']})"
              f"  Brier {f['brier']:.4f}  ESS/s {f['ess_per_s']:.2f}  sampler {f['sampler_time']:.0f}s")
    for n in sorted({r['n'] for r in runs}):
        S = [r for r in runs if r['n'] == n]
        print(f'n = {n} ({len(S)} reps)' + ('   [BLIND: coreset arm hidden]' if blind else ''))
        for a in arms:
            m = lambda k: np.array([r['arms'][a][k] for r in S])
            ari = m('ari')
            modes = ''
            if all('K_modes_draws' in r['arms'][a] for r in S):   # not recorded by older runs
                modes = f"modes/draw {np.median([np.median(r['arms'][a]['K_modes_draws']) for r in S]):.0f}  "
            print(f"  {a:8s} ARI {np.median(ari):.3f} [IQR {np.subtract(*np.percentile(ari, [75, 25])):.3f}]  "
                  f"macro-F1 {np.median(m('macro_f1')):.3f}  K̂ {np.median(m('K_hat')):.0f} (|K̂−K| {np.mean(m('K_bias')):.1f})  "
                  + modes +
                  f"Brier {np.median(m('brier')):.4f}  ESS/s {np.median(m('ess_per_s')):.1f}")
        print(f"  time per replicate {np.median([r['timings']['total'] for r in S]):.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'test'], required=True)
    p.add_argument('--ns', default='250')
    p.add_argument('--reps', default='0-29')
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--out', default=None, help='default: experiments/results/levine13_<split>')
    p.add_argument('--full', action='store_true')
    p.add_argument('--summarise', action='store_true')
    p.add_argument('--blind', action='store_true', help='hide the coreset arm (development tuning)')
    p.add_argument('--L', type=int, default=None, help='override trajectory length (development)')
    p.add_argument('--K_cl', type=int, default=None, help='override Clust resolution (development)')
    p.add_argument('--w_merge', type=float, default=None, help='weight on H(full | coreset) (development)')
    p.add_argument('--pca_dim', type=int, default=None, help='project on PCs fitted on the development half')
    p.add_argument('--dataset', default=None)
    p.add_argument('--fixed_K', type=int, default=None, help='fixed-K mixture (Dirichlet α = 1), components = clusters')
    p.add_argument('--clust_min_share', type=float, default=None,
                   help='Clust: components below this share of a trajectory are sub-threshold')
    a = p.parse_args()
    a.out = a.out or os.path.join(RESULTS_DIR, f'levine13_{a.split}')
    if a.clust_min_share is not None:
        CONFIG['clust_min_share'] = a.clust_min_share
    if a.L:
        CONFIG['L'] = a.L
    if a.K_cl:
        CONFIG['K_cl'] = a.K_cl
    if a.w_merge:
        CONFIG['w_merge'] = a.w_merge
    if a.pca_dim:
        CONFIG['pca_dim'] = a.pca_dim
    if a.dataset:
        CONFIG['dataset'] = a.dataset
    if a.fixed_K:
        CONFIG.update(K_max=a.fixed_K, dir_alpha=1.0, modal=False)
    os.makedirs(a.out, exist_ok=True)
    if a.summarise:
        return summarise(a.out, a.split, a.blind)
    if a.full:
        r = full_reference(a.split, CONFIG, a.out)
        return print(f"full data: ARI {r['ari']:.3f}, floor ARI {np.mean(r['floor_ari']):.3f}, sampler {r['sampler_time']:.0f}s")
    ref = full_reference(a.split, CONFIG, a.out)        # reference fits first: replicates are scored against them
    print(f"full data: ARI vs gating {ref['ari']:.3f}; floor (full vs full) ARI {np.mean(ref.get('floor_ari', np.nan)):.3f} "
          f"VoI {np.mean(ref.get('floor_voi', np.nan)):.3f}; sampler {ref['sampler_time']:.0f}s", flush=True)
    jobs = [(a.split, n, r, CONFIG, a.out) for n in [int(x) for x in a.ns.split(',')] for r in parse_range(a.reps)]
    if a.n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.n_jobs) as pool:
            for path in pool.imap_unordered(_worker, jobs):
                print(os.path.basename(path), flush=True)
    else:
        for j in jobs:
            print(os.path.basename(_worker(j)), flush=True)


if __name__ == '__main__':
    main()
