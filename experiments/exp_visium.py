"""
10x Visium mouse brain (Supplement, "Additional studies: Spatial transcriptomics").

Features: the first d principal components of the expression data (d = d* from
visium_choose_d.py) and the standardized tissue coordinates with weight lambda_spatial = 0.5.
For each replicate r:

  A  coreset:   support uniform from y_A, transform T_out fitted by build_coreset_selected on
                the features (Clust with min_share = 0.01)
  C  baseline:  a spatially stratified subsample (k-means on the tissue coordinates into n
                strata, one spot per stratum)

The paired comparison of the coreset with uniform subsampling on the same support (fresh
evaluation draws) is stored in d{d:03d}_rep{r:02d}.json under 'coreset'.  Every arm also fits
a diagonal BayesianGaussianMixture (K = 8) and labels all N spots (likelihood power N/n for
subsets, implemented by replicating each point round(N/n) times); these downstream fits are
stored but not reported.

Replicate r uses the same seeds at every d, so the split and the support depend on r only.

Outputs (in --out):
  ref_d{d:03d}.json           full-data BGMM labels for several seeds
  d{d:03d}_rep{r:02d}.json    one replicate

Usage
-----
  python experiments/visium_choose_d.py                   # once, writes d_star.json
  python experiments/exp_visium.py --reps 0-19 --n_jobs 4
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import argparse
import json
import time
from itertools import combinations

import numpy as np

from experiments.runlog import child_seeds, seed_int, save_json, parse_range

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'visium')

CONFIG = dict(
    n=100, niter=30, L=300, alpha=1.0, N_r=600, nval=100, minsize=1,
    lambda_spatial=0.5, K=8, n_steps=300, lr=3e-3, n_ref_seeds=10,
    downstream_power='N/n',
)

# Order and position unchanged from earlier runs, so each arm keeps its seeds.
SEED_NAMES = ['coreset', 'strata', 'strata_pick', 'bgmm_A', 'bgmm_C']
BGMM_SEED = {'A': 'bgmm_A', 'C': 'bgmm_C', 'U': 'bgmm_A'}   # U paired with A


def arms_for(cfg):
    return ['A', 'C'] if cfg['lambda_spatial'] > 0 else ['A', 'U']


def fit_bgmm(Z: np.ndarray, K: int, seed: int):
    """Downstream diagonal BGMM, with the initialisation seed exposed."""
    from sklearn.mixture import BayesianGaussianMixture
    return BayesianGaussianMixture(
        n_components=K, covariance_type='diag',
        n_init=3, max_iter=300,
        weight_concentration_prior=1.0 / K,
        random_state=seed,
    ).fit(Z)


def fit_bgmm_powered(Z: np.ndarray, K: int, seed: int, power: int):
    """fit_bgmm with each observation's log-likelihood multiplied by an integer power."""
    return fit_bgmm(np.repeat(Z, power, axis=0), K, seed)


def _load(out_dir):
    from data.visium import load_visium
    return load_visium(os.path.join(out_dir, 'visium_cache.npz'))


def full_reference(d: int, cfg: dict, out_dir: str) -> dict:
    """Full-data BGMM at d for seeds 0..n_ref_seeds-1, cached to ref_d{d}.json."""
    from coreset.distances import voi
    path = os.path.join(out_dir, f'ref_d{d:03d}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    v = _load(out_dir)
    X = v['X_pca'][:, :d]
    labels = []
    t0 = time.perf_counter()
    for s in range(cfg['n_ref_seeds']):
        labels.append(fit_bgmm(X, cfg['K'], seed=s).predict(X))
    t_fit = (time.perf_counter() - t0) / cfg['n_ref_seeds']
    pair = [voi(a, b) for a, b in combinations(labels, 2)]
    rec = {
        'd': d,
        'labels': labels,
        'noise_floor_mean': float(np.mean(pair)),
        'noise_floor_min': float(np.min(pair)),
        'noise_floor_max': float(np.max(pair)),
        'voi_to_leiden': [voi(l, v['leiden']) for l in labels],
        't_fit_mean': t_fit,
    }
    save_json(path, rec)
    with open(path) as f:
        return json.load(f)


def run_replicate(d: int, rep: int, master_seed: int, cfg: dict, out_dir: str) -> dict:
    from sklearn.cluster import KMeans
    from coreset.algorithm import build_coreset_selected
    from coreset.distances import voi
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base

    v   = _load(out_dir)
    ref = full_reference(d, cfg, out_dir)
    ref_labels = [np.asarray(l) for l in ref['labels']]

    seeds = child_seeds(master_seed, rep, SEED_NAMES)
    X       = v['X_pca'][:, :d]
    X_joint = np.hstack([X, cfg['lambda_spatial'] * v['spatial_std']]) if cfg['lambda_spatial'] > 0 else X
    ARMS = arms_for(cfg)
    N, n, K = len(X), cfg['n'], cfg['K']

    def sample_hp(rng_):       return sample_niw_hyperprior(X_joint, rng_)
    def build_base(phi, rng_): return build_gaussian_base(*phi, rng_)

    # ── Arm A: coreset ───────────────────────────────────────────────────────
    cs = build_coreset_selected(
        data=X_joint, n=n, niter=cfg['niter'], L=cfg['L'],
        sample_hyperprior=sample_hp, build_base_measure=build_base,
        N_r=cfg['N_r'], K_cl=K, alpha=cfg['alpha'],
        nval=cfg['nval'], minsize=cfg['minsize'],
        lr=cfg['lr'], n_steps=cfg['n_steps'],
        seed=seeds['coreset'], min_share=cfg.get('clust_min_share', 0.0),
    )
    Z = {'A': cs['y_coreset'][:, :d], 'U': cs['y_star'][:, :d]}

    # ── Arm C: stratified uniform subsample (spatial design only) ────────────
    t0 = time.perf_counter()
    idx_C = None
    if 'C' in ARMS:
        strata = KMeans(n_clusters=n, n_init=1, random_state=seed_int(seeds['strata'])) \
            .fit_predict(v['spatial_std'])
        rng_pick = np.random.default_rng(seeds['strata_pick'])
        idx_C = np.array([rng_pick.choice(np.where(strata == k)[0]) for k in range(n)])
        Z['C'] = X[idx_C]
    t_strata = time.perf_counter() - t0

    # ── Downstream BGMM (likelihood power N/n), labels on all N spots ───────
    power = int(round(N / n))
    arms, timings = {}, {}
    for a in ARMS:
        t0 = time.perf_counter()
        model = fit_bgmm_powered(Z[a], K, seed=seed_int(seeds[BGMM_SEED[a]]), power=power)
        lab = model.predict(X)
        timings[f'bgmm_{a}'] = time.perf_counter() - t0
        n_train = np.bincount(model.predict(Z[a]), minlength=K)
        v_full = [voi(lab, r) for r in ref_labels]
        arms[a] = {
            'labels': lab,
            'voi_full_per_seed': v_full,
            'voi_full': float(np.mean(v_full)),
            'voi_leiden': voi(lab, v['leiden']),
            'K_eff': int(len(np.unique(lab))),
            'n_single_point_components': int(np.sum(n_train == 1)),
        }

    timings.update({
        'coreset_collect':   cs['timings']['collect'],
        'coreset_transform': cs['timings']['transform'],
        'coreset_select':    cs['timings']['select'],
        'strata':            t_strata,
    })
    return {
        'd': d, 'rep': rep, 'master_seed': master_seed, 'seeds': seeds, 'config': cfg,
        'downstream_power_used': power,
        'arms': arms,
        'coreset': {
            'config': cs['config'],
            'selected': cs['selected'],
            'lambda_out': cs['lambda_out'],
            'lambda_grid': cs['lambda_grid'],
            'select_curve': cs['select_curve'],
            'J_hat_bar': cs['J_hat_bar'], 'J_hat_I': cs['J_hat_I'],
            'V_bar': cs['V_bar'], 'V_I': cs['V_I'],
            'clust_params': cs['clust_params'],
            'A_bar': cs['A_bar'], 'b_bar': cs['b_bar'],
            'surrogate_log': cs['surrogate_log'],
            'support_idx': cs['support_idx'],
            'ref_idx': cs['ref_idx'],
        },
        'strata_idx': idx_C,
        'timings': timings,
    }


def _worker(args):
    d, rep, master_seed, cfg, out_dir = args
    path = os.path.join(out_dir, f'd{d:03d}_rep{rep:02d}.json')
    if os.path.exists(path):
        return d, rep, 'skipped'
    t0  = time.perf_counter()
    rec = run_replicate(d, rep, master_seed, cfg, out_dir)
    rec['timings']['total'] = time.perf_counter() - t0
    save_json(path, rec)
    a = rec['arms']
    base = 'U' if 'U' in a else 'C'
    return d, rep, (f"coreset={a['A']['voi_full']:.3f} {base}={a[base]['voi_full']:.3f} "
                    f"sel={rec['coreset']['selected']} "
                    f"{rec['timings']['total']:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dims', default=None, help="e.g. '1-100'; default: d* from d_star.json")
    p.add_argument('--reps', default='0-19')
    p.add_argument('--master_seed', type=int, default=42)
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--out', default=OUT_DIR)
    p.add_argument('--nval', type=int, default=None, help='override nval (pilot only)')
    p.add_argument('--lambda_spatial', type=float, default=None,
                   help='weight of the spatial coordinates in the features (reported runs: 0.5)')
    p.add_argument('--clust_min_share', type=float, default=0.01,
                   help='Clust: components below this share of a trajectory are sub-threshold')
    args = p.parse_args()

    cfg = dict(CONFIG)
    cfg['clust_min_share'] = args.clust_min_share
    if args.lambda_spatial is not None:
        cfg['lambda_spatial'] = args.lambda_spatial
    if args.nval is not None:
        cfg['nval'] = args.nval

    if args.dims is None:
        d_path = os.path.join(args.out, 'd_star.json')
        if not os.path.exists(d_path):
            sys.exit('d_star.json not found: run experiments/visium_choose_d.py first')
        with open(d_path) as f:
            dims = [json.load(f)['d_star']]
    else:
        dims = parse_range(args.dims)
    reps = parse_range(args.reps)
    os.makedirs(args.out, exist_ok=True)

    _load(args.out)                    # build the cache once, before any worker starts
    for d in dims:                     # full-data references, serially, once per d
        ref = full_reference(d, cfg, args.out)
        print(f"d={d:3d} reference: seed-to-seed VoI {ref['noise_floor_mean']:.3f} "
              f"[{ref['noise_floor_min']:.3f}, {ref['noise_floor_max']:.3f}]", flush=True)

    jobs = [(d, r, args.master_seed, cfg, args.out) for d in dims for r in reps]
    if args.n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(args.n_jobs) as pool:
            for d, r, msg in pool.imap_unordered(_worker, jobs):
                print(f'd={d:3d} rep={r:02d}: {msg}', flush=True)
    else:
        for job in jobs:
            d, r, msg = _worker(job)
            print(f'd={d:3d} rep={r:02d}: {msg}', flush=True)


if __name__ == '__main__':
    main()
