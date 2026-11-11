"""
Curved-cluster simulation, d = 20 (Supplement, "Additional studies: Curved clusters in 20 dimensions").

Misspecified mixture of curved clusters (data.generate.generate_curved_mixture_hd), N = 10,000.
Per run the coreset is built by build_coreset_selected; the uniform baseline is the identity
coreset y* (same support, paired).  The supplement reports the paired loss comparison stored
under 'coreset' in each JSON.

The script also fits downstream mixture models (full-data NUTS and SVI, NUTS on the coreset,
uniform and QNC subsets) and records point estimates and co-clustering summaries; these
outputs are stored ('point', 'point_svi', 'uq') but not reported in the paper.

One JSON per run in --out; existing files are skipped, so the script resumes.

Usage
-----
  python experiments/exp_sim_curved.py --runs 0-99 --n_jobs 4              # n = 50
  python experiments/exp_sim_curved.py --n 25 --runs 0-19 --n_jobs 4       # also n = 100, 200
  python experiments/exp_sim_curved.py --runs 0-1 --smoke                  # quick end-to-end check
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import argparse
import time

import numpy as np

from experiments.runlog import child_seeds, save_json, parse_range

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Coreset and uniform share their NUTS seeds (unit and N/n), so when the
# selection step returns I the two fits are identical rather than differing
# by MCMC noise.
SEED_NAMES = [
    'data', 'coreset', 'eval', 'qnc_init_support', 'qnc_init_nuts', 'qnc',
    'svi_full', 'nuts_full', 'nuts_subset', 'nuts_qnc', 'nuts_subset_pow', 'cocl',
]

CONFIG_FULL = dict(
    N=10_000, d=20, n_common=3, n_rare=5, n=50, niter=40, L=500, alpha=1.0,
    N_r=2000, nval=100, minsize=1, n_steps=500, lr=3e-3,
    N_eval=2000, M=1500,
    nuts_draws=200, nuts_tune=500, svi_steps=500, svi_samples=200,
    qnc_proj_dim=200,
)

# Plumbing check only: with nuts_tune=30 the NUTS posteriors collapse to one
# component, so the point-2 numbers from a smoke run are meaningless.
CONFIG_SMOKE = dict(
    CONFIG_FULL,
    N=1000, niter=5, L=100, N_r=200, nval=5, n_steps=20,
    N_eval=400, M=300, nuts_draws=20, nuts_tune=30, svi_steps=50, svi_samples=20,
    qnc_proj_dim=20,
)


def _timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def run_single(run_idx: int, master_seed: int, cfg: dict) -> dict:
    from data.generate import generate_curved_mixture_hd
    from coreset.algorithm import build_coreset_selected
    from coreset.distances import voi
    from coreset.qnc import build_qnc_coreset
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    from inference.gmm_numpyro import (
        fit_gmm_svi, fit_gmm_nuts_diag, fit_gmm_nuts_diag_weighted, cluster_posterior,
    )
    from inference.coclustering import summarise_posterior, coclustering_metrics

    seeds = child_seeds(master_seed, run_idx, SEED_NAMES)
    R     = lambda name: np.random.default_rng(seeds[name])

    N, d, n = cfg['N'], cfg['d'], cfg['n']
    K = K_cl = cfg['n_common'] + cfg['n_rare']

    data, _, _, _ = generate_curved_mixture_hd(
        N=N, d=d, n_common=cfg['n_common'], n_rare=cfg['n_rare'],
        rare_weight=0.01, nested=True, shape_scale=4.0, noise_std=0.6,
        rare_offset_frac=0.5, rng=R('data'),
    )
    eval_perm = R('eval').permutation(N)
    data_eval = data[eval_perm[:cfg['N_eval']]]     # point-estimate VoI (Figure 3)
    data_M    = data[eval_perm[:cfg['M']]]          # shared co-clustering subset

    def sample_hp(rng_):       return sample_niw_hyperprior(data, rng_)
    def build_base(phi, rng_): return build_gaussian_base(*phi, rng_)

    # ── Coreset with selection step ──────────────────────────────────────────
    cs = build_coreset_selected(
        data=data, n=n, niter=cfg['niter'], L=cfg['L'],
        sample_hyperprior=sample_hp, build_base_measure=build_base,
        N_r=cfg['N_r'], K_cl=K_cl, alpha=cfg['alpha'],
        nval=cfg['nval'], minsize=cfg['minsize'],
        lr=cfg['lr'], n_steps=cfg['n_steps'],
        seed=seeds['coreset'], min_share=cfg.get('clust_min_share', 0.0),
    )
    if cfg.get('coreset_only'):            # criterion only: no QNC / NUTS / SVI downstream
        return {'run_idx': run_idx, 'master_seed': master_seed, 'seeds': seeds, 'config': cfg,
                'coreset': {k: cs[k] for k in ('config', 'selected', 'lambda_out', 'lambda_grid', 'select_curve',
                                               'J_hat_bar', 'J_hat_I', 'V_bar', 'V_I', 'clust_params',
                                               'A_bar', 'b_bar', 'surrogate_log', 'support_idx')},
                'timings': {f'coreset_{k}': v for k, v in cs['timings'].items()}}
    y_core = cs['y_coreset']
    y_unif = cs['y_star']             # identity coreset = uniform subsample from y_A

    # ── Shared hyperprior and inference settings ─────────────────────────────
    hp = {
        'mu0': data.mean(axis=0), 'kappa': 1.0, 'nu': float(d + 2),
        'Psi': np.cov(data.T, ddof=1) + 1e-4 * np.eye(d),
    }
    nuts_kw = dict(K=K, draws=cfg['nuts_draws'], tune=cfg['nuts_tune'], chains=1,
                   hyperprior_params=hp)

    # ── QNC ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_init    = data[R('qnc_init_support').choice(N, size=n, replace=False)]
    post_init = fit_gmm_nuts_diag(y_init, rng=R('qnc_init_nuts'), **nuts_kw)
    y_qnc, w_qnc, _ = build_qnc_coreset(data, n, post_init,
                                        proj_dim=cfg['qnc_proj_dim'], rng=R('qnc'))
    t_build_qnc = time.perf_counter() - t0

    # ── Full-data references ─────────────────────────────────────────────────
    post_full, t_full = _timed(fit_gmm_nuts_diag, data, rng=R('nuts_full'), **nuts_kw)
    post_svi, t_svi = _timed(fit_gmm_svi, data, K=K, n_steps=cfg['svi_steps'],
                             batch_size=256, n_samples=cfg['svi_samples'],
                             hyperprior_params=hp, rng=R('svi_full'))

    # ── Point-estimate comparison (Figure 3) ─────────────────────────────────
    post_core, t_core = _timed(fit_gmm_nuts_diag, y_core, rng=R('nuts_subset'), **nuts_kw)
    post_unif, t_unif = _timed(fit_gmm_nuts_diag, y_unif, rng=R('nuts_subset'), **nuts_kw)
    post_qnc,  t_qnc  = _timed(fit_gmm_nuts_diag_weighted, y_qnc, w_qnc,
                               rng=R('nuts_qnc'), **nuts_kw)

    t0 = time.perf_counter()
    lab = {
        'full':     cluster_posterior(data_eval, post_full, K_cl),
        'full_svi': cluster_posterior(data_eval, post_svi,  K_cl),
        'core':     cluster_posterior(data_eval, post_core, K_cl),
        'unif':     cluster_posterior(data_eval, post_unif, K_cl),
        'qnc':      cluster_posterior(data_eval, post_qnc,  K_cl),
    }
    t_label = time.perf_counter() - t0

    def _point(ref):
        out = {m: voi(lab[m], lab[ref]) for m in ('core', 'unif', 'qnc')}
        return {
            'voi_core': out['core'], 'voi_unif': out['unif'], 'voi_qnc': out['qnc'],
            'delta_core': out['unif'] - out['core'],
            'delta_qnc':  out['unif'] - out['qnc'],
        }

    point = _point('full')                  # primary: full-data NUTS
    point['K_eff'] = {k: int(len(np.unique(v))) for k, v in lab.items()}
    point['voi_full_svi_to_full'] = voi(lab['full_svi'], lab['full'])
    point_svi = _point('full_svi')          # secondary: full-data SVI (original Figure 3)

    record = {
        'run_idx': run_idx,
        'master_seed': master_seed,
        'seeds': seeds,
        'config': cfg,
        'coreset': {
            'config': cs['config'],
            'selected': cs['selected'],
            'lambda_out': cs['lambda_out'],
            'lambda_grid': cs['lambda_grid'],
            'select_curve': cs['select_curve'],
            'J_hat_bar': cs['J_hat_bar'],
            'J_hat_I': cs['J_hat_I'],
            'V_bar': cs['V_bar'],
            'V_I': cs['V_I'],
            'clust_params': cs['clust_params'],
            'A_bar': cs['A_bar'], 'b_bar': cs['b_bar'],
            'surrogate_log': cs['surrogate_log'],
            'support_idx': cs['support_idx'],
        },
        'point': point,
        'point_svi': point_svi,
        'timings': {
            'coreset_collect':   cs['timings']['collect'],
            'coreset_transform': cs['timings']['transform'],
            'coreset_select':    cs['timings']['select'],
            'infer_core':        t_core,
            'infer_unif':        t_unif,
            'build_qnc':         t_build_qnc,
            'infer_qnc':         t_qnc,
            'infer_full_svi':    t_svi,
            'infer_full_nuts':   t_full,
            'label_point':       t_label,
        },
    }

    # ── Posterior co-clustering vs full-data NUTS (not reported) ─────────────
    w_pow = np.full(n, N / n)
    post_core_pow, t_core_pow = _timed(fit_gmm_nuts_diag_weighted, y_core, w_pow,
                                       rng=R('nuts_subset_pow'), **nuts_kw)
    post_unif_pow, t_unif_pow = _timed(fit_gmm_nuts_diag_weighted, y_unif, w_pow,
                                       rng=R('nuts_subset_pow'), **nuts_kw)
    posts = {
        'full': post_full,
        'core_unit': post_core, 'core_pow': post_core_pow,
        'unif_unit': post_unif, 'unif_pow': post_unif_pow,
        'qnc': post_qnc,
    }
    rng_c = R('cocl')
    summ  = {k: summarise_posterior(data_M, p, rng_c) for k, p in posts.items()}
    ref   = summ['full']
    uq = {}
    for k, s in summ.items():
        entry = {'voi_spread': s['voi_spread'],
                 'voi_hat_to_full_hat': voi(s['s_hat'], ref['s_hat'])}
        if k != 'full':
            entry.update(coclustering_metrics(s['P'], ref['P']))
        uq[k] = entry
    record['uq'] = uq
    record['timings'].update({
        'infer_core_pow':  t_core_pow,
        'infer_unif_pow':  t_unif_pow,
    })

    return record


def _worker(args):
    run_idx, master_seed, cfg, out_dir = args
    path = os.path.join(out_dir, f'run_{run_idx:03d}.json')
    if os.path.exists(path):
        return run_idx, 'skipped'
    t0 = time.perf_counter()
    rec = run_single(run_idx, master_seed, cfg)
    rec['timings']['total'] = time.perf_counter() - t0
    save_json(path, rec)
    return run_idx, f"done in {rec['timings']['total']:.0f}s"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='0-99')
    p.add_argument('--master_seed', type=int, default=42)
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--out', default=None)
    p.add_argument('--smoke', action='store_true', help='tiny configuration for an end-to-end check')
    p.add_argument('--n', type=int, default=None,
                   help='coreset size (default 50); results go to results/sim_curved_n{n} unless --out is given')
    p.add_argument('--clust_min_share', type=float, default=0.0,
                   help='Clust: components below this share of a trajectory are sub-threshold')
    p.add_argument('--coreset_only', action='store_true', help='build the coreset and record the criterion only')
    args = p.parse_args()

    cfg       = dict(CONFIG_SMOKE if args.smoke else CONFIG_FULL)
    cfg.update(clust_min_share=args.clust_min_share, coreset_only=args.coreset_only)
    if args.n is not None:
        cfg['n'] = args.n
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"sim_curved_n{cfg['n']}")
    runs      = parse_range(args.runs)
    jobs = [(i, args.master_seed, cfg, args.out) for i in runs]
    os.makedirs(args.out, exist_ok=True)

    if args.n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(args.n_jobs) as pool:
            for run_idx, msg in pool.imap_unordered(_worker, jobs):
                print(f'run {run_idx:03d}: {msg}', flush=True)
    else:
        for job in jobs:
            run_idx, msg = _worker(job)
            print(f'run {run_idx:03d}: {msg}', flush=True)


if __name__ == '__main__':
    main()
