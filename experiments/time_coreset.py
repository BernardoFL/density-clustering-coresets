"""
Construction times of the coreset, measured cleanly: one build at a time, one thread, the
same configurations as the reported studies (valley simulation, curved simulation,
Levine_13dim test half, Visium).  The reported studies ran many builds in parallel, so
their recorded times include contention.

Usage:  python experiments/time_coreset.py --study valley|curved|levine|visium [--reps 0-2] [--ns 50,100]
        python experiments/time_coreset.py --summarise
Each build is written to experiments/results/timings/<study>_n<n>[_<setting>]_rep<r>.json
(existing files are skipped).  Run the studies one after another for clean numbers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'TQDM_DISABLE': '1',
             'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false'}.items():
    os.environ.setdefault(k, v)
import argparse, glob, json, platform, time
import numpy as np
from experiments.runlog import save_json, parse_range

OUT = os.path.join(os.path.dirname(__file__), 'results', 'timings')
RES = os.path.join(os.path.dirname(__file__), 'results')


def _build(X, n, seed, K_cl, cfg, **extra):
    from coreset.algorithm import build_coreset_selected
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    t0 = time.perf_counter()
    cs = build_coreset_selected(
        data=X, n=n, niter=cfg['niter'], L=cfg['L'],
        sample_hyperprior=lambda g: sample_niw_hyperprior(X, g),
        build_base_measure=lambda p, g: build_gaussian_base(*p, g),
        N_r=cfg['N_r'], K_cl=K_cl, alpha=cfg.get('alpha', 1.0), nval=cfg['nval'],
        minsize=cfg.get('minsize', 1), lr=cfg.get('lr', 3e-3), n_steps=cfg['n_steps'], seed=seed, **extra)
    total = time.perf_counter() - t0
    return {'total': total, **cs['timings'], 'N': len(X), 'd': X.shape[1], 'n': n,
            'lambda_out': cs['lambda_out']}


def jobs(study, reps):
    """(file name, thunk) per build."""
    out = []
    if study == 'valley':
        import experiments.exp_sim_valleys as E
        for sep in E.SEPS:
            for p in E.PMINS:
                for n in E.NS:
                    for r in reps:
                        def f(sep=sep, p=p, n=n, r=r):
                            cell = E.SEPS.index(sep) * 100 + E.PMINS.index(p) * 10 + E.NS.index(n)
                            s_data, s_core, _ = np.random.SeedSequence([E.MASTER, cell, r]).spawn(3)
                            X = E.generate(sep, p, np.random.default_rng(s_data))[0]
                            return _build(X, n, s_core, E.CFG['K'], E.CFG)
                        out.append((f'valley_n{n:04d}_s{sep:g}_p{p:g}_rep{r:02d}', f))
    elif study == 'curved':
        from experiments.exp_sim_curved import CONFIG_FULL as C, SEED_NAMES
        from experiments.runlog import child_seeds
        from data.generate import generate_curved_mixture_hd
        for n in (25, 50, 100, 200):
            for r in reps:
                def f(n=n, r=r):
                    seeds = child_seeds(42, r, SEED_NAMES)
                    X = generate_curved_mixture_hd(
                        N=C['N'], d=C['d'], n_common=C['n_common'], n_rare=C['n_rare'], rare_weight=0.01,
                        nested=True, shape_scale=4.0, noise_std=0.6, rare_offset_frac=0.5,
                        rng=np.random.default_rng(seeds['data']))[0]
                    return _build(X, n, seeds['coreset'], C['n_common'] + C['n_rare'], C)
                out.append((f'curved_n{n:04d}_rep{r:02d}', f))
    elif study == 'levine':
        from experiments.exp_levine13 import half, CONFIG as C, MASTER, SEED_NAMES
        from experiments.runlog import child_seeds
        for n in (50, 100, 150, 250, 400, 1000):
            for r in reps:
                def f(n=n, r=r):
                    X = half('test', C)[0]
                    return _build(X, n, child_seeds(MASTER, r, SEED_NAMES)['coreset'], C['K_cl'], C,
                                  w_merge=C.get('w_merge', 1.0))
                out.append((f'levine_n{n:04d}_rep{r:02d}', f))
    elif study == 'visium':
        from experiments.exp_visium import _load, CONFIG, SEED_NAMES
        from experiments.runlog import child_seeds
        C = dict(CONFIG, lambda_spatial=0.5)
        for r in reps:
            def f(r=r):
                v = _load(os.path.join(RES, 'visium'))
                X = np.hstack([v['X_pca'][:, :40], C['lambda_spatial'] * v['spatial_std']])
                return _build(X, C['n'], child_seeds(42, r, SEED_NAMES)['coreset'], C['K'], C, min_share=0.01)
            out.append((f'visium_n0100_rep{r:02d}', f))
    return out


def summarise():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, '*.json')))]
    print(f"{'study':8s} {'n':>5s} {'builds':>6s} {'median total (s)':>17s} {'collect':>8s} {'transform':>10s} {'select':>7s}")
    for st in ('valley', 'curved', 'levine', 'visium'):
        for n in sorted({r['n'] for r in rows if r['study'] == st}):
            S = [r for r in rows if r['study'] == st and r['n'] == n]
            m = lambda k: np.median([r[k] for r in S])
            print(f"{st:8s} {n:5d} {len(S):6d} {m('total'):17.0f} {m('collect'):8.1f} {m('transform'):10.0f} {m('select'):7.0f}")
    if rows:
        print(f"\nmachine: {rows[0]['machine']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', choices=['valley', 'curved', 'levine', 'visium'])
    p.add_argument('--reps', default='0')
    p.add_argument('--ns', default=None, help="only these coreset sizes, e.g. '50,100'")
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.summarise or not a.study:
        return summarise()
    ns = {int(x) for x in a.ns.split(',')} if a.ns else None
    for name, f in jobs(a.study, parse_range(a.reps)):
        if ns and int(name.split('_n')[1][:4]) not in ns:
            continue
        path = os.path.join(OUT, name + '.json')
        if os.path.exists(path):
            continue
        rec = f()
        rec.update(study=a.study, name=name, machine=f'{platform.platform()} / {platform.processor()} / 1 thread')
        save_json(path, rec)
        print(f"{name}: {rec['total']:.0f}s", flush=True)
    summarise()


if __name__ == '__main__':
    main()
