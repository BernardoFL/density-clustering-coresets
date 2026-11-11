"""
Sensitivity to the DP concentration alpha (Supplement, "Sensitivity to the concentration parameter").

Rebuilds coresets of reported valley-separated replicates (same data, reference split, support and
seeds) at other values of alpha.
alpha changes both the full-data and the coreset predictive laws, so each alpha is assessed
against its own full-data target.  The alpha = 1 rows are the reported results (sim_valleys).

  valley: all (Delta, p_min) at n = 100, alpha in {0.01, 0.1, 0.3, 3, 10, 30, 100}, reps 0-9
  check : alpha = 1 replay of one replicate, compared with the stored result

Usage:  python experiments/exp_alpha_sensitivity.py --n_jobs 14
        python experiments/exp_alpha_sensitivity.py --summarise
"""
import os, sys, glob, json, itertools, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from experiments.runlog import save_json

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = os.path.join(RES, 'alpha_sensitivity')
ALPHAS, N, REPS = (0.01, 0.1, 0.3, 3.0, 10.0, 30.0, 100.0), 100, range(10)


def _path(job):
    sep, p, a, rep = job
    return os.path.join(OUT, f's{sep:g}_p{p:g}_a{a:g}_rep{rep:02d}.json')


def _worker(job):
    from experiments.exp_sim_valleys import run
    path = _path(job)
    if os.path.exists(path):
        return path, 'skip'
    sep, p, a, rep = job
    r = run(sep, p, N, rep, urn_reinforce=True, alpha=a)
    res = {'sep': sep, 'p_min': p, 'n': N, 'alpha': a, 'rep': rep, 'eval': r['eval'],
           'lambda_out': r['lambda_out'], 't_build': r['t_build']}
    save_json(path, res)
    return path, f"{r['t_build']:.0f}s"


def jobs():
    from experiments.exp_sim_valleys import SEPS, PMINS
    J = [(3.0, 0.05, 1.0, 0)]   # replay check
    J += [(s, p, a, r) for r in REPS for s, p in itertools.product(SEPS, PMINS) for a in ALPHAS]
    return J


def _lo(a, b):
    from scipy import stats
    dd = np.asarray(a) - np.asarray(b)
    return dd.mean() - stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if dd.std() > 0 else 0.0


def summarise():
    rows = []
    for f in glob.glob(os.path.join(OUT, '*.json')):
        r = json.load(open(f))
        if r['alpha'] == 1.0:
            continue
        if r['rep'] in REPS:
            rows.append((r['sep'], r['p_min'], r['alpha'], r['eval'], r['lambda_out']))
    for f in glob.glob(os.path.join(RES, 'sim_valleys', f's*_n{N:04d}_rep*.json')):
        r = json.load(open(f))
        if r['rep'] in REPS:
            rows.append((r['sep'], r['p_min'], 1.0, r['eval'], r['lambda_out']))
    chk = os.path.join(OUT, 's3_p0.05_a1_rep00.json')
    if os.path.exists(chk):
        a = json.load(open(chk))['eval']; b = json.load(open(os.path.join(RES, 'sim_valleys', 's3_p0.05_n0100_rep00.json')))['eval']
        print('replay check (alpha=1): max |diff| unif', np.max(np.abs(np.subtract(a['voi_unif'], b['voi_unif']))),
              'core', np.max(np.abs(np.subtract(a['voi_core'], b['voi_core']))))
    print('sep p_min alpha | J_unif J_core | decrease [IQR] int>0 | ARI full/unif/core | lambda>0')
    for key in sorted({r[:3] for r in rows}, key=lambda k: (-k[0], -k[1], k[2])):
        S = [r for r in rows if r[:3] == key]
        ju = np.array([np.mean(r[3]['voi_unif']) for r in S]); jc = np.array([np.mean(r[3]['voi_core']) for r in S])
        dec = 100 * (ju - jc) / ju; ip = sum(_lo(r[3]['voi_unif'], r[3]['voi_core']) > 0 for r in S)
        af, au, ac = (np.mean([np.mean(r[3][k]) for r in S]) for k in ('ari_full', 'ari_unif', 'ari_core'))
        print(f'{key[0]:g} {key[1]:g} {key[2]:6g} | {np.median(ju):5.2f} {np.median(jc):5.2f} | '
              f'{np.median(dec):5.1f}% [{np.percentile(dec, 25):3.0f},{np.percentile(dec, 75):3.0f}] {ip:2d}/{len(S):<2d} | '
              f'{af:.2f}/{au:.2f}/{ac:.2f} | {sum(r[4] > 0 for r in S)}/{len(S)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    if a.summarise:
        return summarise()
    os.makedirs(OUT, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.n_jobs) as pool:
        for out in pool.imap_unordered(_worker, jobs()):
            print(*out, flush=True)
    summarise()


if __name__ == '__main__':
    main()
