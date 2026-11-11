"""
Prespecify the PCA dimension d* for the Visium study,
from the full data alone and before any coreset result is seen.

Rule (primary): Horn's parallel analysis on the scaled HVG matrix.  Each gene
column is permuted independently B times; d* is the number of leading
principal components whose eigenvalue exceeds the 95th percentile of the
corresponding eigenvalue under permutation (counting stops at the first
component that fails).

Also recorded (not used unless --rule evr): the smallest d whose cumulative
explained-variance ratio reaches --evr_share.

BIC across different d is not used: it compares likelihoods of different data.

Writes experiments/results/visium/d_star.json.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA

from data.visium import load_visium

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'visium')


def parallel_analysis(X: np.ndarray, n_comp: int, B: int, q: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=seed)
    obs = pca.fit(X).explained_variance_
    null = np.empty((B, n_comp))
    for b in range(B):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        null[b] = PCA(n_components=n_comp, svd_solver='randomized',
                      random_state=seed + b + 1).fit(Xp).explained_variance_
    thr  = np.percentile(null, 100 * q, axis=0)
    keep = obs > thr
    d_star = int(np.argmin(keep)) if not keep.all() else n_comp
    return {'d_star': d_star, 'observed': obs, 'threshold': thr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rule', choices=['parallel', 'evr'], default='parallel')
    p.add_argument('--B', type=int, default=20)
    p.add_argument('--quantile', type=float, default=0.95)
    p.add_argument('--evr_share', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=OUT_DIR)
    args = p.parse_args()

    v  = load_visium(os.path.join(args.out, 'visium_cache.npz'))
    pa = parallel_analysis(v['X_hvg'].astype(float), n_comp=v['X_pca'].shape[1],
                           B=args.B, q=args.quantile, seed=args.seed)
    cum   = np.cumsum(v['pca_var_ratio'])
    d_evr = int(np.searchsorted(cum, args.evr_share) + 1) if cum[-1] >= args.evr_share else None

    d_star = pa['d_star'] if args.rule == 'parallel' else d_evr
    if d_star is None:
        sys.exit(f'cumulative EVR of all PCs is {cum[-1]:.1%} < {args.evr_share:.0%}')
    rec = {
        'd_star': d_star,
        'rule': args.rule,
        'rule_text': (
            f'Horn parallel analysis, {args.B} column permutations, '
            f'{args.quantile:.0%} quantile' if args.rule == 'parallel'
            else f'smallest d with cumulative explained variance >= {args.evr_share:.0%}'
        ),
        'd_parallel': pa['d_star'],
        'd_evr': d_evr,
        'evr_share': args.evr_share,
        'cum_evr_at_d_star': float(cum[d_star - 1]),
        'pa_observed': pa['observed'].tolist(),
        'pa_threshold': pa['threshold'].tolist(),
    }
    with open(os.path.join(args.out, 'd_star.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(f"d* = {d_star}  ({rec['rule_text']}); parallel={pa['d_star']}, "
          f"evr{args.evr_share:.0%}={d_evr}; cumulative EVR at d* = {cum[d_star - 1]:.1%}")


if __name__ == '__main__':
    main()
