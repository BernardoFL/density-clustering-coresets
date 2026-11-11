"""
Report for the Levine_13dim experiment: per n, paired comparisons coreset vs uniform on every
metric (sign test, mean paired difference with 95% t-interval), IQR of ARI across replicates,
the selection step's fresh-draw VoI_ref gain, and the full-data fit.

Usage:  python experiments/levine13_report.py [--in experiments/results/levine13_test]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json

import numpy as np
from scipy import stats
from scipy.stats import binomtest

from experiments.runlog import load_runs

# metric, direction (+1: higher is better), label
METRICS = [('ari_full', +1, 'ARI vs full fits'), ('voi_full', -1, 'VoI vs full fits'), ('ari', +1, 'ARI vs gating'), ('macro_f1', +1, 'macro-F1'), ('K_bias', -1, '|K̂ − K_true|'),
           ('brier', -1, 'Brier'), ('ess_per_s', +1, 'ESS/s')]


def paired(c, u, sign):
    d = sign * (np.asarray(c) - np.asarray(u))          # positive = coreset better
    nz = d[d != 0]
    p = binomtest(int((nz > 0).sum()), len(nz)).pvalue if len(nz) else 1.0
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum()), int((d < 0).sum()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=os.path.join(os.path.dirname(__file__), 'results', 'levine13_test'))
    a = ap.parse_args()
    split = 'test' if 'test' in a.inp else 'dev'
    runs = load_runs(a.inp, f'{split}_n*_rep*.json')
    fp = os.path.join(a.inp, f'full_{split}.json')
    print(f'Levine_13dim, {split} half — {len(runs)} replicates.  Positive differences = coreset better.\n')
    if os.path.exists(fp):
        f = json.load(open(fp))
        if 'floor_ari' in f:
            print(f"Full-data fits vs each other: ARI {np.mean(f['floor_ari']):.3f}, VoI {np.mean(f['floor_voi']):.3f}")
        print(f"Full data (N={f['N']}): ARI {f['ari']:.3f}, macro-F1 {f['macro_f1']:.3f}, K̂ {f['K_hat']} (K_true {f['K_true']}), "
              f"Brier {f['brier']:.4f}, ESS/s {f['ess_per_s']:.1f}, sampler {f['sampler_time']:.0f}s\n")
    for n in sorted({r['n'] for r in runs}):
        S = [r for r in runs if r['n'] == n]
        print(f'n = {n}  ({len(S)} replicates, K_true = {S[0]["K_true"]})')
        print(f'  {"metric":16s} {"coreset":>9s} {"uniform":>9s}   mean diff [95% CI]          wins/losses  sign p')
        for k, sgn, lab in METRICS:
            if k not in S[0]['arms']['coreset']:
                continue
            c = [r['arms']['coreset'][k] for r in S]
            u = [r['arms']['uniform'][k] for r in S]
            m, lo, hi, w, l, p = paired(c, u, sgn)
            print(f'  {lab:16s} {np.median(c):9.4f} {np.median(u):9.4f}   {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   {w:2d}/{l:2d}       {p:.3g}')
        iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))
        print(f"  IQR of ARI across replicates: coreset {iqr([r['arms']['coreset']['ari'] for r in S]):.4f}, "
              f"uniform {iqr([r['arms']['uniform']['ari'] for r in S]):.4f}")
        D = [np.mean(np.asarray(r['coreset']['V_I']) - np.asarray(r['coreset']['V_bar'])) for r in S]
        lam = [r['coreset']['lambda_out'] for r in S]
        print(f"  selection step: λ*>0 in {sum(l > 0 for l in lam)}/{len(S)}; fresh-draw VoI_ref gain median {np.median(D):+.3f}")
        print(f"  K̂ median: coreset {np.median([r['arms']['coreset']['K_hat'] for r in S]):.0f}, "
              f"uniform {np.median([r['arms']['uniform']['K_hat'] for r in S]):.0f}")
        print(f"  time: coreset build {np.median([r['timings']['coreset_build'] for r in S]):.0f}s, "
              f"Gibbs {np.median([r['timings']['gibbs_coreset'] for r in S]):.0f}s\n")


if __name__ == '__main__':
    main()
