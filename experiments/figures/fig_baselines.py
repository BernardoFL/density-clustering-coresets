"""
Main-text figure: loss against coreset size for the uniform subsample, the QNC reweighting, the
observation-space W_2 affine map and the predictive partition-matching coreset, with the full-data
replication loss, for the valley simulation (Delta = 3 and 2, pooled over p_min) and Levine_13.
All four methods share the support and the evaluation draws of each replicate.

Usage:  python experiments/figures/fig_baselines.py
"""
import os, sys, glob, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, 'experiments', 'results')
FIG = os.path.join(ROOT, 'figures', 'fig_baselines.pdf')
STYLE = {  # palette slots 1-3 in fixed order; neutral gray for the reference method
    'unif': ('Uniform', '#6b6b66', 'o'),
    'qnc': ('QNC', '#eb6834', 's'),
    'w2': (r'$W_2$ affine', '#1baf7a', '^'),
    'core': ('Partition coreset', '#2a78d6', 'D'),
}


def load():
    rows = []
    for f in glob.glob(os.path.join(RES, 'claici_baseline', 'valley', '*.json')):
        r = json.load(open(f)); b = os.path.basename(f)
        q = json.load(open(os.path.join(RES, 'qnc_baseline', 'valley', b)))
        p = json.load(open(os.path.join(RES, 'supp_replication', 'valley', b)))
        rows.append({'panel': f'Simulation, $\\Delta={r["sep"]:g}$', 'n': r['n'],
                     'unif': np.mean(r['eval']['voi_unif']), 'w2': np.mean(r['eval']['voi_w2']),
                     'qnc': np.mean(q['eval']['voi_qnc']), 'core': np.mean(r['voi_core']),
                     'repl': np.mean(p['eval']['voi_repl'])})
    for f in glob.glob(os.path.join(RES, 'claici_baseline', 'levine', '*.json')):
        r = json.load(open(f)); b = os.path.basename(f)
        q = json.load(open(os.path.join(RES, 'qnc_baseline', 'levine', b)))
        p = json.load(open(os.path.join(RES, 'supp_replication', 'levine', b)))
        rows.append({'panel': 'Levine$_{13}$', 'n': r['n'],
                     'unif': np.mean(r['eval']['voi_unif']), 'w2': np.mean(r['eval']['voi_w2']),
                     'qnc': np.mean(q['eval']['voi_qnc']), 'core': np.mean(r['voi_core']),
                     'repl': np.mean(p['eval']['voi_repl'])})
    return rows


def main():
    rows = load()
    panels = ['Simulation, $\\Delta=3$', 'Simulation, $\\Delta=2$', 'Levine$_{13}$']
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#8a8a85', 'xtick.color': '#55554f', 'ytick.color': '#55554f'})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.15), constrained_layout=True)
    for ax, pan in zip(axes, panels):
        S = [r for r in rows if r['panel'] == pan]
        ns = sorted({r['n'] for r in S})
        ax.axhline(np.median([r['repl'] for r in S]), color='#1f1f1d', lw=1, ls=(0, (4, 3)), zorder=1)
        for k, (lab, col, mk) in STYLE.items():
            med = [np.median([r[k] for r in S if r['n'] == n]) for n in ns]
            lo = [np.percentile([r[k] for r in S if r['n'] == n], 25) for n in ns]
            hi = [np.percentile([r[k] for r in S if r['n'] == n], 75) for n in ns]
            ax.fill_between(ns, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(ns, med, color=col, lw=1.6, marker=mk, ms=4.5, mec='white', mew=0.8, label=lab, zorder=3)
        ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels([str(n) if n in (50, 100, 250, 1000) else '' for n in ns])
        ax.minorticks_off(); ax.set_title(pan, fontsize=8.5)
        ax.set_xlabel('Coreset size $n$')
        ax.grid(axis='y', color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel(r'Loss $\widehat J_{\mathrm{eval}}$ (nats)')
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color='#1f1f1d', lw=1, ls=(0, (4, 3)))); l.append('Full-data replication')
    fig.legend(h, l, loc='outside upper center', ncol=5, frameon=False, fontsize=7.5, handlelength=2.2)
    fig.savefig(FIG)
    print('wrote', FIG)


if __name__ == '__main__':
    main()
