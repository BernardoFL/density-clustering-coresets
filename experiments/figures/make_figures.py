"""
Figures of the paper and supplement, built only from saved results (no refitting).

  fig_sim_truth.pdf  valley simulation: ARI vs the true clusters, full data / uniform / coreset
  fig_main.pdf       relative decrease of the loss, coreset vs uniform: valley simulation and Levine_13
  fig_main_supp.pdf  the same for the curved simulation and Visium (supplement)
  fig_valleys.pdf    appendix: convergence of the uniform subsample's predictive clustering
  fig_timings.pdf    construction time of the coreset (one build at a time, one thread; time_coreset.py)

Palette: the first three categorical slots of the reference palette (validated all-pairs for
normal vision and CVD); colour follows the entity (coreset blue, uniform orange, full data grey
reference). Ordered quantities (cluster separation) use one blue ramp, light to dark.
"""
import os, sys, json, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, 'experiments', 'results')
FIG = os.path.join(ROOT, 'figures')

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
CORESET, UNIFORM, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FULL = '#52514e'
BLUES = {2.0: '#86b6ec', 3.0: '#2a78d6', 4.0: '#123f7a'}          # separation: light -> dark

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, pattern)))]


def rel_decrease(runs, key_i='J_hat_I', key_t='J_hat_bar'):
    a = np.array([r[key_i] if key_i in r else r['coreset'][key_i] for r in runs])
    b = np.array([r[key_t] if key_t in r else r['coreset'][key_t] for r in runs])
    return 100 * (a - b) / a


# ── Fig. 1: valley simulation, recovery of the true clusters ────────────────
runs = load('sim_valleys/s*_rep*.json')
NS = (50, 100, 250)
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True, sharey=True)
for i, s in enumerate((3.0, 2.0)):
    for j, p in enumerate((0.05, 0.01)):
        ax = axes[i, j]
        S = {n: [r for r in runs if r['sep'] == s and r['p_min'] == p and r['n'] == n] for n in NS}
        full = np.mean([np.mean(r['eval']['ari_full']) for n in NS for r in S[n]])
        ax.axhline(full, color=FULL, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for key, col, mk, lab, dx in (('ari_unif', UNIFORM, 's', 'uniform subsample', -0.07), ('ari_core', CORESET, 'o', 'coreset', 0.07)):
            v = np.array([[np.mean(r['eval'][key]) for r in S[n]] for n in NS])
            m, se = v.mean(1), v.std(1, ddof=1) / np.sqrt(v.shape[1])
            ax.errorbar(np.arange(len(NS)) + dx, m, yerr=1.96 * se, color=col, lw=1.8, marker=mk, ms=5.5,
                        mec='white', mew=1.0, capsize=0, elinewidth=1.2, label=lab, zorder=3)
        ax.set_title(f'separation {s:g}, rare prevalence {p:g}', loc='left')
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set_xlim(-0.35, 2.35)
for ax in axes[1]:
    ax.set_xlabel('coreset size $n$')
for ax in axes[:, 0]:
    ax.set_ylabel('ARI vs true clusters')
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=FULL, lw=1.2, ls=(0, (4, 3)))); labels.append('full data')
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, 'fig_sim_truth')

# ── Fig. 2: relative decrease of the loss, all studies ─────────────────
def strip(ax, x, vals, col, jitter=0.12, seed=0):
    rng = np.random.default_rng(seed)
    ax.scatter(x + rng.uniform(-jitter, jitter, len(vals)), vals, s=7, color=col, alpha=0.25, lw=0, zorder=2)
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color=col, lw=2.2, solid_capstyle='round', zorder=3)
    ax.plot([x], [med], marker='o', ms=5.5, color=col, mec='white', mew=1.0, zorder=4)

# main text: valley simulation and Levine_13
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw=dict(width_ratios=[3, 3.6]))
ax = axes[0]
for k, s in enumerate((3.0, 2.0)):
    for i, n in enumerate(NS):
        v = [100 * (np.mean(r['eval']['voi_unif']) - np.mean(r['eval']['voi_core'])) / np.mean(r['eval']['voi_unif'])
             for r in runs if r['sep'] == s and r['n'] == n]
        strip(ax, i + (k - 0.5) * 0.36, np.array(v), BLUES[s], jitter=0.07, seed=10 * k + i)
ax.set_xticks(range(len(NS)), [str(n) for n in NS]); ax.set_xlabel('coreset size $n$')
ax.set_title('valley simulation ($d = 8$)', loc='left')
ax.legend(handles=[plt.Line2D([], [], color=BLUES[3.0], marker='o', lw=2.2, ms=5, mec='white', label='separation 3'),
                   plt.Line2D([], [], color=BLUES[2.0], marker='o', lw=2.2, ms=5, mec='white', label='separation 2')],
          loc='upper right', handlelength=1.4)
ax.set_ylabel('decrease in loss vs uniform (%)')
ax = axes[1]
lev = (50, 100, 150, 250, 400, 1000)
for i, n in enumerate(lev):
    strip(ax, i, rel_decrease(load(f'levine13_test/test_n{n:04d}_rep*.json')), CORESET, seed=30 + i)
ax.set_xticks(range(len(lev)), [str(n) for n in lev]); ax.set_xlabel('coreset size $n$')
ax.set_title('mass cytometry (Levine$_{13}$)', loc='left')
for ax in axes:
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
fig.tight_layout()
save(fig, 'fig_main')

# supplement: curved simulation and Visium
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6), sharey=True, gridspec_kw=dict(width_ratios=[3, 1.3]))
ax = axes[0]
sim = (25, 50, 100, 200)
for i, n in enumerate(sim):
    strip(ax, i, rel_decrease(load(f'sim_curved_n{n}/run_*.json')), CORESET, seed=20 + i)
ax.set_xticks(range(len(sim)), [str(n) for n in sim]); ax.set_xlabel('coreset size $n$')
ax.set_title('curved simulation ($d = 20$)', loc='left')
ax.set_ylabel('decrease in loss vs uniform (%)')
ax = axes[1]
strip(ax, 0, rel_decrease(load('visium/d040_rep*.json')), AQUA, seed=40)
ax.set_xticks([0], ['$n=100$']); ax.set_xlim(-0.8, 0.8); ax.set_xlabel(' ')
ax.set_title('Visium', loc='left')
for ax in axes:
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
fig.tight_layout()
save(fig, 'fig_main_supp')

# ── Construction time (experiments/time_coreset.py) ─────────────────────────
def boxes(ax, groups, col):
    ax.boxplot([g for _, g in groups], positions=[x for x, _ in groups], widths=0.5, patch_artist=True,
               medianprops=dict(color='white', lw=1.4), whiskerprops=dict(color=col, lw=0.9),
               capprops=dict(color=col, lw=0.9), flierprops=dict(marker='o', ms=2.5, mfc=col, mec='none', alpha=0.6),
               boxprops=dict(facecolor=col, edgecolor=col, lw=0.9))

tim = load('timings/*.json')
if tim:
    T = lambda st, n: [r['total'] for r in tim if r['study'] == st and r['n'] == n]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True, gridspec_kw=dict(width_ratios=[2.2, 4.0]))
    lev = (50, 100, 150, 250, 400, 1000)
    panels = (('valley', NS, 'valley simulation ($d = 8$)'),
              ('levine', lev, 'mass cytometry (Levine$_{13}$)'))
    for ax, (st, ns, title) in zip(axes, panels):
        boxes(ax, [(i, T(st, n)) for i, n in enumerate(ns) if T(st, n)], CORESET)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns]); ax.set_xlabel('coreset size $n$')
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('construction time (s)'); axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, 'fig_timings')

# ── Fig. A1 (appendix): convergence of the uniform subsample ────────────────
res = json.load(open(os.path.join(RES, 'uniform_convergence.json')))
NSv = (50, 112, 253, 570, 1281, 2883)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
for ax, p in zip(axes, (0.05, 0.01)):
    for s in (2.0, 3.0, 4.0):
        S = [x for x in res if x['sep'] == s and x['p_min'] == p]
        a = np.array([[x[str(n)][0] for n in NSv] for x in S]); full = np.mean([x['full'][0] for x in S])
        m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(S))
        ax.fill_between(NSv, m - 1.96 * se, m + 1.96 * se, color=BLUES[s], alpha=0.18, lw=0)
        ax.plot(NSv, m, color=BLUES[s], lw=1.8, marker='o', ms=4.5, mec='white', mew=0.8, label=f'separation {s:g}')
        ax.axhline(full, color=BLUES[s], lw=1.0, ls=(0, (4, 3)))
    ax.set_xscale('log'); ax.set_xlabel('subsample size $n$')
    ax.set_xticks(NSv, [str(n) for n in NSv]); ax.minorticks_off()
    ax.set_title(f'rare prevalence {p:g}', loc='left')
axes[0].set_ylabel('ARI vs true clusters')
h, l = axes[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)))); l.append('full data (dashed, same color)')
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, 'fig_valleys')

# ── Representative clusterings (median-improvement replicates; see make_cluster_maps.py) ──
# Ten colours picked by palette_check.py from the Tol and Okabe-Ito palettes: every pair stays >= 8.7 apart
# (OKLab x100) under normal vision and simulated protan / deutan / tritan vision.  Clusters are numbered by
# full-data size; clusters without a full-data counterpart take the next numbers, and numbers above ten share
# one dark colour.
QUAL = ['#6250d6', '#CC3311', '#eda100', '#F0E442', '#33BBEE', '#AA4499', '#eb6834', '#882255', '#332288', '#009E73']
OTHER = '#2b2b2b'
BELOW = '#a9a8a3'                       # sub-threshold points (top row): mid grey


def aligned(ref, lab, share=0.01):
    """Number each cluster of `lab` after the full-data cluster it matches (Hungarian on the contingency table).
    Only full-data clusters holding >= `share` of the points are numbered, 1, 2, ... by size; every other cluster
    gets the next numbers and the colour OTHER.  Returns the colour of every point and the number of every cluster."""
    from scipy.optimize import linear_sum_assignment
    rk = sorted((k for k in np.unique(ref) if k >= 0 and np.sum(ref == k) >= share * len(ref)), key=lambda k: -np.sum(ref == k))
    lk = [k for k in np.unique(lab) if k >= 0]
    M = np.array([[np.sum((ref == a) & (lab == b)) for b in lk] for a in rk])
    r, c = linear_sum_assignment(-M)
    num = {lk[j]: i + 1 for i, j in zip(r, c) if M[i, j] > 0}
    extra = sorted((b for b in lk if b not in num), key=lambda b: -np.sum(lab == b))
    num.update({b: len(rk) + 1 + i for i, b in enumerate(extra)})
    col = {b: (QUAL[k - 1] if k <= len(QUAL) else OTHER) for b, k in num.items()}
    col[-1] = BELOW
    return np.array([col[x] for x in lab]), num


def panel(ax, P, lab, ref, title, size, share=0.01):
    colors, num = aligned(ref, lab)
    order = np.argsort(colors == BELOW)[::-1]          # draw sub-threshold points first
    ax.scatter(P[order, 0], P[order, 1], c=colors[order], s=size, lw=0, rasterized=True)
    import matplotlib.patheffects as pe
    for b, k in num.items():                            # number every cluster holding >= 1% of the points
        idx = np.where(lab == b)[0]
        if len(idx) < share * len(lab):
            continue
        H, ex, ey = np.histogram2d(P[idx, 0], P[idx, 1], bins=12)          # label at the densest part of the cluster
        i, j = np.unravel_index(np.argmax(H), H.shape); m = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
        x, y = P[idx[np.argmin(((P[idx] - m) ** 2).sum(1))]]
        ax.text(x, y, str(k), fontsize=6.5, ha='center', va='center', color=INK, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    ax.set_title(title, loc='left'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GRID)
    ax.set_aspect('equal', adjustable='datalim')


cm = os.path.join(RES, 'cluster_maps')


def matched(ref, lab):
    """Relabel `lab` onto the clusters of `ref` (Hungarian on the contingency table); unmatched -> new ids."""
    from scipy.optimize import linear_sum_assignment
    rk = [k for k in np.unique(ref) if k >= 0]; lk = [k for k in np.unique(lab) if k >= 0]
    M = np.array([[np.sum((ref == a) & (lab == b)) for b in lk] for a in rk])
    r, c = linear_sum_assignment(-M)
    mp = {lk[j]: rk[i] for i, j in zip(r, c)}
    nxt = max(rk) + 1
    out = np.full(len(lab), -1)
    for b in lk:
        if b not in mp:
            mp[b] = nxt; nxt += 1
        out[lab == b] = mp[b]
    return out


DISAGREE, AGREE = '#d1342f', '#e6e5e1'     # bottom row: red vs very light grey


def n_clusters(lab, share=0.01):
    k, c = np.unique(lab[lab >= 0], return_counts=True)
    return int(np.sum(c >= share * len(lab)))


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect('equal', adjustable='datalim')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GRID)


def partition_figure(P, z, size, name, truth=None, sup=None):
    """Top row: inferred predictive partitions (colours matched to the full data). Bottom row: true clusters
    (if known) and the points whose cluster differs from the full data, for the uniform subsample and the coreset."""
    full = z['full']
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0))
    for ax, key, title in zip(axes[0], ('full', 'uniform', 'coreset'), ('full data', 'uniform subsample', 'coreset')):
        k = n_clusters(z[key])
        panel(ax, P, z[key], full, f'{title}: {k} cluster' + ('s' if k != 1 else ''), size)
    if truth is not None:
        panel(axes[1, 0], P, truth, full, f'true clusters: {n_clusters(truth)}', size)
    else:
        axes[1, 0].axis('off')
    rates = {}
    for ax, key, title in ((axes[1, 1], 'uniform', 'uniform subsample'), (axes[1, 2], 'coreset', 'coreset')):
        dis = matched(full, z[key]) != full
        order = np.random.default_rng(0).permutation(len(dis))          # random drawing order
        ax.scatter(P[order, 0], P[order, 1], c=np.where(dis, DISAGREE, AGREE)[order], s=size, lw=0, rasterized=True)
        ax.set_title(f'{title}: {100 * dis.mean():.0f}% differ from full data', loc='left')
        frame(ax)
        rates[key] = dis.mean()
    handles = [plt.Line2D([], [], ls='', marker='o', ms=4, color=OTHER, label='top: clusters numbered above ten'),
               plt.Line2D([], [], ls='', marker='o', ms=4, color=BELOW, label='top: below the density threshold'),
               plt.Line2D([], [], ls='', marker='o', ms=4, color=DISAGREE, label='bottom: cluster differs from full data'),
               plt.Line2D([], [], ls='', marker='o', ms=4, mec=MUTED, mew=0.4, color=AGREE, label='bottom: same cluster as full data')]
    if sup is not None:
        ys, yc = sup
        for ax, pts in ((axes[0, 1], ys), (axes[0, 2], yc)):
            ax.scatter(pts[:, 0], pts[:, 1], marker='x', s=9, lw=0.7, color=INK, zorder=4)
        for a_, b_ in zip(ys, yc):
            axes[0, 2].plot([a_[0], b_[0]], [a_[1], b_[1]], color=INK2, lw=0.4, zorder=3)
        handles.append(plt.Line2D([], [], ls='', marker='x', ms=4, color=INK, label='top: support points'))
    if truth is None:
        axes[1, 0].legend(handles=handles, loc='center', fontsize=7.5)
        fig.tight_layout(w_pad=0.6, h_pad=1.0)
    else:
        fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6, h_pad=1.0)
    save(fig, name)
    return rates


z = np.load(os.path.join(cm, 'valley.npz'))
X = z['X']; mean = X.mean(0)
U = np.linalg.svd(X - mean, full_matrices=False)[2][:2].T
rs = partition_figure((X - mean) @ U, z, 0.6, 'fig_cluster_valley', truth=z['truth'],
                      sup=((z['y_star'] - mean) @ U, (z['y_core'] - mean) @ U))
z = np.load(os.path.join(cm, 'visium.npz'))
rv = partition_figure(z['spatial'] * np.array([1, -1]), z, 3.0, 'fig_cluster_visium')
z = np.load(os.path.join(cm, 'levine.npz'))
rl = partition_figure(z['tsne'], z, 0.25, 'fig_cluster_levine')
print(f'disagreement: simulation uniform {rs["uniform"]:.1%} coreset {rs["coreset"]:.1%} | '
      f'Visium uniform {rv["uniform"]:.1%} coreset {rv["coreset"]:.1%} | '
      f'Levine uniform {rl["uniform"]:.1%} coreset {rl["coreset"]:.1%}')

print('figures written:', sorted(os.listdir(FIG)))
