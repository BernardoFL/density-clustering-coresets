"""
Representative clusterings for the paper (computed once, cached, then plotted by make_figures.py).

For a representative replicate (median improvement, not the best one):
  * valley simulation (separation 3, rare prevalence 0.05, n = 50): the coreset is rebuilt from the
    replicate's seeds (deterministic), since its map was not stored;
  * Visium (d* = 40): the stored map, support and reference split are reused.
For V fresh predictive draws, every point is labelled under the full-data, the uniform-subsample and
the coreset predictive clustering (the subsample and coreset clustered with the full trajectory's
h, tau, as in the method). The representative clustering of each is the draw minimising the mean VoI
to the other draws (draws-based point estimate). Output: experiments/results/cluster_maps/*.npz
"""
import os, sys, json, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
for k, v in {'JAX_PLATFORM_NAME': 'cpu', 'OMP_NUM_THREADS': '1'}.items():
    os.environ.setdefault(k, v)
import numpy as np

RES = os.path.join(ROOT, 'experiments', 'results')
OUT = os.path.join(RES, 'cluster_maps')
V = 30


def point_estimate(draws):
    from coreset.distances import voi
    S = len(draws)
    C = np.zeros((S, S))
    for a in range(S):
        for b in range(a + 1, S):
            C[a, b] = C[b, a] = voi(draws[a], draws[b])
    return draws[int(np.argmin(C.mean(1)))]


def predictive_labels(X, XA, y, T, K, L, rng, min_share=0.0):
    from coreset.polya_urn import polya_urn
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    out = {'full': [], 'uniform': [], 'coreset': []}
    for _ in range(V):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        tN = polya_urn(XA, base, alpha=1.0, L=L, rng=rng)
        tn = polya_urn(y, base, alpha=1.0, L=L, rng=rng)
        mf = fit_clust(tN, K, min_share=min_share); hp = hp_from_model(mf, min_share=min_share)
        out['full'].append(hard_ref_labels(hp, X, tN, mf.labels, int(mf.labels.max() + 1)))
        for nm, tr in (('uniform', tn), ('coreset', T(tn))):
            lab, m = clust_shared(hp, tr)
            out[nm].append(hard_ref_labels(hp, X, tr, lab, m))
    return {k: point_estimate(v) for k, v in out.items()}


def valley_case():
    import experiments.exp_sim_valleys as E
    from coreset.algorithm import build_coreset_selected
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    sep, p_min, n = 3.0, 0.05, 50
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RES, 'sim_valleys', f's3_p0.05_n{n:04d}_rep*.json'))]
    gain = np.array([np.mean(r['eval']['ari_core']) - np.mean(r['eval']['ari_unif']) for r in runs])
    rep = runs[int(np.argsort(gain)[len(gain) // 2])]['rep']
    cell = E.SEPS.index(sep) * 100 + E.PMINS.index(p_min) * 10 + E.NS.index(n)
    s_data, s_core, s_eval = np.random.SeedSequence([E.MASTER, cell, rep]).spawn(3)
    X, lab, mu, sd, w = E.generate(sep, p_min, np.random.default_rng(s_data))
    cs = build_coreset_selected(
        data=X, n=n, niter=E.CFG['niter'], L=E.CFG['L'],
        sample_hyperprior=lambda g: sample_niw_hyperprior(X, g),
        build_base_measure=lambda p, g: build_gaussian_base(*p, g),
        N_r=E.CFG['N_r'], K_cl=E.CFG['K'], nval=E.CFG['nval'], n_steps=E.CFG['n_steps'], seed=s_core)
    A, b = np.asarray(cs['A_out']), np.asarray(cs['b_out'])
    XA = X[np.setdiff1d(np.arange(len(X)), cs['ref_idx'])]
    labs = predictive_labels(X, XA, cs['y_star'], lambda Y: Y @ A.T + b, E.CFG['K'], E.CFG['L'], np.random.default_rng(7))
    np.savez_compressed(os.path.join(OUT, 'valley.npz'), X=X, truth=lab, y_star=cs['y_star'], y_core=cs['y_coreset'],
                        rep=rep, lambda_out=cs['lambda_out'], **labs)
    print(f'valley: replicate {rep}, λ* = {cs["lambda_out"]}')


def visium_case():
    from data.visium import load_visium
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'visium', 'd040_rep*.json')))]
    rel = np.array([(r['coreset']['J_hat_I'] - r['coreset']['J_hat_bar']) / r['coreset']['J_hat_I'] for r in runs])
    r = runs[int(np.argsort(rel)[len(rel) // 2])]
    v = load_visium(os.path.join(RES, 'visium', 'visium_cache.npz'))
    cfg = r['config']
    X = np.hstack([v['X_pca'][:, :40], cfg['lambda_spatial'] * v['spatial_std']])
    c, d = r['coreset'], X.shape[1]
    A = np.eye(d) + c['lambda_out'] * (np.asarray(c['A_bar']) - np.eye(d)); b = c['lambda_out'] * np.asarray(c['b_bar'])
    y = X[np.asarray(c['support_idx'])]
    XA = X[np.setdiff1d(np.arange(len(X)), np.asarray(c['ref_idx']))]
    labs = predictive_labels(X, XA, y, lambda Y: Y @ A.T + b, cfg['K'], cfg['L'], np.random.default_rng(11),
                             min_share=cfg.get('clust_min_share', 0.0))
    np.savez_compressed(os.path.join(OUT, 'visium.npz'), spatial=v['spatial_std'], support=np.asarray(c['support_idx']),
                        rep=r['rep'], lambda_out=c['lambda_out'], **labs)
    print(f'visium: replicate {r["rep"]}, λ* = {c["lambda_out"]}')


def levine_case(n=100, n_plot=20000):
    """Levine_13dim test half; n = 100 (at n = 250 the full-data point estimate is one 83% region);
    replicate with the median decrease in the loss.  Adds a 2-D t-SNE layout of the plotted cells."""
    from experiments.exp_levine13 import half
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'levine13_test', f'test_n{n:04d}_rep*.json')))]
    rel = np.array([(r['coreset']['J_hat_I'] - r['coreset']['J_hat_bar']) / r['coreset']['J_hat_I'] for r in runs])
    r = runs[int(np.argsort(rel)[len(rel) // 2])]
    X, lab, _ = half('test', {'dataset': 'Levine_13dim'})
    c, d = r['coreset'], X.shape[1]
    A = np.eye(d) + c['lambda_out'] * (np.asarray(c['A_bar']) - np.eye(d)); b = c['lambda_out'] * np.asarray(c['b_bar'])
    sup = np.asarray(c['support_idx']); y = X[sup]
    XA = X[np.setdiff1d(np.arange(len(X)), np.asarray(c['ref_idx']))]
    E = np.random.default_rng(0).choice(np.setdiff1d(np.arange(len(X)), sup), n_plot, replace=False)
    labs = predictive_labels_at(X, X[E], XA, y, lambda Y: Y @ A.T + b, r['config']['K_cl'], r['config']['L'],
                                np.random.default_rng(13))
    np.savez_compressed(os.path.join(OUT, 'levine.npz'), X=X[E], y_star=y, y_core=y @ A.T + b, n=n,
                        rep=r['rep'], lambda_out=c['lambda_out'], **labs)
    print(f'levine: n = {n}, replicate {r["rep"]}, λ* = {c["lambda_out"]}')
    levine_tsne()



def levine_tsne():
    """2-D t-SNE layout (for display only) of the plotted Levine cells, added to levine.npz."""
    from sklearn.manifold import TSNE
    path = os.path.join(OUT, 'levine.npz')
    z = dict(np.load(path))
    if 'tsne' not in z:
        z['tsne'] = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(z['X'])
        np.savez_compressed(path, **z)


def predictive_labels_at(X, P, XA, y, T, K, L, rng):
    """As predictive_labels, but labelling only the points P (a subset of X)."""
    from coreset.polya_urn import polya_urn
    from coreset.clustering import fit_clust
    from coreset.soft_levelset import hp_from_model, clust_shared, hard_ref_labels
    from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base
    out = {'full': [], 'uniform': [], 'coreset': []}
    for _ in range(V):
        base = build_gaussian_base(*sample_niw_hyperprior(X, rng), rng)
        tN = polya_urn(XA, base, alpha=1.0, L=L, rng=rng)
        tn = polya_urn(y, base, alpha=1.0, L=L, rng=rng)
        mf = fit_clust(tN, K); hp = hp_from_model(mf)
        out['full'].append(hard_ref_labels(hp, P, tN, mf.labels, int(mf.labels.max() + 1)))
        for nm, tr in (('uniform', tn), ('coreset', T(tn))):
            lab, m = clust_shared(hp, tr)
            out[nm].append(hard_ref_labels(hp, P, tr, lab, m))
    return {k: point_estimate(v) for k, v in out.items()}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['visium', 'valley', 'levine']
    for w in which:
        globals()[f'{w}_case']()
