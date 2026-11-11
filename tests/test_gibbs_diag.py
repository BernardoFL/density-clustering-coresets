import numpy as np

from inference.gibbs_diag import gibbs_diag_mixture, ess, point_partition, project
from coreset.distances import voi


def test_recovers_number_of_clusters():
    rng = np.random.default_rng(0)
    C = rng.normal(scale=6.0, size=(5, 4))
    z = rng.integers(5, size=600)
    X = C[z] + rng.normal(size=(600, 4))
    fit = gibbs_diag_mixture(X, K_max=20, n_iter=600, burn=300, thin=5, seed=1)
    occ = [len(np.unique(zz)) for zz in fit['z']]
    assert np.median(occ) == 5
    zh = point_partition(fit['z'])
    assert voi(zh, z) < 0.1
    assert voi(project(X, X, zh, fit['prior']), z) < 0.1


def test_ess_iid_and_ar1():
    rng = np.random.default_rng(0)
    assert 800 < ess(rng.normal(size=1000)) < 1300
    x = np.zeros(5000)
    for t in range(1, 5000):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    assert 100 < ess(x) < 600          # theory: n (1-ρ)/(1+ρ) ≈ 263


def test_modal_map_merges_components_of_one_mode():
    from inference.gibbs_diag import modal_map
    # three heavily overlapping components (one unimodal blob) + one far component
    w = np.array([0.3, 0.3, 0.2, 0.2])
    mu = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [10.0, 10.0]])
    s2 = np.ones((4, 2))
    cmap, modes = modal_map(w, mu, s2)
    assert len(modes) == 2
    assert cmap[0] == cmap[1] == cmap[2] != cmap[3]


def test_modal_map_keeps_separated_modes():
    from inference.gibbs_diag import modal_map
    mu = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    cmap, modes = modal_map(np.full(3, 1 / 3), mu, np.ones((3, 2)))
    assert len(modes) == 3 and len(set(cmap)) == 3


def test_overfitted_fit_of_skewed_cluster_is_one_mode():
    """A single skewed (lognormal) cluster is covered by several Gaussian components but has one mode."""
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.lognormal(0, 0.6, 800), rng.normal(size=800)])
    fit = gibbs_diag_mixture(X, K_max=15, n_iter=600, burn=300, thin=5, seed=0)
    zh = point_partition(fit['z'])
    assert len(np.unique(zh)) > 1                                     # components split the cluster
    assert len(np.unique(project(X, X, zh, fit['prior']))) == 1       # modes do not
