import numpy as np

from coreset.clustering import fit_clust


def _blobs(rng, d, n_per=150, sep=6.0, k=3):
    centers = rng.normal(scale=sep, size=(k, d))
    return np.vstack([c + rng.normal(size=(n_per, d)) for c in centers])


def test_levelset_label_reproduces_trajectory_labels():
    rng  = np.random.default_rng(0)
    traj = _blobs(rng, d=2)
    m    = fit_clust(traj, K_cl=3)
    assert m.backend == 'levelset'
    np.testing.assert_array_equal(m.label(traj), m.labels)


def test_levelset_minsize_merge_is_applied_to_new_points():
    rng  = np.random.default_rng(1)
    traj = _blobs(rng, d=2)
    m    = fit_clust(traj, K_cl=3, minsize=20)
    np.testing.assert_array_equal(m.label(traj), m.labels)
    sizes = np.bincount(m.labels[m.labels >= 0])
    assert sizes[sizes > 0].min() >= 20


def test_levelset_far_point_is_subthreshold():
    rng  = np.random.default_rng(2)
    traj = _blobs(rng, d=2)
    m    = fit_clust(traj, K_cl=3)
    far  = np.array([[1e3, 1e3]])
    assert m.label(far)[0] == -1


def test_high_dimensional_levelset():
    rng  = np.random.default_rng(3)
    traj = _blobs(rng, d=20, k=8, n_per=60, sep=4.0)
    m    = fit_clust(traj, K_cl=8)
    assert m.backend == 'levelset'
    np.testing.assert_array_equal(m.label(traj), m.labels)
    # well-separated blobs: each blob's core points share one label
    truth = np.repeat(np.arange(8), 60)
    core  = m.labels >= 0
    for k in range(8):
        assert len(np.unique(m.labels[core & (truth == k)])) <= 1
    assert len(np.unique(m.labels[core])) == 8
