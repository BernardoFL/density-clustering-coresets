import numpy as np

from coreset.distances import voi, voi_ref, singletons_for_unassigned


def _voi_direct(a, b):
    """Eq. (3) from an explicit joint-count loop, natural log."""
    n  = len(a)
    ua, ub = sorted(set(a)), sorted(set(b))
    out = 0.0
    for i in ua:
        p_i = np.sum(a == i) / n
        for j in ub:
            q_j  = np.sum(b == j) / n
            r_ij = np.sum((a == i) & (b == j)) / n
            if r_ij > 0:
                out -= r_ij * (np.log(r_ij / p_i) + np.log(r_ij / q_j))
    return out


def _singletons_direct(s):
    """Reference implementation: each −1 becomes a fresh label."""
    s = list(s)
    nxt = max([x for x in s if x >= 0], default=-1) + 1
    out = []
    for x in s:
        if x < 0:
            out.append(nxt)
            nxt += 1
        else:
            out.append(x)
    return np.array(out)


def test_zero_on_identical():
    s = np.random.default_rng(0).integers(-1, 5, size=500)
    assert voi_ref(s, s) == 0.0


def test_relabelling_invariance_of_clusters():
    rng = np.random.default_rng(1)
    s   = rng.integers(-1, 5, size=500)
    t   = rng.integers(-1, 4, size=500)
    perm = {-1: -1, 0: 3, 1: 4, 2: 0, 3: 2, 4: 1}       # clusters permuted, −1 kept
    s_perm = np.array([perm[x] for x in s])
    assert np.isclose(voi_ref(s, t), voi_ref(s_perm, t))
    assert np.isclose(voi_ref(s, t), voi_ref(t, s))


def test_matches_direct_formula_with_singletons():
    rng = np.random.default_rng(2)
    for _ in range(5):
        s = rng.integers(-1, 6, size=400)
        t = rng.integers(-1, 3, size=400)
        assert np.isclose(voi_ref(s, t), _voi_direct(_singletons_direct(s), _singletons_direct(t)),
                          atol=1e-10)


def test_all_unassigned_is_penalised_not_rewarded():
    """Labelling every reference point −1 must not beat a real, noisy partition."""
    rng = np.random.default_rng(3)
    s_full  = rng.integers(0, 6, size=1000)
    noisy   = np.where(rng.uniform(size=1000) < 0.7, s_full, rng.integers(0, 6, size=1000))
    nothing = np.full(1000, -1)
    assert voi_ref(s_full, nothing) == voi(s_full, np.arange(1000))   # every point a singleton
    assert voi_ref(s_full, nothing) > voi_ref(s_full, noisy)
    # under the old shared-cell convention the degenerate labelling won:
    assert voi(s_full, nothing) < voi(s_full, noisy)


def test_singletons_helper_keeps_clusters():
    s = np.array([2, -1, 0, -1, 2])
    out = singletons_for_unassigned(s)
    assert list(out[[0, 2, 4]]) == [2, 0, 2]
    assert len(set(out[[1, 3]])) == 2 and min(out[[1, 3]]) > 2
