import numpy as np

from coreset.algorithm import build_coreset_selected
from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base


def _run(data, **kw):
    def sample_hp(rng_):       return sample_niw_hyperprior(data, rng_)
    def build_base(phi, rng_): return build_gaussian_base(*phi, rng_)
    return build_coreset_selected(
        data=data, n=20, niter=2, L=80,
        sample_hyperprior=sample_hp, build_base_measure=build_base,
        N_r=100, K_cl=3, nval=6, seed=0, **kw,
    )


def _data(d):
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=6.0, size=(3, d))
    return np.vstack([c + rng.normal(size=(150, d)) for c in centers])


def test_identity_transform_gives_zero_paired_difference():
    for d in (2, 6):
        data = _data(d)
        out  = _run(data, A_bar_override=np.eye(d), b_bar_override=np.zeros(d))
        np.testing.assert_array_equal(out['V_bar'], out['V_I'])
        assert np.all(out['select_curve'] == out['select_curve'][0])   # every λ identical
        assert out['lambda_out'] == 0.0 and out['selected'] == 'I'      # ties go to λ = 0


def test_far_shift_is_not_taken_in_full():
    """A transform that moves the coreset far from the data must not be chosen at λ = 1."""
    data = _data(2)
    out  = _run(data, A_bar_override=np.eye(2), b_bar_override=np.array([200.0, 200.0]))
    assert out['lambda_out'] < 1.0
    assert out['select_curve'][-1] > out['select_curve'][0]


def test_output_is_on_the_path():
    data = _data(2)
    A, b = np.array([[1.5, 0.0], [0.3, 0.8]]), np.array([0.5, -0.2])
    out  = _run(data, A_bar_override=A, b_bar_override=b)
    lam  = out['lambda_out']
    np.testing.assert_allclose(out['A_out'], np.eye(2) + lam * (A - np.eye(2)))
    np.testing.assert_allclose(out['b_out'], lam * b)
    assert out['config']['nval_select'] + out['config']['nval_eval'] == 6


def test_split_and_support():
    data = _data(2)
    out  = _run(data, A_bar_override=np.eye(2), b_bar_override=np.zeros(2))
    N    = len(data)
    ref  = set(out['ref_idx'].tolist())
    assert len(ref) == 100
    A    = set(range(N)) - ref
    assert set(out['support_idx'].tolist()) <= A
    assert len(set(out['support_idx'].tolist())) == 20
