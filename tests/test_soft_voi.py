"""Tests for the soft level-set VoI surrogate (coreset/soft_levelset.py)."""

import numpy as np
import pytest
from jax.experimental import enable_x64

from coreset.clustering import fit_clust
from coreset.polya_urn import polya_urn
from coreset.soft_levelset import (
    hp_from_model, clust_shared, hard_ref_labels, hard_criterion, soft_voi_one,
    optimize_soft_levelset,
)
from inference.hyperpriors import sample_niw_hyperprior, build_gaussian_base

MODES = [('singletons', True), ('cell', False)]


@pytest.fixture(scope='module')
def setup():
    rng = np.random.default_rng(1)
    d = 3
    centers = rng.normal(scale=5.0, size=(3, d))
    data = np.vstack([c + rng.normal(size=(300, d)) for c in centers])
    perm = rng.permutation(len(data))
    ref, rest = data[perm[:300]], data[perm[300:]]
    y_star = rest[rng.choice(len(rest), 40, replace=False)]
    phi = sample_niw_hyperprior(data, rng)
    base = build_gaussian_base(*phi, rng)
    YN = polya_urn(rest, base, alpha=1.0, L=200, rng=rng)
    Yn = polya_urn(y_star, base, alpha=1.0, L=200, rng=rng)
    mf = fit_clust(YN, 3)
    hp = hp_from_model(mf)
    kN = int(mf.labels.max() + 1)
    sN = hard_ref_labels(hp, ref, YN, mf.labels, kN)
    return dict(d=d, ref=ref, YN=YN, Yn=Yn, mf=mf, hp=hp, kN=kN, sN=sN, rng=rng)


def _soft(S, A, b, Yn, lab, m, sN, kappa, beta, singletons, M_pad=None, X=None, tau=None, grad=False):
    import jax
    M_pad = M_pad or max(m, 1)
    X = S['ref'] if X is None else X
    hp = S['hp']
    lab_p = np.where(lab >= 0, lab, M_pad)
    args = (Yn, hp['mean'], hp['s'], hp['h'], hp['tau'] if tau is None else tau, lab_p, sN, X, kappa, beta)

    def f(A_, b_):
        return soft_voi_one(A_, b_, *args, M_pad=M_pad, KN_pad=S['kN'], singletons=singletons)
    with enable_x64():
        A, b = np.asarray(A, np.float64), np.asarray(b, np.float64)
        if grad:
            v, (gA, gb) = jax.value_and_grad(f, argnums=(0, 1))(A, b)
            return float(v), np.asarray(gA), np.asarray(gb)
        return float(f(A, b))


def _transforms(S, k=5):
    rng = np.random.default_rng(7)
    d = S['d']
    out = [(np.eye(d), np.zeros(d))]
    for _ in range(k):
        out.append((np.eye(d) + 0.05 * rng.normal(size=(d, d)), 0.2 * rng.normal(size=d)))
    return out


def test_agreement_on_trajectory_points(setup):
    S = setup
    rate = np.mean(hard_ref_labels(S['hp'], S['YN'], S['YN'], S['mf'].labels, S['kN']) == S['mf'].labels)
    print(f'\nhard_ref_labels vs Clust agreement on trajectory points: {rate:.3f}'
          + ('   (FLAG: < 0.95)' if rate < 0.95 else ''))
    lab, m = clust_shared(S['hp'], S['YN'])          # shared-hyperparameter Clust reproduces fit_clust
    assert m == S['kN']
    assert hard_criterion(lab, S['mf'].labels, 'cell') < 1e-12


@pytest.mark.parametrize('mode,singletons', MODES)
def test_hard_limit(setup, mode, singletons):
    S = setup
    for A, b in _transforms(S):
        Y = S['Yn'] @ A.T + b
        lab, m = clust_shared(S['hp'], Y)
        s, lf = hard_ref_labels(S['hp'], S['ref'], Y, lab, m, return_lf=True)
        hard = hard_criterion(S['sN'], s, mode)
        soft = _soft(S, A, b, S['Yn'], lab, m, S['sN'], 1e-4, 1e-4, singletons)
        near = np.sum(np.abs(lf - S['hp']['tau']) < 1e-2)
        assert abs(soft - hard) < 1e-6, f'soft {soft} hard {hard}; {near} points within 1e-2 of τ'


@pytest.mark.parametrize('mode,singletons', MODES)
def test_zero_when_coreset_is_full_trajectory(setup, mode, singletons):
    S = setup
    lab, m = clust_shared(S['hp'], S['YN'])
    d = S['d']
    assert hard_criterion(S['sN'], hard_ref_labels(S['hp'], S['ref'], S['YN'], lab, m), mode) < 1e-12
    assert abs(_soft(S, np.eye(d), np.zeros(d), S['YN'], lab, m, S['sN'], 1e-4, 1e-4, singletons)) < 1e-6


@pytest.mark.parametrize('mode,singletons', MODES)
def test_label_invariance(setup, mode, singletons):
    S = setup
    A, b = _transforms(S)[1]
    lab, m = clust_shared(S['hp'], S['Yn'] @ A.T + b)
    base = _soft(S, A, b, S['Yn'], lab, m, S['sN'], 0.5, 0.5, singletons)
    rng = np.random.default_rng(3)
    pz, pn = rng.permutation(m), rng.permutation(S['kN'])
    lab2 = np.where(lab >= 0, pz[np.maximum(lab, 0)], -1)
    sN2 = np.where(S['sN'] >= 0, pn[np.maximum(S['sN'], 0)], -1)
    assert abs(_soft(S, A, b, S['Yn'], lab2, m, sN2, 0.5, 0.5, singletons) - base) < 1e-10


@pytest.mark.parametrize('mode,singletons', MODES)
def test_gradient_matches_finite_differences(setup, mode, singletons):
    S = setup
    d = S['d']
    A, b = _transforms(S)[2]
    lab, m = clust_shared(S['hp'], S['Yn'] @ A.T + b)          # frozen topology
    _, gA, gb = _soft(S, A, b, S['Yn'], lab, m, S['sN'], 0.5, 0.5, singletons, grad=True)
    g_ad = np.concatenate([gA.ravel(), gb])
    eps, g_fd = 1e-6, np.zeros(d * d + d)
    for i in range(d * d + d):
        e = np.zeros(d * d + d); e[i] = eps
        vals = []
        for sgn in (1, -1):
            th = np.concatenate([A.ravel(), b]) + sgn * e
            vals.append(_soft(S, th[:d * d].reshape(d, d), th[d * d:], S['Yn'], lab, m, S['sN'], 0.5, 0.5, singletons))
        g_fd[i] = (vals[0] - vals[1]) / (2 * eps)
    rel = np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd)
    assert rel < 1e-4, f'relative error {rel:.2e}'


@pytest.mark.parametrize('mode,singletons', MODES)
def test_no_nan_gradients_in_edge_cases(setup, mode, singletons):
    S = setup
    d = S['d']
    A, b = np.eye(d), np.zeros(d)
    lab, m = clust_shared(S['hp'], S['Yn'])
    cases = {
        'empty component': (np.where(lab >= 1, lab + 1, lab), m + 1, None),
        'm = 0': (np.full_like(lab, -1), 0, None),
        'all reference points sub-threshold': (lab, m, 1e6),
    }
    for name, (lab_c, m_c, tau) in cases.items():
        v, gA, gb = _soft(S, A, b, S['Yn'], lab_c, m_c, S['sN'], 0.5, 0.5, singletons,
                          M_pad=max(m_c, 1) + 1, tau=tau, grad=True)
        assert np.isfinite(v) and np.all(np.isfinite(gA)) and np.all(np.isfinite(gb)), name


def test_optimizer_runs_and_checkpoint_is_best(setup):
    S = setup
    rng = np.random.default_rng(0)
    data = np.vstack([S['ref'], S['YN']])
    trajs_full, trajs_core = [], []
    for _ in range(3):
        phi = sample_niw_hyperprior(data, rng)
        base = build_gaussian_base(*phi, rng)
        trajs_full.append(polya_urn(S['YN'], base, alpha=1.0, L=150, rng=rng))
        trajs_core.append(polya_urn(S['Yn'][:40], base, alpha=1.0, L=150, rng=rng))
    res = optimize_soft_levelset(trajs_full, trajs_core, S['ref'], 3, rng=rng, n_steps=12, B=64,
                                 iter_batch=2, eval_every=4, progress=False)
    log = res['log']
    assert log['eval_step'][0] == 0 and log['checkpoint_F'] == min(log['hard_F'])
    assert log['checkpoint_F'] <= log['F_identity']
    assert np.all(np.isfinite(log['soft_loss']))


@pytest.mark.parametrize('mode,singletons', MODES)
def test_weighted_hard_limit(setup, mode, singletons):
    """w_merge != 1: soft value still equals the weighted hard criterion as κ, β → 0."""
    S = setup
    hp = S['hp']
    for A, b in _transforms(S, 2):
        Y = S['Yn'] @ A.T + b
        lab, m = clust_shared(hp, Y)
        s = hard_ref_labels(hp, S['ref'], Y, lab, m)
        hard = hard_criterion(S['sN'], s, mode, w_merge=3.0)
        M_pad = max(m, 1)
        with enable_x64():
            soft = float(soft_voi_one(np.asarray(A, float), np.asarray(b, float), S['Yn'], hp['mean'], hp['s'],
                                      hp['h'], hp['tau'], np.where(lab >= 0, lab, M_pad), S['sN'], S['ref'],
                                      1e-4, 1e-4, M_pad, S['kN'], singletons, 3.0))
        assert abs(soft - hard) < 1e-6


def test_weighting_penalises_merging():
    full = np.repeat([0, 1, 2, 3], 25)
    merged = np.repeat([0, 0, 2, 3], 25)          # two full clusters merged
    split = np.repeat([0, 1, 2, 3], 25) * 2 + np.tile([0, 1], 50)   # each cluster split in two
    assert hard_criterion(full, merged, w_merge=1.0) == pytest.approx(hard_criterion(full, merged))
    assert hard_criterion(full, merged, w_merge=3.0) > 2.5 * hard_criterion(full, merged, w_merge=1.0)
    assert hard_criterion(full, split, w_merge=3.0) == pytest.approx(hard_criterion(full, split, w_merge=1.0))
