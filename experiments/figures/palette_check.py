"""
Colour-vision-deficiency check for the categorical palette of the cluster maps.

For every pair of colours: perceptual distance (OKLab, x100) under normal vision and under
simulated protanopia, deuteranopia and tritanopia (Machado, Oliveira & Fernandes 2009, severity 1).
`pick(pool, k)` chooses k colours greedily to maximise the worst pairwise distance.
"""
import itertools
import numpy as np

MACHADO = {
    'protan': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    'deutan': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    'tritan': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def _lin(h):
    c = np.array([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    rgb = np.clip(rgb, 0, 1)
    l, m, s = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                [0.2119034982, 0.6806995451, 0.1073969566],
                                [0.0883024619, 0.2817188376, 0.6299787005]]) @ rgb)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ np.array([l, m, s])


def dist(a, b):
    """Worst-case distance (x100) over normal vision and the three simulated deficiencies."""
    A, B = _lin(a), _lin(b)
    d = [np.linalg.norm(_oklab(A) - _oklab(B))]
    for M in MACHADO.values():
        d.append(np.linalg.norm(_oklab(M @ A) - _oklab(M @ B)))
    return 100 * min(d)


def report(cols, others=()):
    pairs = [(dist(a, b), a, b) for a, b in itertools.combinations(cols, 2)]
    vs = [(dist(a, o), a, o) for a in cols for o in others]
    return min(pairs), (min(vs) if vs else None)


def pick(pool, k, avoid=(), start=None):
    chosen = list(start or [])
    while len(chosen) < k:
        best = max((c for c in pool if c not in chosen),
                   key=lambda c: min([dist(c, x) for x in chosen + list(avoid)] or [1e9]))
        chosen.append(best)
    return chosen


POOL = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499',   # Tol muted
        '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',                                    # Tol bright
        '#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988',                                    # Tol vibrant
        '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7',                         # Okabe-Ito
        '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']              # reference

if __name__ == '__main__':
    avoid = ['#a9a8a3', '#ffffff', '#2b2b2b']          # below-threshold grey, background, "other" cluster
    for k in (6, 8, 10, 12):
        cols = pick(POOL, k, avoid)
        (dmin, a, b), (vmin, c, o) = report(cols, avoid)
        print(f'k={k:2d}  worst pair {dmin:5.1f} ({a} vs {b})  worst vs grey/white/other {vmin:5.1f}  {cols}')
