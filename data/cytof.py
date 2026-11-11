"""
Levine_13dim mass cytometry benchmark (Weber & Robinson 2016), exported from the
Bioconductor package HDCytoData by data/export_cytof.R.

load_cytof(name) returns the arcsinh(x / 5) transformed cell-type markers, the
expert-gated population labels (−1 = unassigned, i.e. outside every gate) and the
population names, cached as .npz next to the CSV.

split_cells(N, seed) gives the fixed development / test split of the cells:
designs are tuned on the development half only, results are reported on the
test half.
"""

import os

import numpy as np

HERE = os.path.join(os.path.dirname(__file__), 'cytof')
COFACTOR = 5.0
SPLIT_SEED = 20261013          # fixed once, before any coreset was run on these data


def load_cytof(name: str = 'Levine_13dim') -> dict:
    cache = os.path.join(HERE, f'{name}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, f'{name}.csv.gz'))
    pop = df.pop('population').astype(str).to_numpy()
    names = np.array(sorted(set(pop) - {'unassigned'}), dtype=str)
    lab = np.full(len(pop), -1, dtype=int)
    for k, nm in enumerate(names):
        lab[pop == nm] = k
    X = np.arcsinh(df.to_numpy(dtype=float) / COFACTOR)
    out = {'X': X, 'labels': lab, 'names': names, 'markers': np.array(df.columns, dtype=str)}
    np.savez_compressed(cache, **out)
    return out


def split_cells(N: int, seed: int = SPLIT_SEED):
    """Fixed random halves: (development indices, test indices)."""
    perm = np.random.default_rng(seed).permutation(N)
    return np.sort(perm[: N // 2]), np.sort(perm[N // 2:])
