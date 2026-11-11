"""
10x Visium H&E mouse brain (squidpy), preprocessed with scanpy: normalize_total(1e4) → log1p →
top-2000 HVGs (seurat) → scale(max_value=10) → PCA(100).

The processed arrays are cached to an .npz so every replicate (and every
worker process) uses the same PCA.
"""

import os

import numpy as np

N_EXPECTED = 2688


def _preprocess(max_d: int):
    import scanpy as sc
    import squidpy as sq
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    adata = sq.datasets.visium_hne_adata()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=max_d)

    label_col = next(c for c in ['cluster', 'leiden'] if c in adata.obs.columns)
    leiden    = LabelEncoder().fit_transform(adata.obs[label_col].values)
    spatial   = np.array(adata.obsm['spatial'], dtype=float)
    X_hvg     = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                           dtype=np.float32)
    return {
        'X_pca':        np.asarray(adata.obsm['X_pca'], dtype=float),
        'pca_variance': np.asarray(adata.uns['pca']['variance'], dtype=float),
        'pca_var_ratio': np.asarray(adata.uns['pca']['variance_ratio'], dtype=float),
        'X_hvg':        X_hvg,
        'spatial':      spatial,
        'spatial_std':  StandardScaler().fit_transform(spatial),
        'leiden':       leiden,
    }


def load_visium(cache_path: str, max_d: int = 100) -> dict:
    """Load the preprocessed Visium arrays, computing and caching them on first use."""
    if not os.path.exists(cache_path):
        arrays = _preprocess(max_d)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **arrays)
    with np.load(cache_path) as f:
        out = {k: f[k] for k in f.files}
    N = len(out['X_pca'])
    assert N == N_EXPECTED, f'expected {N_EXPECTED} spots, got {N}'
    return out
