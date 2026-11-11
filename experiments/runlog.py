"""
Seeds and per-run JSON logs.

Every run gets one SeedSequence([master_seed, run_idx]) whose children are
named, so each component (data, coreset, NUTS fits, ...) is reproducible on
its own regardless of the order in which the run calls them.
"""

import glob
import json
import os

import numpy as np


def child_seeds(master_seed: int, run_idx: int, names: list) -> dict:
    """Return {name: SeedSequence} spawned from SeedSequence([master_seed, run_idx])."""
    root = np.random.SeedSequence([master_seed, run_idx])
    return dict(zip(names, root.spawn(len(names))))


def seed_int(ss: np.random.SeedSequence) -> int:
    """A 31-bit integer seed derived from a SeedSequence (for sklearn / JAX)."""
    return int(ss.generate_state(1)[0] % (2 ** 31))


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.random.SeedSequence):
        return {'entropy': str(obj.entropy), 'spawn_key': list(obj.spawn_key)}
    return obj


def save_json(path: str, record: dict):
    """Write a run record atomically (tmp file + rename), converting numpy types."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_to_jsonable(record), f)
    os.replace(tmp, path)


def load_runs(directory: str, pattern: str = '*.json') -> list:
    """Load every run record in a directory, sorted by file name."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def parse_range(spec: str, n_max: int = None) -> list:
    """'0-19' -> [0..19]; '3,5,7' -> [3,5,7]; 'all' -> range(n_max); '' -> []."""
    if spec in (None, '', 'none'):
        return []
    if spec == 'all':
        return list(range(n_max))
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
