import numpy as np


def voi(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Variation of Information between two partitions (in nats).
    VoI(P, Q) = H(P) + H(Q) - 2 * I(P; Q)
    """
    L = len(labels_a)

    # Contiguous integer re-encoding via return_inverse (no Python loop)
    _, a_inv, counts_a = np.unique(labels_a, return_inverse=True, return_counts=True)
    _, b_inv, counts_b = np.unique(labels_b, return_inverse=True, return_counts=True)

    pa = counts_a / L
    pb = counts_b / L

    H_a = -np.sum(pa * np.log(pa + 1e-300))
    H_b = -np.sum(pb * np.log(pb + 1e-300))

    # Joint distribution: vectorised accumulation (replaces Python loop over L)
    Ka, Kb = len(counts_a), len(counts_b)
    joint = np.zeros((Ka, Kb))
    np.add.at(joint, (a_inv, b_inv), 1.0 / L)

    # Mutual information
    with np.errstate(divide='ignore', invalid='ignore'):
        outer = np.outer(pa, pb)
        ratio = np.where(joint > 0, joint / (outer + 1e-300), 0.0)
        MI = np.sum(np.where(joint > 0, joint * np.log(ratio + 1e-300), 0.0))

    return max(0.0, H_a + H_b - 2.0 * MI)


def singletons_for_unassigned(labels: np.ndarray) -> np.ndarray:
    """Give every sub-threshold point (label −1) its own singleton cluster."""
    labels = np.asarray(labels).copy()
    neg = labels < 0
    if neg.any():
        start = labels.max() + 1 if (~neg).any() else 0
        labels[neg] = start + np.arange(neg.sum())
    return labels


def voi_ref(labels_s: np.ndarray, labels_s_prime: np.ndarray) -> float:
    """
    VoI_ref(s, s'): VoI (nats) between two partitions of the N_r reference points.

    Both inputs are label vectors on the same reference set, produced by
    ClustModel.label.  A sub-threshold point (label −1) lies outside every
    cluster of the level set, so it is co-clustered with nothing: each one is
    its own singleton.  Treating −1 as one shared cell instead makes "label
    every reference point −1" a constant partition with VoI equal to H(s),
    which a map that moves the coreset away from the data attains trivially.
    """
    labels_s       = np.asarray(labels_s)
    labels_s_prime = np.asarray(labels_s_prime)
    assert labels_s.shape == labels_s_prime.shape, 'label vectors must match in length'
    return voi(singletons_for_unassigned(labels_s), singletons_for_unassigned(labels_s_prime))
