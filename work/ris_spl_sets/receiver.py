"""Distribution-free-in-signal, Gaussian-noise order-set receiver for four boards."""

from itertools import combinations, permutations

import numpy as np
from scipy.stats import f


SHAPE = (2, 2, 2, 2, 4, 2)
RX, TX = 4, 5
PAIRS = tuple(pair for pair in combinations(range(6), 2) if pair != (RX, TX))
ORDERS = tuple(permutations(range(4)))
ALPHA = 0.05
CONTRASTS = 2048
SIZE = int(np.prod(SHAPE))
ALL_ORDERS = (1 << len(ORDERS)) - 1
ORDER_BITS = 1 << np.arange(len(ORDERS), dtype=np.uint32)
EDGE_BITS = 1 << np.arange(len(PAIRS), dtype=np.uint16)


def incidence(order):
    nodes = (TX,) + order + (RX,)
    edges = {tuple(sorted(pair)) for pair in zip(nodes[:-1], nodes[1:])}
    return [edge in edges for edge in PAIRS]


PATHS = np.asarray([incidence(order) for order in ORDERS])
PATH_MASKS = PATHS @ EDGE_BITS


def residual_shape(pair):
    m, n = (SHAPE[a] for a in pair)
    return SIZE // (m * n) * (m - 1) * (n - 1)


K = np.array([residual_shape(pair) for pair in PAIRS])
LOCAL_CRIT = K * f.isf(ALPHA / len(PAIRS), 2 * K, 2 * CONTRASTS)
BALL_CRIT = SIZE * f.isf(ALPHA, 2 * SIZE, 2 * CONTRASTS)


def receive(observed, vhat):
    """Classify a batch; only separated observations and their noise estimates enter."""
    observed = np.asarray(observed)
    vhat = np.asarray(vhat)
    if observed.shape != (len(vhat),) + SHAPE or not np.isfinite(observed).all():
        raise ValueError("Invalid observed tensor batch")
    if vhat.ndim != 1 or not np.all(np.isfinite(vhat) & (vhat > 0)):
        raise ValueError("Expected positive finite coefficient variance estimates")
    residuals = []
    for pair in PAIRS:
        m, n = (SHAPE[a] for a in pair)
        moved = np.moveaxis(observed, tuple(a + 1 for a in pair), (-2, -1))
        matrices = moved.reshape(len(vhat), -1, m, n)
        singular = np.linalg.svd(matrices, compute_uv=False)
        residuals.append(np.sum(singular[..., 1:] ** 2, axis=(1, 2)))
    residuals = np.stack(residuals, axis=-1)
    masks, certified, conflicts = [], [], []
    for critical in (LOCAL_CRIT, BALL_CRIT):
        edges = (residuals > vhat[:, None] * critical) @ EDGE_BITS
        fits = (PATH_MASKS[None, :] & edges[:, None]) == edges[:, None]
        mask = fits @ ORDER_BITS
        certified.append(edges)
        conflicts.append(mask == 0)
        masks.append(np.where(mask == 0, ALL_ORDERS, mask))
    path_scores = residuals @ PATHS.T
    forced = np.argmax(path_scores, axis=1)
    energy = np.sum(abs(observed.reshape(len(vhat), -1)) ** 2, axis=1)
    normalized = path_scores / np.maximum(energy[:, None], np.finfo(float).tiny)
    best = np.max(normalized, axis=1)
    tie = np.sum(best[:, None] - normalized <= 1e-12 * np.maximum(1, abs(best[:, None])), axis=1) > 1
    return {
        "raw_residuals": residuals,
        "candidate_masks": np.stack(masks, axis=1),
        "certified_edge_masks": np.stack(certified, axis=1),
        "conflicts": np.stack(conflicts, axis=1),
        "forced": forced.astype(np.uint8),
        "near_tie": tie,
    }
