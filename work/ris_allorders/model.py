"""Evaluation-side hidden-order finite-cascade simulation, not full-wave physics.

All arrays/factors are drawn in canonical board order before using the hidden
permutation. Each subset has one order induced by the hidden global order.
The classifier receives no object returned by this module.
"""
from itertools import product
import numpy as np


def complex_normal(rng, shape, scale=1.):
    return scale*(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2.)


def chain_tensor(factors, sequence, canonical_labels):
    """Independent einsum product, with canonical board axes then Rx,Tx."""
    order = len(sequence)
    canonical = tuple(name for name in canonical_labels if name in sequence)
    axis = {name: i for i, name in enumerate(canonical)}
    operands = [factors['R'][sequence[0]], [axis[sequence[0]], order+1]]
    for before, after in zip(sequence[:-1], sequence[1:]):
        operands.extend((factors['B'][(after, before)], [axis[after], axis[before]]))
    operands.extend((factors['L'][sequence[-1]], [order, axis[sequence[-1]]],
                     list(range(order))+[order, order+1]))
    return np.einsum(*operands, optimize=True)


def make_trial_system(rng, sequence, n_elements=2, n_tx=2, n_rx=4,
                      mode='generic', hop_gain=.6, labels=tuple('ABCD')):
    labels, sequence = tuple(labels), tuple(sequence)
    if sorted(labels) != sorted(sequence):
        raise ValueError('Hidden order must visit every named board once')
    if mode == 'siso':
        n_tx = n_rx = 1
    factors = {'H0': complex_normal(rng, (n_rx, n_tx), .3), 'L': {}, 'R': {}, 'B': {}}
    for name in labels:
        factors['R'][name] = complex_normal(rng, (n_elements, n_tx), 1/np.sqrt(n_elements))
        factors['L'][name] = complex_normal(rng, (n_rx, n_elements), 1/np.sqrt(n_elements))
    for after in labels:
        for before in labels:
            if after == before:
                continue
            full = complex_normal(rng, (n_elements, n_elements), hop_gain/np.sqrt(n_elements))
            if mode in ('rank1', 'near_rank1'):
                # Fixed draw schedule for every hidden class, independent of class.
                u = complex_normal(rng, (n_elements,))
                v = complex_normal(rng, (n_elements,))
                rank1 = hop_gain*np.outer(u, v)/np.sqrt(n_elements)
                full = rank1 if mode == 'rank1' else rank1+.03*full
            factors['B'][(after, before)] = full
    patterns = tuple(product(range(n_elements+1), repeat=len(labels)))
    coefficients = np.zeros((len(patterns), n_rx, n_tx), complex)
    index = {pattern: i for i, pattern in enumerate(patterns)}
    coefficients[index[(0,)*len(labels)]] = factors['H0']
    truths, truth_orders = {}, {}
    for mask in product((False, True), repeat=len(labels)):
        subset = tuple(name for name, present in zip(labels, mask) if present)
        if not subset:
            continue
        order = tuple(name for name in sequence if name in subset)
        tensor = chain_tensor(factors, order, labels)
        truths[subset] = tensor
        truth_orders[subset] = order
        for cells in product(range(n_elements), repeat=len(subset)):
            chosen = dict(zip(subset, cells))
            pattern = tuple(chosen[name]+1 if name in chosen else 0 for name in labels)
            coefficients[index[pattern]] = tensor[cells]
    return {'coefficients': coefficients, 'truth_components': truths,
            'truth_orders': truth_orders, 'factors': factors,
            'sequence': sequence, 'mode': mode, 'hop_gain': hop_gain,
            'interpretation': 'One directed finite-cascade path per subset; no recurrent or coexisting same-subset order field.'}


def direct_response(factors, sequence, controls, board_indices):
    """Independent forward sum over all nonempty subsets for implementation QA."""
    sequence = tuple(sequence)
    result = np.broadcast_to(factors['H0'], (len(controls),)+factors['H0'].shape).copy()
    for mask in product((False, True), repeat=len(sequence)):
        path = tuple(name for name, present in zip(sequence, mask) if present)
        if not path:
            continue
        current = controls[:, board_indices[path[0]], None]*factors['R'][path[0]]
        for before, after in zip(path[:-1], path[1:]):
            current = controls[:, board_indices[after], None]*(factors['B'][(after, before)]@current)
        result += factors['L'][path[-1]]@current
    return result
