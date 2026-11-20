"""Receiver-visible orthogonal control design and multilinear separation.

This finite-cascade experiment assumes at most one visit to each board.
It does not replace the historical recurrent coupled-dipole model.
"""
from itertools import product
import numpy as np


def feature_matrix(controls, board_indices, patterns=None):
    controls = np.asarray(controls, dtype=complex)
    boards = tuple(board_indices.values())
    if patterns is None:
        patterns = tuple(product(*(range(len(idx)+1) for idx in boards)))
    design = np.ones((len(controls), len(patterns)), dtype=complex)
    for column, pattern in enumerate(patterns):
        for indices, choice in zip(boards, pattern):
            if choice:
                design[:, column] *= controls[:, indices[choice-1]]
    return design


def make_codebook(n_boards=4, n_elements=2, repeats=2, *, labels=None,
                  n_validation=128, validation_seed=231000001):
    if min(n_boards, n_elements, repeats) < 1:
        raise ValueError('Positive dimensions and repetitions required')
    labels = tuple(labels or tuple(chr(65+i) for i in range(n_boards)))
    if len(labels) != n_boards or len(set(labels)) != n_boards:
        raise ValueError('One distinct known label per board is required')
    groups = {name: np.arange(i*n_elements, (i+1)*n_elements)
              for i, name in enumerate(labels)}
    # Cartesian +/-1 phase states, independently controlling every element.
    base = np.asarray(tuple(product((-1., 1.), repeat=n_boards*n_elements)), complex)
    train = np.tile(base, (repeats, 1))
    rng = np.random.default_rng(validation_seed)
    validation = np.exp(1j*rng.uniform(-np.pi, np.pi,
                                      (n_validation, n_boards*n_elements)))
    patterns = tuple(product(range(n_elements+1), repeat=n_boards))
    design = feature_matrix(train, groups, patterns)
    if not np.allclose(design.conj().T@design, len(train)*np.eye(len(patterns)), atol=1e-12):
        raise AssertionError('Training columns must be orthogonal')
    return {'train_controls': train, 'validation_controls': validation,
            'board_indices': groups, 'basis_patterns': patterns,
            'train_design': design,
            'validation_design': feature_matrix(validation, groups, patterns)}


def separate(y_train, codebook):
    """Use only received channels and known controls; return canonical axes."""
    y = np.asarray(y_train, dtype=complex)
    design = np.asarray(codebook['train_design'])
    if y.ndim != 3 or y.shape[0] != len(design) or not np.isfinite(y).all():
        raise ValueError('Expected finite received channels [settings,Rx,Tx]')
    coefficients = (design.conj().T@y.reshape(len(y), -1)/len(y)).reshape(
        (design.shape[1],)+y.shape[1:])
    groups = codebook['board_indices']
    names = tuple(groups)
    index = {tuple(pattern): i for i, pattern in enumerate(codebook['basis_patterns'])}
    components = {}
    for mask in product((False, True), repeat=len(names)):
        subset = tuple(name for name, present in zip(names, mask) if present)
        if not subset:
            continue
        shape = tuple(len(groups[name]) for name in subset)
        tensor = np.empty(shape+y.shape[1:], complex)
        for cells in product(*(range(size) for size in shape)):
            selected = dict(zip(subset, cells))
            pattern = tuple(selected[name]+1 if name in selected else 0 for name in names)
            tensor[cells] = coefficients[index[pattern]]
        components[subset] = tensor
    return {'coefficients': coefficients, 'components': components,
            'background': coefficients[index[(0,)*len(names)]]}


def infer_all_components(separated, coefficient_noise_variance=0.):
    from .receiver import infer_order
    return {'>'.join(subset): infer_order(tensor, ris_labels=subset,
                                         noise_variance=coefficient_noise_variance)
            for subset, tensor in separated['components'].items() if len(subset) >= 2}
