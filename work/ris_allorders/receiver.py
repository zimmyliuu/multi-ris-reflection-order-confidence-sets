"""Blind single-chain ordering from a separated RIS coefficient tensor.

The only input carrying channel information is an observed coefficient tensor,
with canonical named RIS-cell axes followed by Rx and Tx.  Neither the true
sequence nor channel factors, geometry, simulator metadata, or lower-order
coefficients enter this receiver.

For the finite single-chain model, tensor entries factor as pairwise potentials
on Tx--first RIS--...--last RIS--Rx.  Fixing all but two axes gives a rank-one
matrix when those two axes are not adjacent.  An adjacent pair has rank greater
than one generically.  Summed best-rank-one residuals recover the path when all
its edges are nondegenerate.  This condition excludes scalar/rank-one channels
that can erase order information.  It does not cover a sum of coexisting chains
with the same participating set or a full recurrent scattering model.

The winner margin and noise metadata are diagnostics, not calibrated
confidence probabilities.  Candidate enumeration is factorial in RIS count;
the pairwise diagnostic itself does not depend on enumerating permutations.
"""

from itertools import combinations, permutations

import numpy as np


ALGORITHM_VERSION = "conditional-rank-chain-v1"
NUMERICAL_TIE_TOLERANCE = 1e-12
LOW_RELATIVE_MARGIN = 0.05
NEAR_RANK_ONE_EDGE_SCORE = 1e-4


def _validate(coefficient_tensor, ris_labels, noise_variance):
    tensor = np.asarray(coefficient_tensor, dtype=np.complex128)
    if tensor.ndim < 3:
        raise ValueError("Expected one or more RIS-cell axes followed by Rx, Tx")
    if any(size < 1 for size in tensor.shape):
        raise ValueError("Every tensor axis must be nonempty")
    if not np.isfinite(tensor).all():
        raise ValueError("The coefficient tensor must be finite")
    count = tensor.ndim - 2
    labels = tuple(chr(ord("A") + i) for i in range(count)) if ris_labels is None else tuple(ris_labels)
    if len(labels) != count or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Use one nonempty string label per RIS axis")
    if len(set(labels)) != len(labels) or {"Rx", "Tx"}.intersection(labels):
        raise ValueError("RIS labels must be distinct and must not be Rx or Tx")
    variance = float(noise_variance)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    return tensor, labels, variance


def conditional_pair_diagnostics(coefficient_tensor, *, ris_labels=None, noise_variance=0.0):
    """Measure adjacency evidence for every pair of tensor axes.

    ``noise_variance`` is E[|complex coefficient error|^2], obtained from a
    receiver-side acquisition/noise model when available.  It is not an SNR
    calculated using noiseless simulator coefficients.  The value is recorded
    for analysis only; it does not correct or modify the ordering scores.

    Raw singular values use the original tensor scale.  All outputs contain
    only Python scalars/lists/dictionaries and are directly JSON serializable.
    The slice conditioning-index order is recorded for reproducible analysis.
    """
    tensor, labels, variance = _validate(coefficient_tensor, ris_labels, noise_variance)
    nodes = labels + ("Rx", "Tx")
    total_energy = float(np.vdot(tensor.ravel(), tensor.ravel()).real)
    tiny = np.finfo(float).tiny
    expected_noise_energy = float(tensor.size * variance)
    scores = np.zeros((tensor.ndim, tensor.ndim), dtype=float)
    diagnostics = []
    for first, second in combinations(range(tensor.ndim), 2):
        conditioning_axes = tuple(axis for axis in range(tensor.ndim) if axis not in (first, second))
        conditioning_shape = tuple(tensor.shape[axis] for axis in conditioning_axes)
        slices = tensor.transpose(conditioning_axes + (first, second)).reshape(
            -1, tensor.shape[first], tensor.shape[second]
        )
        singular = np.linalg.svd(slices, compute_uv=False)
        slice_energy = np.sum(singular ** 2, axis=-1)
        slice_residual = np.sum(singular[:, 1:] ** 2, axis=-1)
        residual = float(np.sum(slice_residual))
        raw_score = residual / max(total_energy, tiny)
        score = raw_score
        scores[first, second] = scores[second, first] = score
        if singular.shape[1] > 1:
            ratios = np.divide(singular[:, 1], singular[:, 0],
                               out=np.zeros(len(slices)), where=singular[:, 0] > tiny)
        else:
            ratios = np.zeros(len(slices))
        diagnostics.append({
            "nodes": [nodes[first], nodes[second]],
            "axes": [first, second],
            "matrix_shape": [tensor.shape[first], tensor.shape[second]],
            "conditioning_nodes": [nodes[axis] for axis in conditioning_axes],
            "conditioning_shape": list(conditioning_shape),
            "slice_count": len(slices),
            "singular_values": singular.tolist(),
            "second_to_first_singular_ratio": ratios.tolist(),
            "slice_energy": slice_energy.tolist(),
            "slice_rank_one_residual": slice_residual.tolist(),
            "raw_residual_energy": residual,
            "raw_edge_score": raw_score,
            "edge_score": score,
        })
    return {
        "algorithm": ALGORITHM_VERSION,
        "ris_labels": list(labels),
        "nodes": list(nodes),
        "tensor_shape": list(tensor.shape),
        "tensor_energy": total_energy,
        "coefficient_noise_variance": variance,
        "expected_noise_energy": expected_noise_energy,
        "noise_variance_used_for_scoring": False,
        "pairwise_scores": scores.tolist(),
        "pair_scores": scores.tolist(),
        "singular_diagnostics": diagnostics,
    }


def infer_order(coefficient_tensor, *, ris_labels=None, noise_variance=0.0):
    """Infer a complete Tx-to-Rx RIS order without known endpoint identities.

    Each candidate contains every supplied RIS exactly once.  Scores sum edge
    evidence along Tx--candidate--Rx.  No candidate-specific channel fitting or
    access to simulator parameters is performed.  All candidate scores and
    conditional singular values are returned, not just the selected label.

    ``selected_order`` is None for a numerical tie or zero/undetectable signal;
    ``argmax_order`` always records the first maximizer for auditing.  Low
    margins and weak edges remain flagged on otherwise unique outputs.  The
    thresholds are fixed diagnostics, not a calibrated acceptance gate.
    """
    output = conditional_pair_diagnostics(
        coefficient_tensor, ris_labels=ris_labels, noise_variance=noise_variance
    )
    labels = tuple(output["ris_labels"])
    count = len(labels)
    rx, tx = count, count + 1
    scores = np.asarray(output["pairwise_scores"])
    total_pair_score = float(np.triu(scores, 1).sum())
    candidates = []
    for order_indices in permutations(range(count)):
        path = (tx,) + order_indices + (rx,)
        edges = list(zip(path[:-1], path[1:]))
        edge_scores = [float(scores[first, second]) for first, second in edges]
        candidates.append({
            "order": [labels[index] for index in order_indices],
            "path": [output["nodes"][index] for index in path],
            "score": float(sum(edge_scores)),
            "edge_scores": edge_scores,
            "off_path_score": max(0.0, total_pair_score - sum(edge_scores)),
        })
    ranked = sorted(range(len(candidates)), key=lambda index: -candidates[index]["score"])
    best = candidates[ranked[0]]
    runner_up = candidates[ranked[1]] if len(ranked) > 1 else None
    margin = best["score"] - runner_up["score"] if runner_up else best["score"]
    relative_margin = margin / max(abs(best["score"]), np.finfo(float).tiny)
    tie_threshold = NUMERICAL_TIE_TOLERANCE * max(1.0, abs(best["score"]))
    tied_orders = [candidate["order"] for candidate in candidates
                   if abs(best["score"] - candidate["score"]) <= tie_threshold]
    signal_detected = output["tensor_energy"] > 0.0
    is_unique = len(tied_orders) == 1 and signal_detected
    weak_edge = min(best["edge_scores"]) <= NEAR_RANK_ONE_EDGE_SCORE
    flags = []
    if not signal_detected:
        flags.append("zero_tensor")
    if len(tied_orders) > 1:
        flags.append("numerically_tied_orders")
    if relative_margin < LOW_RELATIVE_MARGIN:
        flags.append("low_relative_margin")
    if weak_edge:
        flags.append("near_rank_one_selected_edge")
    if min(output["tensor_shape"]) < 2:
        flags.append("scalar_axis_cannot_resolve_all_adjacent_edges")
    output.update({
        "selected_order": best["order"] if is_unique else None,
        "argmax_order": best["order"],
        "numerically_unique": is_unique,
        "best_score": best["score"],
        "runner_up_order": runner_up["order"] if runner_up else None,
        "runner_up_score": runner_up["score"] if runner_up else None,
        "margin": float(margin),
        "absolute_margin": float(margin),
        "relative_margin": float(relative_margin),
        "tied_orders": tied_orders,
        "candidates": candidates,
        "candidate_orders": [candidate["order"] for candidate in candidates],
        "candidate_scores": [candidate["score"] for candidate in candidates],
        "candidate_ranking": ranked,
        "minimum_selected_edge_score": min(best["edge_scores"]),
        "off_path_score_fraction": best["off_path_score"] / max(total_pair_score, np.finfo(float).tiny),
        "diagnostic_flags": flags,
        "diagnostic_thresholds": {
            "numerical_tie_tolerance": NUMERICAL_TIE_TOLERANCE,
            "low_relative_margin": LOW_RELATIVE_MARGIN,
            "near_rank_one_edge_score": NEAR_RANK_ONE_EDGE_SCORE,
        },
        "confidence_calibrated": False,
    })
    return output
