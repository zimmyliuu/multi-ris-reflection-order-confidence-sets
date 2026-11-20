"""Independent numerical audit of the noisy order-set experiment.

This verifier does not import the experiment's decoder or its summary code.
It recovers each slice residual via two-by-two minors instead of an SVD and
uses beta quantiles to reconstruct both Studentized decision boundaries.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
from scipy.special import betaincinv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "ris_spl_sets"
ORDERS = list(permutations(range(4)))
PAIRS = [pair for pair in combinations(range(6), 2) if pair != (4, 5)]
SHAPE = (2, 2, 2, 2, 4, 2)
SUBSETS = [part for size in (1, 2, 3) for part in combinations(range(4), size)]
REGIMES = ("generic", "near_rank1", "rank1")
FULL_MASK = (1 << len(ORDERS)) - 1


def check(condition, description):
    if not np.all(condition):
        raise AssertionError(description)


def paths_and_degrees():
    paths = np.zeros((len(ORDERS), len(PAIRS)), dtype=bool)
    for i, order in enumerate(ORDERS):
        vertices = (5, *order, 4)
        for u, v in zip(vertices[:-1], vertices[1:]):
            paths[i, PAIRS.index(tuple(sorted((u, v))))] = True
    degrees = []
    for i, j in PAIRS:
        m, n = SHAPE[i], SHAPE[j]
        num_slices = np.prod(SHAPE) // (m * n)
        degrees.append(num_slices * (m - 1) * (n - 1))
    return paths, np.array(degrees)


def critical_multiplier(shape, tail_probability):
    # If U~Gamma(shape) and V~Gamma(2048) are independent, U/(U+V)
    # is beta distributed. This is the quantile of U/(V/2048).
    unit = betaincinv(shape, 2048, 1 - tail_probability)
    return 2048 * unit / (1 - unit)


def all_minor_residuals(observations):
    residuals = np.empty((len(observations), len(PAIRS)))
    for index, (left, right) in enumerate(PAIRS):
        left_dim, right_dim = SHAPE[left], SHAPE[right]
        a = np.moveaxis(observations, (1 + left, 1 + right), (-2, -1))
        a = a.reshape(len(observations), -1, left_dim, right_dim)
        if left_dim != 2:
            a = a.swapaxes(-2, -1)
        # Cauchy-Binet gives the Gram determinant as a sum of squared
        # minors, avoiding subtraction of nearly equal diagonal products.
        gram_det = np.zeros(a.shape[:2], dtype=float)
        for p, q in combinations(range(a.shape[-1]), 2):
            minor = a[..., 0, p] * a[..., 1, q] - a[..., 0, q] * a[..., 1, p]
            gram_det += np.abs(minor) ** 2
        trace = np.sum(np.abs(a) ** 2, axis=(-2, -1))
        disc = np.maximum(trace**2 - 4 * gram_det, 0)
        small_root = np.divide(2 * gram_det, trace + np.sqrt(disc),
                               out=np.zeros_like(gram_det), where=trace > 0)
        residuals[:, index] = small_root.sum(axis=1)
    return residuals


def pack(membership):
    bits = np.left_shift(np.int64(1), np.arange(membership.shape[-1]))
    return np.sum(membership.astype(np.int64) * bits, axis=-1)


def check_full_shape(ids):
    expected = set(product(range(3), range(3), range(24), range(50)))
    actual = {tuple(map(int, row)) for row in ids}
    check(actual == expected and len(actual) == len(ids), "Trial design or seeds differ")


def truth_from_structure(ids):
    masks = []
    for regime, _, order_id, _ in ids:
        order = ORDERS[order_id]
        matches = [order]
        if regime == 2:
            matches.append((order[0], order[2], order[1], order[3]))
        masks.append(sum(1 << ORDERS.index(match) for match in matches))
    return np.array(masks)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_decisions(data):
    observations = data["observed"]
    vhat = data["vhat"]
    ids = data["seed_indices"]
    check_full_shape(ids)
    check(observations.shape == (10800, *SHAPE), "Observed tensor axes")
    check(vhat.shape == (10800,) and np.all(np.isfinite(vhat) & (vhat > 0)),
          "Noise scales")
    paths, deg = paths_and_degrees()
    local = critical_multiplier(deg, .05 / 14)
    global_ball = critical_multiplier(128, .05)
    truth = truth_from_structure(ids)
    check(truth == data["compatible_mask"], "Structural compatible sets")
    recorded_masks, recorded_edges = data["candidate_masks"], data["certified_edge_masks"]
    raw = data["raw_residuals"]
    recorded_conflicts = data["conflicts"]
    forced, near_tie = data["forced"], data["near_tie"]
    max_relative_residual_error = 0.
    for start in range(0, len(ids), 96):
        block = slice(start, start + 96)
        y = observations[block]
        residual = all_minor_residuals(y)
        discrepancy = np.abs(residual - raw[block]) / np.maximum(raw[block], 1e-300)
        max_relative_residual_error = max(max_relative_residual_error, discrepancy.max())
        check(np.allclose(residual, raw[block], atol=0, rtol=1e-10),
              f"Independent Cauchy-Binet residuals at trial {start}")
        scores = residual @ paths.T
        check(np.argmax(scores, axis=1) == forced[block], "Forced winners")
        energy = (np.abs(y.reshape(len(y), -1))**2).sum(axis=1)
        sorted_scores = np.sort(scores / energy[:, None], axis=1)
        ties = (sorted_scores[:, -1] - sorted_scores[:, -2] <=
                1e-12 * np.maximum(1, sorted_scores[:, -1]))
        check(ties == near_tie[block], "Near-tie flags")
        for col, cutoff in enumerate((local, global_ball)):
            detected = residual > vhat[block, None] * cutoff
            check(pack(detected) == recorded_edges[block, col], "Certified pair masks")
            allowed = np.all(~detected[:, None, :] | paths[None, :, :], axis=2)
            failed = ~np.any(allowed, axis=1)
            check(failed == recorded_conflicts[block, col], "Conflict status")
            allowed[failed] = True
            check(pack(allowed) == recorded_masks[block, col], "Candidate membership")
    return float(max_relative_residual_error)


def summarize_ids(data):
    ids = data["seed_indices"]
    masks = data["candidate_masks"]
    cstar = truth_from_structure(ids)
    forced = np.asarray(data["forced"], dtype=np.int64)
    conflicts = data["conflicts"]
    labels = np.asarray(ids[:, 2], dtype=np.int64)
    rows = []
    for regime, snr in product(range(3), repeat=2):
        which = (ids[:, 0] == regime) & (ids[:, 1] == snr)
        truth = cstar[which]
        true_bits = np.left_shift(1, labels[which])
        row = {"regime": REGIMES[regime], "snr_db": 10 * (snr + 1),
               "n": int(which.sum()),
               "forced_true_label": int(np.sum(forced[which] == labels[which])),
               "forced_compatible": int(np.sum((truth & (1 << forced[which])) != 0)),
               "forced_near_tie": int(np.count_nonzero(data["near_tie"][which]))}
        for i, method in enumerate(("pair", "ball")):
            actual = masks[which, i]
            size = np.array([int(mask).bit_count() for mask in actual])
            counts = np.bincount(size, minlength=25)
            singleton = size == 1
            row[method] = {
                "all_compatible_covered": int(np.sum((actual & truth) == truth)),
                "true_label_covered": int(np.sum((actual & true_bits) != 0)),
                "exact": int(np.sum(actual == truth)), "singleton": int(singleton.sum()),
                "wrong_label_singleton": int(np.sum(singleton & (actual != true_bits))),
                "false_unique": int(np.sum(singleton & (actual != truth))),
                "conflicts": int(np.sum(conflicts[which, i])),
                "size_histogram": {str(n): int(x) for n, x in enumerate(counts) if x},
                "mean_size": float(size.mean()), "median_size": float(np.median(size)),
            }
        rows.append(row)
    return rows


def reproduce_grid(trial, controls, chars):
    regime, snr, order_index, replicate = map(int, trial)
    streams = np.random.SeedSequence([20261119, 773410, regime, snr, order_index, replicate])
    factors, labels, noise = map(np.random.default_rng, streams.spawn(3))

    def normal(rng, shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    base = .3 * normal(factors, (4, 2))
    first, last = [], []
    for _ in range(4):
        first.append(normal(factors, (2, 2)) / np.sqrt(2))
        last.append(normal(factors, (4, 2)) / np.sqrt(2))
    link = {}
    for after, before in product(range(4), repeat=2):
        if after == before:
            continue
        matrix = normal(factors, (2, 2)) * .6 / np.sqrt(2)
        if regime:
            u, v = normal(factors, 2), normal(factors, 2)
            outer = u[:, None] * v[None, :] * .6 / np.sqrt(2)
            matrix = outer + .03 * matrix if regime == 1 else outer
        link[before, after] = matrix

    # Simulate each path by ordinary cascaded matrix propagation through
    # explicit element phases, independently of any expansion into monomials.
    clean = np.broadcast_to(base, (256, 4, 2)).copy()
    chosen = []
    for subset in SUBSETS + [tuple(range(4))]:
        options = list(permutations(subset))
        if len(subset) < 4:
            index = int(labels.integers(len(options)))
            chosen.append(index)
            order = options[index]
        else:
            order = ORDERS[order_index]
        wave = controls[:, order[0], :, None] * first[order[0]]
        for past, present in zip(order[:-1], order[1:]):
            wave = controls[:, present, :, None] * (link[past, present] @ wave)
        clean += last[order[-1]] @ wave
    variance = np.vdot(clean, clean).real / clean.size * 10**(-1 - snr)
    acquisitions = clean[None] + np.sqrt(variance) * normal(noise, (2, 256, 4, 2))
    estimate = np.tensordot(chars, acquisitions.mean(axis=0), axes=1) / 256
    clean_target = (np.tensordot(chars, clean, axes=1) / 256).reshape(SHAPE)
    contrasts = np.diff(acquisitions, axis=0).ravel()
    noise_scale = np.vdot(contrasts, contrasts).real / (contrasts.size * 1024)
    return estimate.reshape(SHAPE), float(noise_scale), clean_target, chosen


def audit_grids(data):
    patterns = np.array(list(product((-1., 1.), repeat=8))).reshape(256, 4, 2)
    chars = np.array([np.prod(patterns[:, np.arange(4), el], axis=1)
                      for el in product(range(2), repeat=4)])
    check(chars @ chars.T == 256 * np.eye(16), "Walsh target orthogonality")
    ids = data["seed_indices"]
    paths, _ = paths_and_degrees()
    observed, vhat = data["observed"], data["vhat"]
    nuisance = data["nuisance_orders"]
    maxima = [0., 0.]
    checked = 0
    cstar = truth_from_structure(ids)
    for i, trial in enumerate(ids.astype(int)):
        fresh, scale, clean, chosen = reproduce_grid(trial, patterns, chars)
        error = np.linalg.norm(fresh - observed[i]) / np.linalg.norm(observed[i])
        var_error = abs(scale - vhat[i]) / vhat[i]
        maxima = [max(maxima[0], error), max(maxima[1], var_error)]
        check(max(error, var_error) < 1e-10, "Independent noisy-pilot reconstruction")
        check(chosen == nuisance[i], "Shorter orders from nuisance stream")
        clean_res = all_minor_residuals(clean[None])[0]
        h = clean_res > (np.linalg.norm(clean)**2 * 1e-18)
        compatible = pack(np.all(~h[None, :] | paths, axis=1)[None])[0]
        check(compatible == cstar[i], "Clean slice graph has the structural compatible set")
        checked += 1
    check(checked == 10800, "Complete grid replay count")
    return {"channels": checked, "raw_grids": 2 * checked,
            "max_relative_tensor_error": float(maxima[0]),
            "max_relative_variance_error": float(maxima[1])}


def compare_summary(audit, original):
    keys = {
        "all_compatible_covered": "full_family_covered",
        "true_label_covered": "generating_order_included", "exact": "exact_set_recovery",
        "singleton": "singletons", "wrong_label_singleton": "wrong_label_singletons",
        "false_unique": "structurally_false_singletons", "conflicts": "raw_conflicts",
        "size_histogram": "size_histogram", "mean_size": "mean_size",
        "median_size": "median_size",
    }
    for row in audit:
        cell = original["groups"][row["regime"]][str(row["snr_db"])]
        for method in ("pair", "ball"):
            for a, b in keys.items():
                check(row[method][a] == cell[method][b], f"{method}: {a}")
            check(row["n"] - row[method]["singleton"] == cell[method]["abstentions"],
                  "Abstentions")
        check(row["forced_true_label"] == cell["forced"]["correct"], "Point accuracy")
        check(row["forced_compatible"] == cell["forced"]["compatible"], "Point compatibility")
        check(row["forced_near_tie"] == cell["forced"]["near_ties"], "Point near-ties")


def main():
    with np.load(OUTPUT / "evidence.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive}
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    fingerprint = sha256(OUTPUT / "evidence.npz")
    check(fingerprint == summary["evidence_sha256"], "Evidence hash")
    freeze_path = OUTPUT / "freeze.json"
    check(sha256(freeze_path) == summary["freeze_sha256"], "Freeze hash")
    for name, expected in json.loads(freeze_path.read_text(encoding="utf-8")).items():
        check(sha256(ROOT / name) == expected, f"Frozen source {name}")
    error = audit_decisions(data)
    rows = summarize_ids(data)
    compare_summary(rows, summary)
    grid_replay = audit_grids(data)
    result = {
        "status": "passed", "evidence_sha256": fingerprint,
        "audit_source_sha256": sha256(Path(__file__)),
        "channels": len(data["seed_indices"]),
        "pair_residuals_checked": int(data["raw_residuals"].size),
        "maximum_relative_residual_discrepancy": error,
        "independent_raw_observation_replay": grid_replay,
        "method": "Direct propagation and independent two-grid Walsh decoding for every channel. "
                  "Cauchy-Binet Gram determinants, beta quantiles, enumeration and independent "
                  "aggregate counts for all observed tensors and all nine analysis cells.",
        "cells": rows,
    }
    target = OUTPUT / "audit.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
