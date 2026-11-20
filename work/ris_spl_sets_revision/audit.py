"""Replay the evidence using matrix propagation, Walsh decoding and Gram minors.

Synthetic ground truth is used only to audit simulator outputs and metrics.
The decision check reads only estimated tensors and estimated noise variances.
"""
import hashlib
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np

from work.ris_spl_sets.audit import (
    ORDERS, PAIRS, SHAPE, SUBSETS, all_minor_residuals, check,
    critical_multiplier, pack, paths_and_degrees,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ris_spl_sets_revision"
PHASES = np.array(list(product((-1., 1.), repeat=8))).reshape(256, 4, 2)
WALSH = np.array([PHASES[:, np.arange(4), a].prod(axis=1)
                  for a in product(range(2), repeat=4)])
INCIDENCE, DEGREES = paths_and_degrees()
PAIR_CUT = critical_multiplier(DEGREES, .05/14)


def noise(rng, shape):
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / np.sqrt(2)


def draw_channel(seed, mixed_rank, order_id):
    rng, orders, noise_rng = map(np.random.default_rng, seed.spawn(3))
    direct = .3*noise(rng, (4, 2))
    incoming, outgoing = [], []
    for _ in range(4):
        incoming.append(noise(rng, (2, 2)) / np.sqrt(2))
        outgoing.append(noise(rng, (4, 2)) / np.sqrt(2))
    diffuse, rank_one = {}, {}
    for after, before in product(range(4), repeat=2):
        if before != after:
            key = before, after
            diffuse[key] = .6 / np.sqrt(2)*noise(rng, (2, 2))
            if mixed_rank:
                rank_one[key] = .6 / np.sqrt(2)*np.outer(noise(rng, 2), noise(rng, 2))
    paths = []
    for subset in SUBSETS:
        candidates = list(permutations(subset))
        paths.append(candidates[orders.integers(len(candidates))])
    paths.append(ORDERS[order_id])
    return (direct, incoming, outgoing, diffuse, rank_one, paths), noise_rng


def propagate(channel, epsilon=None, normalize=False):
    direct, incoming, outgoing, diffuse, rank_one, paths = channel
    links = diffuse
    if epsilon is not None:
        gain = np.sqrt(1 + epsilon**2) if normalize else 1.
        links = {edge: (outer + epsilon*diffuse[edge])/gain
                 for edge, outer in rank_one.items()}
    clean = np.broadcast_to(direct, (256, 4, 2)).copy()
    for path in paths:
        wave = PHASES[:, path[0], :, None] * incoming[path[0]]
        for before, after in zip(path[:-1], path[1:]):
            wave = PHASES[:, after, :, None] * (links[before, after] @ wave)
        clean += outgoing[path[-1]] @ wave
    tensor = np.tensordot(WALSH, clean, axes=1).reshape(SHAPE)/256
    return clean, tensor


def structural_edges(order_id, exact_rank_one):
    edges = INCIDENCE[order_id].copy()
    if exact_rank_one:
        edges &= np.array([max(p) >= 4 for p in PAIRS])
    compatible = np.all(~edges[None, :] | INCIDENCE, axis=1)
    return edges, pack(compatible)


def clean_metrics(clean, tensor, snr, order_id, exact_rank_one):
    v = np.vdot(clean, clean).real / (clean.size * 512 * 10**(snr/10))
    snr_t = np.vdot(tensor, tensor).real / (128*v)
    edges, truth = structural_edges(order_id, exact_rank_one)
    residuals = all_minor_residuals(tensor[None])[0]
    minimal = np.sqrt(np.min(residuals[edges])/v)
    return v, 10 * np.log10(snr_t), minimal, pack(edges), truth


def decisions(observed, vhat):
    residuals = all_minor_residuals(observed)
    edges = residuals > vhat[:, None]*PAIR_CUT
    permitted = np.all(~edges[:, None, :] | INCIDENCE[None], axis=2)
    conflict = np.sum(permitted, axis=1) == 0
    permitted[conflict] = True
    forced = np.argmax(residuals @ INCIDENCE.T, axis=1)
    return pack(edges), pack(permitted), conflict, forced


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def historical_metrics(ids):
    metrics = np.empty((len(ids), 3))
    fixed_grids = {}
    for row, case in enumerate(ids):
        r, s, o, n = map(int, case)
        seed = np.random.SeedSequence([20261119, 773410, r, s, o, n])
        channel, _ = draw_channel(seed, r > 0, o)
        clean, target = propagate(channel, epsilon=(None, .03, 0)[r])
        metrics[row] = clean_metrics(clean, target, 10*(s+1), o, r == 2)[:3]
        if s == 2 and o == 0 and n < 2:
            fixed_grids[r, n] = (clean, target, metrics[row, 0])
    return metrics, fixed_grids


def measure(clean, coefficient_variance, standardized_noise):
    pairs = clean[None] + np.sqrt(512*coefficient_variance)*standardized_noise
    estimate = (WALSH @ pairs.mean(axis=0).reshape(256, 8)).reshape(SHAPE)/256
    contrasts = (pairs[0] - pairs[1]).ravel()
    vhat = np.vdot(contrasts, contrasts).real / (contrasts.size * 1024)
    return estimate, float(vhat)


def verify_measurement(estimate, vhat, stored, stored_v):
    relative = np.linalg.norm(estimate - stored) / np.linalg.norm(stored)
    scale_error = abs(vhat - stored_v) / stored_v
    check(relative < 1e-10 and scale_error < 1e-10, "Direct grid replay")
    return [float(relative), float(scale_error)]


def load_evidence(name):
    with np.load(OUT / f"{name}.npz", allow_pickle=False) as arc:
        return {k: arc[k] for k in arc.files}


def verify_scores(data):
    for i in range(0, len(data["ids"]), 80):
        part = slice(i, i+80)
        edges, masks, conflict, chosen = decisions(data["observed"][part], data["vhat"][part])
        for key, calculated in (("edges", edges), ("sets", masks),
                                ("conflict", conflict), ("forced", chosen)):
            check(data[key][part] == calculated, key)


def check_group(data, idx, saved):
    truth = data["truth_mask"][idx].astype(np.int64)
    masks = data["sets"][idx].astype(np.int64)
    sizes = np.array([int(m).bit_count() for m in masks])
    bins = np.bincount(sizes, minlength=25)
    singles = sizes == 1
    null_bits = ((1 << 14)-1) ^ data["visible_edges"][idx].astype(np.int64)
    false_bits = null_bits & data["edges"][idx]
    powers = 1 << np.arange(14)
    counts = np.count_nonzero(false_bits[:, None] & powers, axis=0).tolist()
    exposures = np.count_nonzero(null_bits[:, None] & powers, axis=0).tolist()
    fields = {
        "n": int(idx.sum()), "exact": int(np.sum(masks == truth)),
        "family_covered": int(np.sum((masks & truth) == truth)),
        "singletons": int(singles.sum()),
        "false_singletons": int(np.sum(singles & (masks != truth))),
        "conflicts": int(data["conflict"][idx].sum()), "mean_size": float(sizes.mean()),
        "any_null_rejected": int(np.count_nonzero(false_bits)),
        "false_edge_total": sum(counts), "null_tests": sum(exposures),
        "false_edges_by_pair": counts, "null_exposures_by_pair": exposures,
        "size_histogram": {str(i): int(v) for i, v in enumerate(bins) if v},
    }
    for key, value in fields.items():
        check(saved[key] == value, key)


def check_diagnostics(data, vals, order_ids, degenerate):
    for j, field in enumerate(("true_v", "target_snr_db", "weakest_interaction")):
        check(np.allclose(data[field], vals[:, j], atol=2e-11, rtol=1e-8),
              f"Independent diagnostic {field}")
    for i, (o, deg) in enumerate(zip(order_ids, degenerate)):
        edge, mask = structural_edges(int(o), bool(deg))
        check(pack(edge) == data["visible_edges"][i] and mask == data["truth_mask"][i],
              "Compatible paths or visible edges")


def check_quantiles(data, used, row):
    for var in ("target_snr_db", "weakest_interaction"):
        x = np.sort(data[var][used])
        for p, label in ((.25, "q25"), (.5, "median"), (.75, "q75")):
            rank = p*(len(x)-1)
            lo = int(rank)
            v = x[lo] + (rank - lo) * (x[min(lo+1, len(x)-1)] - x[lo])
            check(np.isclose(v, row[var][label]), f"Summary {var} {label}")


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    check(digest(OUT / "freeze.json") == summary["freeze_sha256"], "Freeze digest")
    for name, ref in json.loads((OUT / "freeze.json").read_text()).items():
        check(digest(ROOT / name) == ref, f"Frozen {name}")
    for name, ref in summary["archive_sha256"].items():
        check(digest(OUT / name) == ref, f"Evidence {name}")
    original = load_evidence("original_diagnostics")
    ids = original["ids"].astype(int)
    check(ids == np.asarray(list(product(range(3), range(3), range(24), range(50)))),
          "Historical design")
    metrics, cases = historical_metrics(ids)
    check_diagnostics(original, metrics, ids[:, 2], ids[:, 0] == 2)
    for i, (r, s) in enumerate(product(range(3), range(3))):
        check_quantiles(original, (ids[:, 0] == r) & (ids[:, 1] == s), summary["original"][i])
    print("Independent original-grid replay complete", flush=True)

    fixed = load_evidence("fixed")
    ids = fixed["ids"].astype(int)
    check(ids == np.asarray(list(product(range(3), range(2), range(2000)))), "Fixed design")
    worst = np.zeros(2)
    for i, (r, rep, k) in enumerate(ids):
        clean, _, variance = cases[r, rep]
        rng = np.random.default_rng(np.random.SeedSequence([20261119,913802,0,r,rep,k]))
        y, vhat = measure(clean, variance, noise(rng, (2,256,4,2)))
        worst = np.maximum(worst, verify_measurement(y, vhat, fixed["observed"][i], fixed["vhat"][i]))
    verify_scores(fixed)
    for i, (r, rep) in enumerate(product(range(3), range(2))):
        use = (ids[:, 0] == r) & (ids[:, 1] == rep)
        clean, target, variance = cases[r, rep]
        m = clean_metrics(clean, target, 30, 0, r == 2)
        for key, value in zip(("true_v", "target_snr_db", "weakest_interaction",
                               "visible_edges", "truth_mask"), m):
            check(np.allclose(fixed[key][use], value), f"Fixed {key}")
        saved = summary["fixed"][i]
        check(np.isclose(saved["target_snr_db"], m[1])
              and np.isclose(saved["weakest_interaction"], m[2]), "Fixed summary")
        check_group(fixed, use, saved)
    print("Independent fixed-channel replay complete", flush=True)

    sweep = load_evidence("sweep")
    ids = sweep["ids"].astype(int)
    eps = [0., .01, .03, .1, .3, .6, 1.]
    check(ids == np.asarray(list(product(range(24), range(25), range(7)))), "Sweep design")
    values = np.empty((len(ids), 3))
    for i, (o, rep, j) in enumerate(ids):
        if j == 0:
            seed = np.random.SeedSequence([20261119,913802,1,o,rep])
            channel, rng = draw_channel(seed, True, o)
            z = noise(rng, (2,256,4,2))
        clean, target = propagate(channel, eps[j], normalize=True)
        values[i] = clean_metrics(clean, target, 30, o, j == 0)[:3]
        y, vhat = measure(clean, values[i, 0], z)
        worst = np.maximum(worst, verify_measurement(y, vhat, sweep["observed"][i], sweep["vhat"][i]))
    check_diagnostics(sweep, values, ids[:, 0], ids[:, 2] == 0)
    verify_scores(sweep)
    for j, e in enumerate(eps):
        use = ids[:, 2] == j
        row = summary["sweep"][j]
        check(row["epsilon"] == e, "Sweep value")
        check_group(sweep, use, row)
        check_quantiles(sweep, use, row)
    result = {
        "status": "passed",
        "method": "Direct complex matrix propagation over all 256 control states. "
                  "Walsh projection and new raw-grid replays. Pair residuals from Gram minors "
                  "and independent beta-quantile cutoffs. Independent summary reconstruction.",
        "historical_channels_reconstructed": len(original["ids"]),
        "fixed_channels": 6, "fixed_reacquisitions": len(fixed["ids"]),
        "paired_sweep_base_channels": 600, "paired_sweep_acquisitions": len(sweep["ids"]),
        "new_raw_grids_replayed": 2*(len(fixed["ids"]) + len(sweep["ids"])),
        "new_noisy_pair_statistics_recomputed": 14*(len(fixed["ids"]) + len(sweep["ids"])),
        "max_relative_tensor_error": float(worst[0]),
        "max_relative_vhat_error": float(worst[1]),
        "inputs": summary["archive_sha256"], "audit_source_sha256": digest(Path(__file__)),
    }
    target = OUT / "audit.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
