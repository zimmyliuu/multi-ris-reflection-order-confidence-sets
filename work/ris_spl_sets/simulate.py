"""Run the preregistered measurement-space simulation and emit auditable data."""

import argparse
import hashlib
from itertools import combinations, product, permutations
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import scipy

from work.ris_allorders.core import make_codebook
from work.ris_allorders.model import complex_normal
from . import receiver


SOURCE = Path(__file__).resolve().parent
DEST = SOURCE.parents[1] / "outputs" / "ris_spl_sets"
REGIMES = ("generic", "near_rank1", "rank1")
SNRS = (10, 20, 30)
SUBSETS = tuple(subset for k in range(1, 5) for subset in combinations(range(4), k))
SUBORDERS = {subset: tuple(permutations(subset)) for subset in SUBSETS}
BOOK = make_codebook()
GRID = BOOK["train_design"][:256]
PATTERNS = np.asarray(BOOK["basis_patterns"])
BLOCKS = {}
for subset in SUBSETS:
    active = np.array([board in subset for board in range(4)])
    rows = np.flatnonzero(np.all((PATTERNS > 0) == active, axis=1))
    BLOCKS[subset] = (rows, PATTERNS[rows] - 1)
TARGET_ROWS = BLOCKS[SUBSETS[-1]][0]
TARGET_DESIGN = np.ascontiguousarray(GRID[:, TARGET_ROWS].T)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes():
    files = [SOURCE / name for name in ("protocol.md", "receiver.py", "simulate.py")]
    files += [SOURCE.parent / "ris_allorders" / f for f in ("model.py", "core.py")]
    return {p.relative_to(SOURCE.parents[1]).as_posix(): digest(p) for p in files}


def mixed_trial(regime_index, snr_index, order_index, replicate):
    """Generate raw despread noisily acquired grids, and evaluation-only nuisance labels."""
    seed = [20261119, 773410, regime_index, snr_index, order_index, replicate]
    links, nuisances, noise = [np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(j,)))
                               for j in range(3)]
    coefficients = np.zeros((81, 4, 2), dtype=np.complex128)
    coefficients[0] = complex_normal(links, (4, 2), .3)
    incoming, outgoing, internal = {}, {}, {}
    for board in range(4):
        incoming[board] = complex_normal(links, (2, 2), 1 / np.sqrt(2))
        outgoing[board] = complex_normal(links, (4, 2), 1 / np.sqrt(2))
    for after in range(4):
        for before in range(4):
            if after == before:
                continue
            full = complex_normal(links, (2, 2), .6 / np.sqrt(2))
            if regime_index:
                u = complex_normal(links, (2,))
                v = complex_normal(links, (2,))
                rank_one = (.6 / np.sqrt(2)) * np.outer(u, v)
                full = rank_one + .03 * full if regime_index == 1 else rank_one
            internal[after, before] = full
    nuisance_indices = []
    for subset in SUBSETS:
        if len(subset) == 4:
            order = receiver.ORDERS[order_index]
        else:
            position = int(nuisances.integers(len(SUBORDERS[subset])))
            order = SUBORDERS[subset][position]
            nuisance_indices.append(position)
        rows, cells = BLOCKS[subset]
        first, last = order[0], order[-1]
        spatial = (outgoing[last].T[cells[:, last], :, None]
                   * incoming[first][cells[:, first], None, :])
        for before, after in zip(order[:-1], order[1:]):
            values = internal[after, before][cells[:, after], cells[:, before]]
            spatial *= values[:, None, None]
        coefficients[rows] = spatial
    clean = (GRID @ coefficients.reshape(81, 8)).reshape(256, 4, 2)
    variance = np.mean(np.abs(clean) ** 2) / 10 ** (SNRS[snr_index] / 10)
    y = clean + complex_normal(noise, (2, 256, 4, 2), np.sqrt(variance))
    return y[0], y[1], nuisance_indices


def separate(y1, y2):
    """The target estimate comes from replica sums, its variance estimate from differences."""
    tensor = (TARGET_DESIGN @ (y1 + y2).reshape(256, 8)) / 512
    vhat = np.mean(abs(y1 - y2) ** 2) / 1024
    return tensor.reshape(receiver.SHAPE), vhat


def compatible_mask(regime_index, order_index):
    order = receiver.ORDERS[order_index]
    mask = 1 << order_index
    if regime_index == 2:
        swapped = (order[0], order[2], order[1], order[3])
        mask |= 1 << receiver.ORDERS.index(swapped)
    return mask


def set_summary(mask, compatible, truth, conflicts):
    sizes = np.array([int(m).bit_count() for m in mask])
    n = len(mask)
    single = sizes == 1
    histogram = {str(i): int(np.sum(sizes == i)) for i in sorted(set(sizes.tolist()))}
    return {
        "channels": n, "full_family_covered": int(np.sum((mask & compatible) == compatible)),
        "generating_order_included": int(np.sum((mask & truth) != 0)),
        "exact_set_recovery": int(np.sum(mask == compatible)),
        "singletons": int(np.sum(single)), "abstentions": int(np.sum(~single)),
        "wrong_label_singletons": int(np.sum(single & (mask != truth))),
        "structurally_false_singletons": int(np.sum(single & (mask != compatible))),
        "raw_conflicts": int(np.sum(conflicts)), "mean_size": float(np.mean(sizes)),
        "median_size": float(np.median(sizes)), "size_histogram": histogram,
    }


def summarize(data):
    result = {}
    indices = data["seed_indices"]
    for r, regime in enumerate(REGIMES):
        result[regime] = {}
        for s, snr in enumerate(SNRS):
            use = (indices[:, 0] == r) & (indices[:, 1] == s)
            truth = indices[use, 2].astype(np.uint32)
            masks = data["candidate_masks"][use]
            compatible = data["compatible_mask"][use]
            if len(truth) != 1200 or not np.all(np.bincount(truth, minlength=24) == 50):
                raise ValueError("Incomplete stratum")
            group = {method: set_summary(masks[:, i], compatible, 1 << truth,
                                         data["conflicts"][use, i])
                     for i, method in enumerate(("pair", "ball"))}
            point = data["forced"][use]
            group["forced"] = {
                "correct": int(np.sum(point == truth)),
                "compatible": int(np.sum(((1 << point.astype(np.uint32)) & compatible) != 0)),
                "near_ties": int(np.sum(data["near_tie"][use])),
            }
            result[regime][str(snr)] = group
    return result


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    freeze_file = directory / "freeze.json"
    if not freeze_file.exists():
        raise RuntimeError("Record the protocol and implementation hashes first with --freeze.")
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if freeze != frozen_hashes():
        raise RuntimeError("Simulation source differs from the frozen implementation")
    indices = np.array(list(product(range(3), range(3), range(24), range(50))), dtype=np.uint8)
    data = {
        "seed_indices": indices,
        "observed": np.empty((len(indices),) + receiver.SHAPE, dtype=np.complex128),
        "vhat": np.empty(len(indices)), "nuisance_orders": np.empty((len(indices), 14), dtype=np.uint8),
        "compatible_mask": np.empty(len(indices), dtype=np.uint32),
    }
    batch = 50
    for start in range(0, len(indices), batch):
        region = slice(start, start + batch)
        for index in range(start, start + batch):
            r, s, o, rep = map(int, indices[index])
            y1, y2, data["nuisance_orders"][index] = mixed_trial(r, s, o, rep)
            data["observed"][index], data["vhat"][index] = separate(y1, y2)
            data["compatible_mask"][index] = compatible_mask(r, o)
        decisions = receiver.receive(data["observed"][region], data["vhat"][region])
        for field, values in decisions.items():
            if field not in data:
                data[field] = np.empty((len(indices),) + values.shape[1:], dtype=values.dtype)
            data[field][region] = values
        if (start + batch) % 1200 == 0:
            print(f"completed {start + batch}/{len(indices)}", flush=True)
    groups = summarize(data)
    with TemporaryDirectory(dir=directory) as temp:
        evidence = Path(temp) / "evidence.npz"
        np.savez_compressed(evidence, **data)
        report = {
            "freeze_sha256": digest(freeze_file), "evidence_sha256": digest(evidence),
            "channels": len(indices), "axes": ["A", "B", "C", "D", "Rx", "Tx"],
            "order_labels": ["".join("ABCD"[i] for i in order) for order in receiver.ORDERS],
            "pairs": receiver.PAIRS, "methods": ["pair", "ball"], "snr_db": SNRS,
            "regimes": REGIMES, "groups": groups, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
        }
        summary = Path(temp) / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        evidence.replace(directory / evidence.name)
        summary.replace(directory / summary.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        args.destination.mkdir(parents=True, exist_ok=True)
        path = args.destination / "freeze.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(frozen_hashes(), stream, indent=2)
            stream.write("\n")
    else:
        run(args.destination)
