"""Reproduce the two bounded manuscript revision experiments and original diagnostics."""
import argparse
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from work.ris_allorders.model import complex_normal
from work.ris_spl_sets import receiver as rx
from work.ris_spl_sets import simulate as old


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "ris_spl_sets_revision"
ARCHIVE = ROOT / "outputs" / "ris_spl_sets"
EPSILON = (0., .01, .03, .10, .30, .60, 1.)
REGIMES = old.REGIMES
ALL_PAIRS = (1 << len(rx.PAIRS)) - 1
N_FIXED = 2000


def frozen_files():
    files = [SOURCE / "compute.py", SOURCE / "protocol.md",
             ARCHIVE / "freeze.json", ARCHIVE / "summary.json"]
    prior = json.loads((ARCHIVE / "freeze.json").read_text())
    for name, expected in prior.items():
        path = ROOT / name
        if old.digest(path) != expected:
            raise ValueError(f"Historical source changed: {path}")
        files.append(path)
    return {p.relative_to(ROOT).as_posix(): old.digest(p) for p in files}


def streams(seed):
    return tuple(np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3))


def draw_factors(links, stressed):
    direct = complex_normal(links, (4, 2), .3)
    first, last, gaussian, outer = {}, {}, {}, {}
    for i in range(4):
        first[i] = complex_normal(links, (2, 2), 1 / np.sqrt(2))
        last[i] = complex_normal(links, (4, 2), 1 / np.sqrt(2))
    for after, before in product(range(4), repeat=2):
        if after == before:
            continue
        k = after, before
        gaussian[k] = complex_normal(links, (2, 2), .6 / np.sqrt(2))
        if stressed:
            u = complex_normal(links, (2,))
            v = complex_normal(links, (2,))
            outer[k] = .6 / np.sqrt(2) * np.outer(u, v)
    return direct, first, last, gaussian, outer


def draw_orders(nuisances, order_index):
    result = {}
    for subset in old.SUBSETS:
        if len(subset) == 4:
            result[subset] = rx.ORDERS[order_index]
        else:
            options = old.SUBORDERS[subset]
            result[subset] = options[int(nuisances.integers(len(options)))]
    return result


def mixed_grid(factors, orders, epsilon=None, normalize=False):
    direct, first, last, gaussian, outer = factors
    if epsilon is None:
        links = gaussian
    else:
        scale = np.sqrt(1 + epsilon**2) if normalize else 1.
        links = {k: (outer[k] + epsilon * gaussian[k]) / scale for k in gaussian}
    coefficients = np.empty((81, 4, 2), dtype=complex)
    coefficients[0] = direct
    for subset, order in orders.items():
        rows, cells = old.BLOCKS[subset]
        a, b = order[0], order[-1]
        wave = last[b].T[cells[:, b], :, None] * first[a][cells[:, a], None, :]
        for before, after in zip(order, order[1:]):
            wave *= links[after, before][cells[:, after], cells[:, before], None, None]
        coefficients[rows] = wave
    clean = (old.GRID @ coefficients.reshape(81, 8)).reshape(256, 4, 2)
    return clean, coefficients[old.TARGET_ROWS].reshape(rx.SHAPE)


def original_grid(regime, snr_index, order_index, replicate):
    link_rng, nuisance_rng, noise_rng = streams([20261119, 773410, regime, snr_index,
                                                 order_index, replicate])
    factors = draw_factors(link_rng, regime != 0)
    orders = draw_orders(nuisance_rng, order_index)
    clean, target = mixed_grid(factors, orders, (None, .03, 0.)[regime])
    return clean, target, noise_rng


def visible_mask(order_index, degenerate):
    order = rx.ORDERS[order_index]
    if not degenerate:
        return int(rx.PATH_MASKS[order_index])
    terminal = [(order[0], rx.TX), (order[-1], rx.RX)]
    return sum(1 << rx.PAIRS.index(edge) for edge in terminal)


def clean_diagnostics(clean, target, order_index, degenerate, snr_db):
    sigma2 = np.mean(abs(clean)**2) / 10**(snr_db / 10)
    variance = sigma2 / 512
    visible = visible_mask(order_index, degenerate)
    interactions = []
    for i, pair in enumerate(rx.PAIRS):
        if visible & (1 << i):
            matrices = np.moveaxis(target, pair, (-2, -1))
            tail = np.linalg.svd(matrices, compute_uv=False)[..., 1:]
            interactions.append(np.linalg.norm(tail) / np.sqrt(variance))
    return {
        "true_v": variance,
        "target_snr_db": 10 * np.log10(np.mean(abs(target)**2) / variance),
        "weakest_interaction": min(interactions),
        "visible_edges": visible,
        "truth_mask": old.compatible_mask(2 if degenerate else 0, order_index),
    }


def noise_grid(rng):
    return complex_normal(rng, (2, 256, 4, 2))


def acquire(clean, standard_noise, true_v):
    y = clean + np.sqrt(true_v * 512) * standard_noise
    return old.separate(y[0], y[1])


def quantiles(values):
    return dict(zip(("q25", "median", "q75"), np.quantile(values, [.25, .5, .75]).tolist()))


def as_arrays(records):
    return {name: np.stack([r[name] for r in records]) for name in records[0]}


def original_replay(dest):
    records = []
    expected = json.loads((ARCHIVE / "summary.json").read_text())["evidence_sha256"]
    if old.digest(ARCHIVE / "evidence.npz") != expected:
        raise ValueError("Archived evidence digest changed")
    with np.load(ARCHIVE / "evidence.npz") as archive:
        fields = {k: archive[k] for k in ("seed_indices", "observed", "vhat")}
    max_error = max_vhat_error = 0.
    for i, identifier in enumerate(fields["seed_indices"]):
        r, s, o, rep = map(int, identifier)
        clean, target, rng = original_grid(r, s, o, rep)
        metrics = clean_diagnostics(clean, target, o, r == 2, old.SNRS[s])
        observed, vhat = acquire(clean, noise_grid(rng), metrics["true_v"])
        ref = fields["observed"][i]
        max_error = max(max_error, np.linalg.norm(observed-ref) / np.linalg.norm(ref))
        max_vhat_error = max(max_vhat_error, abs(vhat / fields["vhat"][i] - 1))
        records.append(dict(ids=identifier, **metrics))
    if max(max_error, max_vhat_error) > 1e-12:
        raise ValueError("Failed to reproduce the input of the historical receiver")
    arrays = as_arrays(records)
    np.savez_compressed(dest / "original_diagnostics.npz", **arrays)
    rows = []
    for r, s in product(range(3), range(3)):
        use = (arrays["ids"][:, 0] == r) & (arrays["ids"][:, 1] == s)
        rows.append(dict(regime=REGIMES[r], total_snr_db=old.SNRS[s], n=int(use.sum()),
                         target_snr_db=quantiles(arrays["target_snr_db"][use]),
                         weakest_interaction=quantiles(arrays["weakest_interaction"][use])))
    return rows, dict(max_relative_tensor_error=max_error,
                      max_relative_vhat_error=max_vhat_error)


def decode_in_batches(arrays):
    parts = []
    for i in range(0, len(arrays["vhat"]), 50):
        y = arrays["observed"][i:i+50]
        v = arrays["vhat"][i:i+50]
        results = rx.receive(y, v)
        parts.append({
            "edges": results["certified_edge_masks"][:, 0], "sets": results["candidate_masks"][:, 0],
            "conflict": results["conflicts"][:, 0], "forced": results["forced"],
        })
    arrays.update({key: np.concatenate([p[key] for p in parts]) for key in parts[0]})


def summary(arrays, use):
    masks = arrays["sets"][use]
    family = arrays["truth_mask"][use]
    visible = arrays["visible_edges"][use]
    edges = arrays["edges"][use].astype(int)
    sizes = np.array([int(m).bit_count() for m in masks])
    nulls = ((visible[:, None] & rx.EDGE_BITS) == 0)
    false = nulls & ((edges[:, None] & rx.EDGE_BITS) != 0)
    return {
        "n": len(masks), "family_covered": int(np.sum((masks & family) == family)),
        "exact": int(np.sum(masks == family)), "singletons": int(np.sum(sizes == 1)),
        "false_singletons": int(np.sum((sizes == 1) & (masks != family))),
        "mean_size": float(sizes.mean()), "conflicts": int(arrays["conflict"][use].sum()),
        "size_histogram": dict(zip(map(str, np.unique(sizes)), np.unique(sizes, return_counts=True)[1].tolist())),
        "any_null_rejected": int(np.any(false, axis=1).sum()),
        "false_edge_total": int(false.sum()), "null_tests": int(nulls.sum()),
        "null_exposures_by_pair": nulls.sum(axis=0).tolist(),
        "false_edges_by_pair": false.sum(axis=0).tolist(),
    }


def fixed(dest):
    records = []
    for r, rep in product(range(3), range(2)):
        clean, target, _ = original_grid(r, 2, 0, rep)
        metrics = clean_diagnostics(clean, target, 0, r == 2, 30)
        for k in range(N_FIXED):
            rng = np.random.default_rng(np.random.SeedSequence([20261119,913802,0,r,rep,k]))
            obs, v = acquire(clean, noise_grid(rng), metrics["true_v"])
            records.append(dict(ids=[r, rep, k], observed=obs, vhat=v, **metrics))
    arrays = as_arrays(records)
    decode_in_batches(arrays)
    np.savez_compressed(dest / "fixed.npz", **arrays)
    rows = []
    for r, rep in product(range(3), range(2)):
        use = (arrays["ids"][:, 0] == r) & (arrays["ids"][:, 1] == rep)
        rows.append(dict(regime=REGIMES[r], replicate=rep, total_snr_db=30,
                         target_snr_db=float(arrays["target_snr_db"][use][0]),
                         weakest_interaction=float(arrays["weakest_interaction"][use][0]),
                         **summary(arrays, use)))
    return rows


def sweep(dest):
    records = []
    for o, rep in product(range(24), range(25)):
        links, nuisances, noise = streams([20261119,913802,1,o,rep])
        factors = draw_factors(links, True)
        orders = draw_orders(nuisances, o)
        standard = noise_grid(noise)
        for j, eps in enumerate(EPSILON):
            clean, target = mixed_grid(factors, orders, eps, normalize=True)
            metrics = clean_diagnostics(clean, target, o, eps == 0, 30)
            obs, v = acquire(clean, standard, metrics["true_v"])
            records.append(dict(ids=[o, rep, j], observed=obs, vhat=v, **metrics))
    arrays = as_arrays(records)
    decode_in_batches(arrays)
    np.savez_compressed(dest / "sweep.npz", **arrays)
    rows = []
    for j, eps in enumerate(EPSILON):
        use = arrays["ids"][:, 2] == j
        rows.append(dict(epsilon=eps, total_snr_db=30,
                         target_snr_db=quantiles(arrays["target_snr_db"][use]),
                         weakest_interaction=quantiles(arrays["weakest_interaction"][use]),
                         **summary(arrays, use)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=OUTPUT)
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    freeze = dest / "freeze.json"
    if args.freeze:
        with freeze.open("x", encoding="utf-8") as f:
            json.dump(frozen_files(), f, indent=2)
        return
    if frozen_files() != json.loads(freeze.read_text()):
        raise ValueError("Sources differ from the pre-run freeze")
    with TemporaryDirectory(dir=dest) as staging:
        temp = Path(staging)
        original, replay = original_replay(temp)
        print("Original diagnostics and noisy input replay complete", flush=True)
        fixed_rows = fixed(temp)
        print("Fixed-channel reacquisitions complete", flush=True)
        sweep_rows = sweep(temp)
        result = dict(original=original, original_replay=replay, fixed=fixed_rows, sweep=sweep_rows,
                      pairs=rx.PAIRS, mode_axes=["A","B","C","D","Rx","Tx"],
                      freeze_sha256=old.digest(freeze),
                      archive_sha256={f.name: old.digest(f) for f in temp.glob("*.npz")})
        (temp/"summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for f in temp.iterdir():
            f.replace(dest/f.name)


if __name__ == "__main__":
    main()
