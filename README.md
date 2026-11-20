# Confidence Sets for Reflection Order From a Multi-RIS Cascade Tensor

**Version 1.0.0 DOI:** [10.5281/zenodo.24392339](https://doi.org/10.5281/zenodo.24392339).

**Ziming Liu** · Research data, simulation code, and figure and table sources for the manuscript of the same title.

This repository reproduces a four-surface experiment that asks which reflection orders fit the coefficient tensor of one isolated cascade. The receiver estimates the tensor from two repeats of a 256-state binary pilot grid and tests 14 pairs of tensor modes. Its output is a subset of the 24 possible surface orders. The data come from numerical simulations of a static complex multilinear channel.

## Figures and tables

| Manuscript item | View or download | What one row or point represents |
|---|---|---|
| Table I, mixed pilot grids | [CSV](results/figures_and_tables/table1_mixed_grid.csv) · [LaTeX](results/figures_and_tables/table1_mixed_grid.tex) | 1,200 independent channels for each combination of link regime and total received SNR. There are nine combinations and 10,800 channels. |
| Table II, repeated acquisitions | [CSV](results/figures_and_tables/table2_fixed_channels.csv) · [LaTeX](results/figures_and_tables/table2_fixed_channels.tex) | 2,000 independent noise draws on one fixed mean channel at 30 dB. Six fixed channels give 12,000 acquisitions. |
| Figure 1, paired perturbations | [Vector PDF](results/figures_and_tables/fig1_perturbation_sweep.pdf) · [Plot data](results/figures_and_tables/fig1_perturbation_sweep.csv) | Each point uses the same 600 base channels. Seven perturbation levels give 4,200 evaluations. The two panels report mean confidence-set size and correct singleton percentage. |

In the tables, an exact result equals the whole noiseless compatible family. A miss excludes at least one compatible order. A conflict occurs when the declared edges fit no candidate path, triggering the rule that returns all 24 paths. The CSV tables retain the precision displayed in the paper. The JSON summaries also include unrounded means, quartiles, histograms, and method-specific counts.

## Trial-level data

All four archives use NumPy's compressed NPZ format. Each array's first dimension indexes trials. Each observed complex tensor has shape `(2, 2, 2, 2, 4, 2)`, with modes ordered A, B, C, D, receive antenna, transmit antenna. The receiver inputs are `observed` and the independent coefficient-noise variance estimate `vhat`. Clean-channel quantities and generating labels support evaluation. The seed indices and the simulation code reconstruct the two underlying noisy pilot grids.

| Archive | Contents |
|---|---|
| [evidence.npz](outputs/ris_spl_sets/evidence.npz) | 10,800 observed tensors, variance estimates, 14 pair residuals per trial, candidate masks from both confidence-set methods, forced-order outputs, and generating identifiers. |
| [original_diagnostics.npz](outputs/ris_spl_sets_revision/original_diagnostics.npz) | Clean diagnostics and compatibility masks for the same 10,800 channels. Join `ids` to `seed_indices` in the original evidence. |
| [fixed.npz](outputs/ris_spl_sets_revision/fixed.npz) | 12,000 acquisitions. The three `ids` columns index the channel regime, fixed-channel replicate, and noise draw. |
| [sweep.npz](outputs/ris_spl_sets_revision/sweep.npz) | 4,200 paired observations. The three `ids` columns index the generating order, base replicate, and perturbation level. |

All identifiers start at zero. Bit positions in an order mask follow the lexicographic permutations of ABCD, beginning with bit 0 for ABCD. In the original evidence, columns 0 and 1 of `candidate_masks` correspond to the pairwise and global methods. The ordered pairs for edge masks and residuals appear in [the mixed-grid summary](outputs/ris_spl_sets/summary.json). The [revision summary](outputs/ris_spl_sets_revision/summary.json) contains the six fixed-channel cases and seven perturbation levels.

```python
import numpy as np

with np.load("outputs/ris_spl_sets/evidence.npz", allow_pickle=False) as archive:
    print(archive["observed"].shape)  # (10800, 2, 2, 2, 2, 4, 2)
    print(archive["candidate_masks"][0])
```

## Reproduce the results

Run commands from the repository root. To check all archived scientific files against the release checksums, run `python verify_release.py` before regenerating any output. The simulation used Python 3.12.15, NumPy 2.5.3, SciPy 1.18.1, and Matplotlib 3.11.2. Use a virtual environment and install the pinned numerical stack:

```sh
python -m pip install -r requirements.txt
python results/figures_and_tables/generate_figure_and_tables.py
python results/figures_and_tables/export_csv.py
```

The first script rebuilds the two LaTeX data tables and the vector figure from the JSON summaries. The second exports the numerical CSV files. Rebuilding the plot writes a fresh creation timestamp into its PDF.

For an independent scientific check, run the following in a working copy. These programs reconstruct the noisy grids from the seeds, recompute the decisions and summaries, and write their findings to the respective `audit.json` files.

```sh
python -m work.ris_spl_sets.audit
python -m work.ris_spl_sets_revision.audit
```

To rerun the forward simulations in separate directories, use this sequence in a fresh checkout. Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before starting Python to match the original thread settings. Different BLAS libraries and operating systems may change floating-point roundoff, line endings, or binary archive hashes.

First copy the prespecified protocol hashes into the new destinations using Python:

```python
from pathlib import Path
from shutil import copyfile

for study in ("ris_spl_sets", "ris_spl_sets_revision"):
    destination = Path("replay") / study
    destination.mkdir(parents=True, exist_ok=True)
    copyfile(Path("outputs") / study / "freeze.json", destination / "freeze.json")
```

Then run the two simulation modules:

```sh
python -m work.ris_spl_sets.simulate --destination replay/ris_spl_sets
python -m work.ris_spl_sets_revision.compute --destination replay/ris_spl_sets_revision
```

The second simulation verifies its inputs against the original mixed-grid archive in `outputs/ris_spl_sets`. Both destination directories receive new summaries and trial-level archives. The [mixed-grid protocol](work/ris_spl_sets/protocol.md) and [fixed-channel and perturbation protocol](work/ris_spl_sets_revision/protocol.md) give the draw schedule, physical dimensions, decision rules, and reported metrics. The archive includes the [mixed-grid audit](outputs/ris_spl_sets/audit.json) and [revision audit](outputs/ris_spl_sets_revision/audit.json) produced with the original environment.

## Citation and reuse

See [CITATION.cff](CITATION.cff) for the release citation and [LICENSE.txt](LICENSE.txt) for the terms of reuse.
