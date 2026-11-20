"""Rebuild both tables and the perturbation figure from the two study summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs"
NAMES = {"generic": "Generic", "near_rank1": "Near rank one", "rank1": "Rank one"}


def tex_rows(name, rows):
    (HERE / name).write_text("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
                              + "\n", encoding="utf-8")


def make_tables(old, new):
    rows = []
    for diag in new["original"]:
        regime, snr = diag["regime"], diag["total_snr_db"]
        cell = old["groups"][regime][str(snr)]
        pair, ball, forced = cell["pair"], cell["ball"], cell["forced"]
        rows.append([NAMES[regime] if snr == 10 else "", snr,
                     f'{diag["target_snr_db"]["median"]:.1f}',
                     f'{diag["weakest_interaction"]["median"]:.2f}',
                     pair["channels"] - pair["full_family_covered"],
                     pair["exact_set_recovery"], f'{pair["mean_size"]:.2f}',
                     pair["raw_conflicts"], ball["exact_set_recovery"],
                     f'{ball["mean_size"]:.2f}', forced["correct"], forced["compatible"]])
    tex_rows("table1_mixed_grid_rows.tex", rows)
    fixed = []
    for case in new["fixed"]:
        fixed.append([NAMES[case["regime"]] if case["replicate"] == 0 else "",
                      case["replicate"]+1, case["false_edge_total"], case["any_null_rejected"],
                      case["n"]-case["family_covered"], case["exact"], f'{case["mean_size"]:.2f}'])
    tex_rows("table2_fixed_channel_rows.tex", fixed)


def make_figure(sweep):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.12, 1.67))
    xs = list(range(len(sweep)))
    labels = [f'{s["epsilon"]:g}'.replace("0.", ".") for s in sweep]
    ys = ([g["mean_size"] for g in sweep],
          [100*(g["singletons"]-g["false_singletons"])/g["n"] for g in sweep])
    for ax, values, title, top, ticks in zip(
            (left, right), ys, ("Mean set size", "Correct singleton (%)"), (6, 100),
            ((0, 2, 4, 6), (0, 50, 100))):
        ax.plot(xs, values, color="#255874", linewidth=1.1, marker="o", markersize=3)
        ax.set_title(title, fontsize=10, pad=4, x=.38 if ax is right else .5)
        ax.set_xticks(xs, labels, rotation=65, ha="right", rotation_mode="anchor")
        ax.set_xlim(-.30, len(xs)-.70)
        ax.set_ylim(-.02*top, top*1.02)
        ax.set_yticks(ticks)
        ax.tick_params(length=2, pad=1, labelsize=10)
        ax.yaxis.grid(color="#dedede", linewidth=.55)
        ax.set_axisbelow(True)
    fig.supxlabel("Perturbation ε", fontsize=10, y=.015)
    fig.subplots_adjust(left=.068, right=.963, bottom=.33, top=.83, wspace=.54)
    fig.savefig(HERE / "fig1_perturbation_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    old = json.loads((OUTPUTS/"ris_spl_sets"/"summary.json").read_text())
    new = json.loads((OUTPUTS/"ris_spl_sets_revision"/"summary.json").read_text())
    make_tables(old, new)
    make_figure(new["sweep"])
