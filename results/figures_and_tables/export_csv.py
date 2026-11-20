"""Export the displayed table rows and the numerical points of Figure 1 as CSV."""
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[1] / "outputs" / "ris_spl_sets_revision" / "summary.json"


def save_csv(filename, columns, rows):
    with (HERE / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def table_rows(filename):
    previous_label = None
    for line in (HERE / filename).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.rstrip("\\ ").split("&")]
        if cells[0]:
            previous_label = cells[0]
        cells[0] = previous_label
        yield cells


def main():
    save_csv("table1_mixed_grid.csv", [
        "link_regime", "total_snr_db", "median_target_snr_db",
        "median_weakest_standardized_interaction", "pair_missing_family_count",
        "pair_exact_family_count", "pair_mean_set_size", "pair_conflict_count",
        "global_exact_family_count", "global_mean_set_size",
        "forced_correct_label_count", "forced_compatible_order_count",
    ], table_rows("table1_mixed_grid_rows.tex"))
    save_csv("table2_fixed_channels.csv", [
        "link_regime", "fixed_channel_id", "false_edge_count", "any_false_edge_count",
        "missing_family_count", "exact_family_count", "mean_set_size",
    ], table_rows("table2_fixed_channel_rows.tex"))

    sweep = json.loads(SUMMARY.read_text(encoding="utf-8"))["sweep"]
    rows = []
    for cell in sweep:
        correct = cell["singletons"] - cell["false_singletons"]
        rows.append([cell["epsilon"], cell["n"], cell["mean_size"],
                     correct, 100 * correct / cell["n"]])
    save_csv("fig1_perturbation_sweep.csv", [
        "epsilon", "paired_base_channels", "mean_set_size",
        "correct_singleton_count", "correct_singleton_percent",
    ], rows)


if __name__ == "__main__":
    main()
