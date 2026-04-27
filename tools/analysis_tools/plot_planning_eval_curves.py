#!/usr/bin/env python3
"""Summarize SSR planning evaluation pickles and plot paper-ready curves."""

import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_KEYS = [
    "plan_L2_1s",
    "plan_L2_2s",
    "plan_L2_3s",
    "plan_obj_col_1s",
    "plan_obj_col_2s",
    "plan_obj_col_3s",
    "plan_obj_box_col_1s",
    "plan_obj_box_col_2s",
    "plan_obj_box_col_3s",
    "plan_L2_stp3_1s",
    "plan_L2_stp3_2s",
    "plan_L2_stp3_3s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate result.pkl planning metrics and draw curves.")
    parser.add_argument(
        "results",
        nargs="+",
        help="Result pickle paths. Each should contain per-sample metric_results.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional curve labels, one per result file.",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_eval",
        help="Directory for CSV and figures.",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "bbox_results" in obj:
        return obj["bbox_results"]
    if isinstance(obj, list):
        return obj
    raise TypeError(f"{path} is not a result list or dict with bbox_results")


def aggregate(path):
    rows = load_rows(path)
    sums = {}
    valid = 0
    for row in rows:
        metrics = row.get("metric_results")
        if not metrics or not metrics.get("fut_valid_flag", False):
            continue
        valid += 1
        for key, value in metrics.items():
            if key == "fut_valid_flag":
                continue
            sums[key] = sums.get(key, 0.0) + float(value)

    if valid == 0:
        raise ValueError(f"{path} contains no valid metric_results")

    means = {key: value / valid for key, value in sums.items()}
    means["valid_samples"] = valid
    means["total_samples"] = len(rows)
    means["L2_avg"] = sum(means[k] for k in ["plan_L2_1s", "plan_L2_2s", "plan_L2_3s"]) / 3
    means["CR_avg_percent"] = (
        sum(means[k] for k in ["plan_obj_col_1s", "plan_obj_col_2s", "plan_obj_col_3s"])
        / 3
        * 100
    )
    return means


def write_csv(records, out_dir):
    fieldnames = [
        "label",
        "path",
        "valid_samples",
        "total_samples",
        *METRIC_KEYS,
        "L2_avg",
        "CR_avg_percent",
    ]
    csv_path = out_dir / "planning_eval_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return csv_path


def save_line_plot(records, metric_prefix, ylabel, title, out_dir, percent=False):
    horizons = [1, 2, 3]
    plt.figure(figsize=(5.0, 3.4))
    for record in records:
        values = [record[f"{metric_prefix}_{t}s"] for t in horizons]
        if percent:
            values = [v * 100 for v in values]
        plt.plot(horizons, values, marker="o", linewidth=2.0, label=record["label"])
    plt.xticks(horizons, [f"{t}s" for t in horizons])
    plt.xlabel("Prediction horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend(frameon=False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"{metric_prefix}.{ext}", dpi=300)
    plt.close()


def main():
    args = parse_args()
    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("--labels must have the same length as results")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(path).stem for path in args.results]
    records = []
    for label, path in zip(labels, args.results):
        metrics = aggregate(path)
        metrics["label"] = label
        metrics["path"] = os.path.abspath(path)
        records.append(metrics)

    csv_path = write_csv(records, out_dir)
    save_line_plot(records, "plan_L2", "L2 error (m)", "Planning L2 by horizon", out_dir)
    save_line_plot(
        records,
        "plan_obj_col",
        "Collision rate (%)",
        "Planning collision rate by horizon",
        out_dir,
        percent=True,
    )
    save_line_plot(
        records,
        "plan_obj_box_col",
        "Box collision rate (%)",
        "Planning box collision rate by horizon",
        out_dir,
        percent=True,
    )

    print(f"Wrote {csv_path}")
    for record in records:
        print(
            f"{record['label']}: valid={record['valid_samples']}/{record['total_samples']}, "
            f"L2_avg={record['L2_avg']:.6f}, CR_avg={record['CR_avg_percent']:.6f}%"
        )


if __name__ == "__main__":
    main()
