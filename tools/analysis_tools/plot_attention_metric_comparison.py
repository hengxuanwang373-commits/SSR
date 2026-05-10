#!/usr/bin/env python3
"""Plot line-chart comparisons for SSR TokenLearner attention metrics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    (
        "CBAM SSR-init",
        "/data3_server8/wanghongxuan/SSR/attention_maps/FrozenTrain/"
        "headonly_cbam_ssrinit_12ep_10p_balanced_epoch_12",
    ),
    (
        "SE",
        "/data3_server8/wanghongxuan/SSR/attention_maps/FrozenTrain/"
        "headonly_se_12ep_10p_balanced_20260506_epoch_12",
    ),
    (
        "SE+CBAM",
        "/data3_server8/wanghongxuan/SSR/attention_maps/FrozenTrain/"
        "se_cbam_after_se_12ep_20260501_095536_epoch_12",
    ),
    (
        "Baseline",
        "/data3_server8/wanghongxuan/SSR/attention_maps/Baseline",
    ),
]

MARKERS = ["o", "s", "^", "D", "P", "X"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SSR TokenLearner explainability metrics from "
            "attention_metrics_frame_*.json files."
        )
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Optional run specs in label=directory format. If omitted, uses "
            "the four paths requested for CBAM SSR-init, SE, SE+CBAM, Baseline."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/data3_server8/wanghongxuan/SSR/attention_maps/metric_comparison",
        help="Directory for output PNG/PDF/CSV files.",
    )
    return parser.parse_args()


def parse_runs(run_specs):
    if not run_specs:
        return [(label, Path(path)) for label, path in DEFAULT_RUNS]

    runs = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid run spec {spec!r}. Expected label=/path/to/run.")
        label, path = spec.split("=", 1)
        if not label.strip() or not path.strip():
            raise ValueError(
                f"Invalid run spec {spec!r}. Label and path are required.")
        runs.append((label.strip(), Path(path.strip())))
    return runs


def find_metrics_file(run_dir):
    files = sorted(run_dir.glob("attention_metrics_frame_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No attention_metrics_frame_*.json found in {run_dir}")
    if len(files) > 1:
        print(f"Warning: found {len(files)} metric files in {run_dir}; "
              f"using {files[0].name}")
    return files[0]


def load_metric_record(label, run_dir):
    metrics_path = find_metrics_file(run_dir)
    with metrics_path.open("r") as f:
        metrics = json.load(f)

    record = {
        "label": label,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "entropy_mean": metrics["entropy"]["normalized_mean"],
        "topk_mass_top_1_percent": metrics["topk_mass"]["top_1_percent"]["mean"],
        "topk_mass_top_5_percent": metrics["topk_mass"]["top_5_percent"]["mean"],
        "topk_mass_top_10_percent": metrics["topk_mass"]["top_10_percent"]["mean"],
        "token_diversity_mean": metrics["token_diversity"][
            "pairwise_cosine_distance_mean"
        ],
        "jaccard_top_1_percent": metrics["token_overlap"][
            "top_1_percent_jaccard"
        ]["mean"],
        "jaccard_top_5_percent": metrics["token_overlap"][
            "top_5_percent_jaccard"
        ]["mean"],
        "jaccard_top_10_percent": metrics["token_overlap"][
            "top_10_percent_jaccard"
        ]["mean"],
    }
    return record


def write_csv(records, out_dir):
    csv_path = out_dir / "attention_metric_comparison.csv"
    fieldnames = [
        "label",
        "run_dir",
        "metrics_path",
        "entropy_mean",
        "topk_mass_top_1_percent",
        "topk_mass_top_5_percent",
        "topk_mass_top_10_percent",
        "token_diversity_mean",
        "jaccard_top_1_percent",
        "jaccard_top_5_percent",
        "jaccard_top_10_percent",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def plot_single_series(ax, labels, values, title, ylabel):
    x = list(range(len(labels)))
    ax.plot(
        x,
        values,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#2f6f9f",
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    for idx, value in enumerate(values):
        ax.annotate(
            f"{value:.3f}",
            (idx, value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )


def plot_topk_series(ax, labels, records, prefix, title, ylabel):
    x = list(range(len(labels)))
    series = [
        ("Top 1%", f"{prefix}_top_1_percent"),
        ("Top 5%", f"{prefix}_top_5_percent"),
        ("Top 10%", f"{prefix}_top_10_percent"),
    ]
    for idx, (series_label, key) in enumerate(series):
        values = [record[key] for record in records]
        ax.plot(
            x,
            values,
            marker=MARKERS[idx % len(MARKERS)],
            linewidth=2.0,
            markersize=6,
            label=series_label,
        )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)


def plot_comparison(records, out_dir):
    labels = [record["label"] for record in records]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    plot_single_series(
        axes[0, 0],
        labels,
        [record["entropy_mean"] for record in records],
        "Attention Entropy",
        "Normalized entropy mean",
    )
    plot_topk_series(
        axes[0, 1],
        labels,
        records,
        "topk_mass",
        "Top-k Attention Mass",
        "Attention mass mean",
    )
    plot_single_series(
        axes[1, 0],
        labels,
        [record["token_diversity_mean"] for record in records],
        "Token Diversity",
        "Pairwise cosine distance mean",
    )
    plot_topk_series(
        axes[1, 1],
        labels,
        records,
        "jaccard",
        "Top-k Mask Jaccard Overlap",
        "Jaccard mean",
    )

    fig.suptitle("SSR TokenLearner Attention Metric Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = out_dir / "attention_metric_comparison.png"
    pdf_path = out_dir / "attention_metric_comparison.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    runs = parse_runs(args.runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [load_metric_record(label, run_dir) for label, run_dir in runs]
    csv_path = write_csv(records, out_dir)
    png_path, pdf_path = plot_comparison(records, out_dir)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PNG: {png_path}")
    print(f"Wrote PDF: {pdf_path}")


if __name__ == "__main__":
    main()
