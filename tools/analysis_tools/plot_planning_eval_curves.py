#!/usr/bin/env python3
"""Summarize SSR planning evaluation pickles and plot paper-ready curves."""

import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

fm._load_fontmanager(try_read_cache=False)
available_fonts = [f.name for f in fm.fontManager.ttflist]
for font in ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC']:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        break


MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

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


def get_curve_style(idx):
    return {
        "marker": MARKERS[idx % len(MARKERS)],
        "linestyle": "-",
        "alpha": min(0.70 + idx * 0.08, 0.95),
        "linewidth": 1.5,
        "markersize": 5,
        "markerfacecolor": 'none',
        "zorder": 2 + idx,
    }


def auto_ylim(values, small_upper=0.01):
    if not values:
        return None

    ymin = min(values)
    ymax = max(values)
    span = ymax - ymin

    if span == 0:
        pad = max(abs(ymax) * 0.15, 0.0005 if ymax <= small_upper else 0.05)
        return ymin - pad, ymax + pad

    pad_ratio = 0.20 if ymax <= small_upper else 0.12
    pad = span * pad_ratio
    lower = ymin - pad
    upper = ymax + pad
    if ymin >= 0 and lower < 0:
        lower = -min(pad, max(ymax * 0.08, 0.0002))
    return lower, upper


def annotate_3s_values(ax, records, metric_prefix, percent=False):
    grouped = {}
    for record in records:
        value = record[f"{metric_prefix}_3s"]
        value = value * 100 if percent else value
        key = round(value, 10)
        grouped.setdefault(key, []).append(record["label"])

    for idx, (value, labels) in enumerate(sorted(grouped.items())):
        suffix = " 重叠" if len(labels) > 1 else ""
        ax.annotate(
            f"{value:.4g}{suffix}",
            xy=(3, value),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=4,
            color='dimgray',
            ha='center',
            va='bottom',
            clip_on=True,
        )
        if len(labels) > 1:
            ax.text(
                3.02,
                value,
                "数值相同",
                fontsize=4,
                color='dimgray',
                ha='left',
                va='center',
                alpha=0.75,
            )


def save_line_plot(records, metric_prefix, ylabel, title, out_dir, percent=False):
    horizons = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    plotted_values = []
    for idx, record in enumerate(records):
        values = [record[f"{metric_prefix}_{t}s"] for t in horizons]
        if percent:
            values = [v * 100 for v in values]
        plotted_values.extend(values)
        ax.plot(
            horizons,
            values,
            color=f"C{idx}",
            **get_curve_style(idx),
            label=record["label"],
        )
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{t}s" for t in horizons])
    ax.set_xlabel("时间范围", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='both', labelsize=5)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ylim = auto_ylim(plotted_values)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        fontsize=6,
        frameon=False,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{metric_prefix}.{ext}", dpi=300, bbox_inches='tight')
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
    save_line_plot(records, "plan_L2", "L2轨迹误差（m）", "不同时间范围内的L2轨迹误差", out_dir)
    save_line_plot(
        records,
        "plan_obj_col",
        "物体碰撞率（%）",
        "不同时间范围内的物体碰撞率",
        out_dir,
        percent=True,
    )
    save_line_plot(
        records,
        "plan_obj_box_col",
        "边界框碰撞率（%）",
        "不同时间范围内的边界框碰撞率",
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
