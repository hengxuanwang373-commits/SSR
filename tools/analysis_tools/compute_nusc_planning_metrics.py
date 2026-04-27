#!/usr/bin/env python3
"""Compute planning L2 metrics from formatted results_nusc.pkl files.

The formatted nuScenes pickle stores only plan_results and plan_gts. This is
enough to recompute ego trajectory L2/FDE metrics, but not collision metrics
because object/map occupancy inputs are not saved in the file.
"""

import argparse
import csv
import pickle
import sys
import types
from pathlib import Path

import torch


METRIC_KEYS = [
    "plan_L2_1s",
    "plan_L2_2s",
    "plan_L2_3s",
    "plan_L2_stp3_1s",
    "plan_L2_stp3_2s",
    "plan_L2_stp3_3s",
]


class ConfigDict(dict):
    """Small stub for unpickling mmcv ConfigDict without importing mmcv."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def install_mmcv_stub():
    if "mmcv.utils.config" in sys.modules:
        return
    mmcv = types.ModuleType("mmcv")
    utils = types.ModuleType("mmcv.utils")
    config = types.ModuleType("mmcv.utils.config")
    config.ConfigDict = ConfigDict
    utils.config = config
    mmcv.utils = utils
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = utils
    sys.modules["mmcv.utils.config"] = config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate planning metrics from results_nusc.pkl files.")
    parser.add_argument("results", nargs="+", help="results_nusc.pkl paths")
    parser.add_argument("--labels", nargs="*", help="Optional labels")
    parser.add_argument(
        "--out",
        default="paper_eval/results_nusc_planning_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Use all samples. By default samples with any all-zero GT future step are skipped.",
    )
    return parser.parse_args()


def load_result(path):
    install_mmcv_stub()
    with open(path, "rb") as f:
        result = pickle.load(f)
    if not isinstance(result, dict) or "plan_results" not in result or "plan_gts" not in result:
        raise TypeError(f"{path} does not look like a formatted results_nusc.pkl")
    return result


def to_xy_tensor(value):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.float().reshape(-1, 2)


def infer_valid(gt_traj):
    # Matches the 5119/6019 valid sample count in results.pkl for this project.
    return bool((gt_traj.abs().sum(dim=-1) > 1e-8).all())


def command_index(cmd):
    cmd = cmd if torch.is_tensor(cmd) else torch.as_tensor(cmd)
    return int(torch.argmax(cmd.reshape(-1, 3)[0]).item())


def aggregate(path, include_invalid=False):
    result = load_result(path)
    plan_results = result["plan_results"]
    plan_gts = result["plan_gts"]

    sums = {key: 0.0 for key in METRIC_KEYS}
    used = 0
    skipped = 0

    for token, pred_pack in plan_results.items():
        pred_all, cmd = pred_pack
        gt = to_xy_tensor(plan_gts[token])
        if not include_invalid and not infer_valid(gt):
            skipped += 1
            continue

        pred_all = pred_all if torch.is_tensor(pred_all) else torch.as_tensor(pred_all)
        pred = pred_all.float()[command_index(cmd)].reshape(-1, 2)

        pred = pred.cumsum(dim=-2)
        gt = gt.cumsum(dim=-2)

        used += 1
        for sec in (1, 2, 3):
            steps = sec * 2
            distances = torch.sqrt(((pred[:steps, :2] - gt[:steps, :2]) ** 2).sum(dim=-1))
            sums[f"plan_L2_{sec}s"] += float(distances.mean())
            sums[f"plan_L2_stp3_{sec}s"] += float(distances[-1])

    if used == 0:
        raise ValueError(f"{path} has no usable samples")

    metrics = {key: value / used for key, value in sums.items()}
    metrics["L2_avg"] = sum(metrics[f"plan_L2_{sec}s"] for sec in (1, 2, 3)) / 3
    metrics["valid_samples"] = used
    metrics["skipped_samples"] = skipped
    metrics["total_samples"] = len(plan_results)
    metrics["collision_metrics"] = "not_available_in_results_nusc"
    return metrics


def main():
    args = parse_args()
    if args.labels and len(args.labels) != len(args.results):
        raise ValueError("--labels must have the same length as results")

    labels = args.labels or [Path(path).parent.parent.name for path in args.results]
    rows = []
    for label, path in zip(labels, args.results):
        metrics = aggregate(path, include_invalid=args.include_invalid)
        metrics["label"] = label
        metrics["path"] = str(Path(path).resolve())
        rows.append(metrics)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "path",
        "valid_samples",
        "skipped_samples",
        "total_samples",
        *METRIC_KEYS,
        "L2_avg",
        "collision_metrics",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)

    print(f"Wrote {out}")
    for row in rows:
        print(
            f"{row['label']}: valid={row['valid_samples']}/{row['total_samples']}, "
            f"L2_avg={row['L2_avg']:.6f}, collision_metrics={row['collision_metrics']}"
        )


if __name__ == "__main__":
    main()
