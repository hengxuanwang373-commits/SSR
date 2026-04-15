# Copyright (c) OpenMMLab. All rights reserved.
"""
Post-processing script to compute Planning Metrics (L2AVG, CRAVG) from results.pkl

Usage:
    python tools/analysis_tools/compute_planning_metrics.py --result results.pkl
"""
import argparse
import copy

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Planning Metrics from results.pkl')
    parser.add_argument('--result', required=True, help='path to results.pkl')
    parser.add_argument(
        '--verbose', action='store_true', help='print detailed metrics')
    args = parser.parse_args()
    return args


def compute_planning_metrics(results):
    """Compute planning metrics from results.

    Args:
        results: List of result dicts containing 'metric_results'

    Returns:
        dict: Aggregated planning metrics
    """
    metric_dict = None
    num_valid = 0

    for res in results:
        if res.get('metric_results', None) is None:
            print(f"Warning: result {res} does not have 'metric_results' key")
            continue

        if not res['metric_results'].get('fut_valid_flag', False):
            continue

        num_valid += 1
        if metric_dict is None:
            metric_dict = copy.deepcopy(res['metric_results'])
        else:
            for k in res['metric_results'].keys():
                if k != 'fut_valid_flag':
                    metric_dict[k] += res['metric_results'][k]

    if num_valid == 0:
        print("Warning: No valid samples found!")
        return {}

    # Average
    for k in metric_dict:
        if k != 'fut_valid_flag':
            metric_dict[k] = metric_dict[k] / num_valid

    return metric_dict


def print_planning_metrics(metric_dict):
    """Print planning metrics in the standard format.

    Args:
        metric_dict: dict of aggregated metrics
    """
    print('\n' + '=' * 50)
    print('         Planning Metrics Summary')
    print('=' * 50)

    # L2AVG (Average Displacement Error in meters)
    print('\n--- L2AVG (m) ---')
    for t in [1, 2, 3]:
        key = f'plan_L2_{t}s'
        if key in metric_dict:
            print(f"  L2AVG (m) {t}s: {metric_dict[key]:.4f}")

    # L2AVG Average
    l2_keys = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s']
    if all(k in metric_dict for k in l2_keys):
        avg_l2 = sum(metric_dict[k] for k in l2_keys) / len(l2_keys)
        print(f"  L2AVG (m) Avg: {avg_l2:.4f}")

    # CRAVG (Collision Rate in percentage)
    print('\n--- CRAVG (%) ---')
    for t in [1, 2, 3]:
        key = f'plan_obj_col_{t}s'
        if key in metric_dict:
            cravg = metric_dict[key] * 100  # Convert to percentage
            print(f"  CRAVG (%) {t}s: {cravg:.2f}")

    # CRAVG Average
    col_keys = ['plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s']
    if all(k in metric_dict for k in col_keys):
        avg_col = sum(metric_dict[k] for k in col_keys) / len(col_keys) * 100
        print(f"  CRAVG (%) Avg: {avg_col:.2f}")

    # Additional metrics: obj_box_col (object box collision)
    print('\n--- Additional Metrics ---')
    for t in [1, 2, 3]:
        key = f'plan_obj_box_col_{t}s'
        if key in metric_dict:
            print(f"  plan_obj_box_col_{t}s: {metric_dict[key]:.4f}")

    # STP3 style metrics
    print('\n--- STP3 Style Metrics (FDE) ---')
    for t in [1, 2, 3]:
        key = f'plan_L2_stp3_{t}s'
        if key in metric_dict:
            print(f"  L2_stp3 {t}s: {metric_dict[key]:.4f}")

    print('\n' + '=' * 50)


def main():
    args = parse_args()

    print(f"Loading results from: {args.result}")
    results = mmcv.load(args.result)

    if isinstance(results, dict) and 'bbox_results' in results:
        results = results['bbox_results']

    print(f"Total results loaded: {len(results)}")

    metric_dict = compute_planning_metrics(results)

    if metric_dict:
        print_planning_metrics(metric_dict)

        # Print in CSV-friendly format
        print('\n--- CSV Format ---')
        print('Metric,Value')
        for k, v in metric_dict.items():
            if k != 'fut_valid_flag':
                print(f'{k},{v:.6f}')
    else:
        print("Failed to compute metrics.")


if __name__ == '__main__':
    main()
