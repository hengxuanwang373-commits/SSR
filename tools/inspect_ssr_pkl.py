#!/usr/bin/env python
"""Inspect the top-level structure and key sample fields in an SSR pickle."""

import argparse
import pickle
from collections.abc import Mapping, Sequence


DEFAULT_PKL = "data/nuscenes/vad_nuscenes_infos_temporal_train.pkl"
FIELDS_TO_INSPECT = (
    "ego_fut_cmd",
    "ego_fut_trajs",
    "ego_fut_masks",
    "gt_ego_fut_cmd",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "scene_token",
    "scene_name",
    "token",
    "timestamp",
)


def format_keys(obj):
    if isinstance(obj, Mapping):
        return [str(key) for key in obj.keys()]
    return None


def get_shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)

    if isinstance(value, (str, bytes)):
        return None

    if isinstance(value, Sequence):
        shape = []
        current = value
        while isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(shape)

    return None


def preview_value(value, max_items=8):
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return repr(value)

    flatten = getattr(value, "flatten", None)
    if callable(flatten):
        flat = flatten()
        try:
            values = flat[:max_items].tolist()
        except AttributeError:
            values = list(flat[:max_items])
        return repr(values)

    if isinstance(value, Mapping):
        keys = list(value.keys())[:max_items]
        return "{" + ", ".join(repr(key) for key in keys) + "}"

    if isinstance(value, Sequence):
        return repr(list(value[:max_items]))

    return repr(value)[:200]


def print_field(name, sample):
    if not isinstance(sample, Mapping) or name not in sample:
        return

    value = sample[name]
    dtype = getattr(value, "dtype", None)
    shape = get_shape(value)

    print(f"  - {name}")
    print(f"      type: {type(value).__name__}")
    if dtype is not None:
        print(f"      dtype: {dtype}")
    print(f"      shape: {shape}")
    print(f"      preview: {preview_value(value)}")


def get_first_sample(data):
    if isinstance(data, Mapping) and "infos" in data:
        infos = data["infos"]
        print("\n[infos]")
        print(f"  type: {type(infos).__name__}")
        print(f"  length: {len(infos)}")
        return infos[0] if infos else None

    if isinstance(data, list):
        print("\n[list]")
        print(f"  length: {len(data)}")
        return data[0] if data else None

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default=DEFAULT_PKL, help=f"Path to pkl file. Default: {DEFAULT_PKL}")
    args = parser.parse_args()

    print(f"[file]\n  path: {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    print("\n[top-level]")
    print(f"  type: {type(data).__name__}")
    keys = format_keys(data)
    if keys is not None:
        print(f"  keys ({len(keys)}):")
        for key in keys:
            print(f"    - {key}")

    sample = get_first_sample(data)
    if sample is None:
        print("\n[sample 0]\n  no sample found")
        return

    print("\n[sample 0]")
    print(f"  type: {type(sample).__name__}")
    sample_keys = format_keys(sample)
    if sample_keys is not None:
        print(f"  keys ({len(sample_keys)}):")
        for key in sample_keys:
            print(f"    - {key}")

    print("\n[selected fields]")
    found = False
    for field in FIELDS_TO_INSPECT:
        if isinstance(sample, Mapping) and field in sample:
            found = True
        print_field(field, sample)
    if not found:
        print("  none of the selected fields were found")


if __name__ == "__main__":
    main()
