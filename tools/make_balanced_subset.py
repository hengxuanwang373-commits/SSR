#!/usr/bin/env python
"""Create a scene-aware, command-balanced subset from an SSR nuScenes info pkl."""

import argparse
import copy
import os
import pickle
import random
from collections import Counter, defaultdict, deque


COMMANDS = (0, 1, 2)
FALLBACK_LATERAL_THRESH = 1.0


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def to_list(value):
    if value is None:
        return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def sum_mask(mask):
    if mask is None:
        return 0
    value = to_list(mask)
    if isinstance(value, (int, float, bool)):
        return int(value)
    return int(sum(float(x) for x in flatten(value)))


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def argmax(values):
    values = list(flatten(to_list(values)))
    if not values:
        return None
    return max(range(len(values)), key=lambda idx: values[idx])


def last_valid_point(trajs, masks):
    trajs = to_list(trajs)
    masks = list(flatten(to_list(masks))) if masks is not None else None
    if trajs is None:
        return None

    valid_indices = []
    if masks is None:
        valid_indices = list(range(len(trajs)))
    else:
        valid_indices = [idx for idx, flag in enumerate(masks) if float(flag) > 0]
    if not valid_indices:
        return None

    idx = valid_indices[-1]
    if idx >= len(trajs):
        return None
    point = trajs[idx]
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return point


def infer_command(sample):
    """Return command label 0/1/2.

    Prefer gt_ego_fut_cmd. If absent, use the final valid trajectory point's
    lateral displacement as a coarse fallback: left/straight/right = 0/1/2.
    """
    cmd = sample.get("gt_ego_fut_cmd")
    if cmd is not None:
        label = argmax(cmd)
        if label in COMMANDS:
            return label
        return None

    point = last_valid_point(sample.get("gt_ego_fut_trajs"), sample.get("gt_ego_fut_masks"))
    if point is None:
        return None

    lateral = float(point[0])
    if lateral < -FALLBACK_LATERAL_THRESH:
        return 0
    if lateral > FALLBACK_LATERAL_THRESH:
        return 2
    return 1


def is_valid_future(sample, min_valid_fut):
    return sum_mask(sample.get("gt_ego_fut_masks")) >= min_valid_fut


def scene_token(sample):
    return sample.get("scene_token") or "__missing_scene_token__"


def scene_stats(samples):
    counts = Counter(scene_token(sample) for sample in samples)
    if not counts:
        return {"num_scenes": 0, "max_per_scene": 0, "avg_per_scene": 0.0}
    return {
        "num_scenes": len(counts),
        "max_per_scene": max(counts.values()),
        "avg_per_scene": sum(counts.values()) / len(counts),
    }


def command_counts(records):
    return Counter(record["command"] for record in records)


def print_stats(title, total_count, records):
    samples = [record["sample"] for record in records]
    scenes = scene_stats(samples)
    counts = command_counts(records)

    print(f"\n[{title}]")
    print(f"  total samples: {total_count}")
    print(f"  valid samples: {len(records)}")
    print("  command counts:")
    for label in COMMANDS:
        print(f"    command {label}: {counts.get(label, 0)}")
    print(f"  scene count: {scenes['num_scenes']}")
    print(f"  max samples per scene: {scenes['max_per_scene']}")
    print(f"  avg samples per scene: {scenes['avg_per_scene']:.2f}")


def build_valid_records(infos, min_valid_fut):
    records = []
    invalid_future = 0
    missing_command = 0

    for idx, sample in enumerate(infos):
        if not isinstance(sample, dict):
            continue
        if not is_valid_future(sample, min_valid_fut):
            invalid_future += 1
            continue

        command = infer_command(sample)
        if command not in COMMANDS:
            missing_command += 1
            continue

        records.append(
            {
                "index": idx,
                "sample": sample,
                "command": command,
                "scene": scene_token(sample),
            }
        )

    return records, invalid_future, missing_command


def balanced_quotas(capacity, target):
    quotas = {label: 0 for label in COMMANDS}
    target = min(target, sum(capacity.values()))

    while sum(quotas.values()) < target:
        candidates = [label for label in COMMANDS if quotas[label] < capacity.get(label, 0)]
        if not candidates:
            break
        label = min(candidates, key=lambda x: (quotas[x], -capacity.get(x, 0), x))
        quotas[label] += 1

    return quotas


class CommandSceneSampler:
    def __init__(self, records, rng):
        by_scene = defaultdict(list)
        for record in records:
            by_scene[record["scene"]].append(record)

        scene_items = []
        for scene, items in by_scene.items():
            rng.shuffle(items)
            scene_items.append((scene, deque(items)))
        rng.shuffle(scene_items)
        self.scenes = deque(scene_items)

    def pop(self, scene_counts, max_per_scene):
        if not self.scenes:
            return None

        attempts = len(self.scenes)
        while attempts > 0:
            scene, items = self.scenes.popleft()
            attempts -= 1

            if scene_counts[scene] >= max_per_scene:
                if items:
                    self.scenes.append((scene, items))
                continue

            record = items.popleft()
            if items:
                self.scenes.append((scene, items))
            return record

        return None


def sample_balanced(records, target, max_per_scene, seed):
    rng = random.Random(seed)
    by_command = {label: [] for label in COMMANDS}
    for record in records:
        by_command[record["command"]].append(record)

    capacity = {label: len(by_command[label]) for label in COMMANDS}
    quotas = balanced_quotas(capacity, target)
    samplers = {
        label: CommandSceneSampler(by_command[label], rng)
        for label in COMMANDS
    }

    selected = []
    selected_by_command = Counter()
    scene_counts = Counter()

    while len(selected) < target:
        candidates = [label for label in COMMANDS if selected_by_command[label] < quotas[label]]
        if not candidates:
            break

        progressed = False
        candidates.sort(key=lambda x: (selected_by_command[x], x))
        for label in candidates:
            if len(selected) >= target or selected_by_command[label] >= quotas[label]:
                continue
            record = samplers[label].pop(scene_counts, max_per_scene)
            if record is None:
                quotas[label] = selected_by_command[label]
                continue
            selected.append(record)
            selected_by_command[label] += 1
            scene_counts[record["scene"]] += 1
            progressed = True

        if not progressed:
            break

    selected.sort(key=lambda record: record["index"])
    return selected, quotas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source pkl path.")
    parser.add_argument("--out", required=True, help="Output pkl path.")
    parser.add_argument("--ratio", required=True, type=float, help="Sampling ratio, e.g. 0.1.")
    parser.add_argument("--max-per-scene", type=int, default=80, help="Max selected samples per scene.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-valid-fut", type=int, default=6, help="Minimum valid future points.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    return parser.parse_args()


def main():
    args = parse_args()

    src_abs = os.path.abspath(args.src)
    out_abs = os.path.abspath(args.out)
    if src_abs == out_abs:
        raise SystemExit(f"Refusing to write output to the source pkl path: {src_abs}")
    if os.path.exists(out_abs) and not args.overwrite:
        raise SystemExit(f"Output already exists: {out_abs}. Pass --overwrite to replace it.")
    if not (0 < args.ratio <= 1):
        raise SystemExit("--ratio must be in the range (0, 1].")
    if args.max_per_scene <= 0:
        raise SystemExit("--max-per-scene must be > 0.")
    if args.min_valid_fut < 0:
        raise SystemExit("--min-valid-fut must be >= 0.")

    print(f"[input]\n  src: {src_abs}\n  out: {out_abs}")
    data = load_pkl(src_abs)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected top-level dict, got {type(data).__name__}.")
    if "infos" not in data:
        raise SystemExit("Expected top-level key 'infos'.")
    infos = data["infos"]
    if not isinstance(infos, list):
        raise SystemExit(f"Expected 'infos' to be list, got {type(infos).__name__}.")

    valid_records, invalid_future, missing_command = build_valid_records(infos, args.min_valid_fut)
    target = min(int(round(len(infos) * args.ratio)), len(valid_records))

    print_stats("before", len(infos), valid_records)
    print(f"  skipped by future mask: {invalid_future}")
    print(f"  skipped by missing command label: {missing_command}")
    print(f"\n[sampling]\n  ratio: {args.ratio}\n  target samples: {target}")
    print(f"  max per scene: {args.max_per_scene}")
    print(f"  seed: {args.seed}")

    selected_records, quotas = sample_balanced(
        valid_records,
        target=target,
        max_per_scene=args.max_per_scene,
        seed=args.seed,
    )
    selected_infos = [record["sample"] for record in selected_records]

    print("  command quotas:")
    for label in COMMANDS:
        print(f"    command {label}: {quotas.get(label, 0)}")
    print_stats("after", len(selected_infos), selected_records)

    out_data = copy.copy(data)
    out_data["infos"] = selected_infos
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    dump_pkl(out_data, out_abs)
    print(f"\n[done]\n  wrote: {out_abs}")


if __name__ == "__main__":
    main()
