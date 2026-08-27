#!/usr/bin/env python3
"""Build a balanced episode and sign-configuration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _validate_split(name: str, configs: dict[str, list[int]]) -> None:
    signs = np.asarray(list(configs.values()), dtype=np.int8)
    if signs.shape != (8, 7):
        raise ValueError(f"{name} must contain eight 7D sign vectors")
    if not np.all(np.isin(signs, (-1, 1))):
        raise ValueError(f"{name} signs must contain only -1 and +1")
    if len({tuple(row) for row in signs}) != len(signs):
        raise ValueError(f"{name} contains duplicate sign vectors")
    if not np.all((signs < 0).sum(axis=0) == 4):
        raise ValueError(f"{name} must be 50/50 balanced for every joint")
    distances = [
        int(np.count_nonzero(signs[i] != signs[j]))
        for i in range(len(signs))
        for j in range(i + 1, len(signs))
    ]
    if set(distances) != {4}:
        raise ValueError(f"{name} pairwise Hamming distance must equal four")


def _balanced_pairs(
    physical_indexes: np.ndarray,
    config_ids: list[str],
    rng: np.random.Generator,
) -> dict[int, list[str]]:
    """Assign two configs per frame with near-exact global config balance."""

    assignments: dict[int, list[str]] = {}
    cursor = 0
    while cursor < len(physical_indexes):
        shuffled = rng.permutation(config_ids).tolist()
        for offset in range(0, len(shuffled), 2):
            if cursor >= len(physical_indexes):
                break
            assignments[int(physical_indexes[cursor])] = sorted(
                shuffled[offset : offset + 2]
            )
            cursor += 1
    return assignments


def _sampling_stats(
    sampled: list[list[str]],
    physical_indexes: np.ndarray,
    configs: dict[str, list[int]],
) -> dict:
    selected = [sampled[int(index)] for index in physical_indexes]
    counts = {
        config_id: sum(config_id in pair for pair in selected)
        for config_id in configs
    }
    signs = np.asarray(
        [configs[config_id] for pair in selected for config_id in pair],
        dtype=np.int8,
    )
    return {
        "config_counts": counts,
        "negative_fraction_per_joint": (signs < 0).mean(axis=0).tolist(),
    }


def build_manifest(args: argparse.Namespace) -> dict:
    design_path = Path(args.design).resolve()
    design = json.loads(design_path.read_text())
    train_configs = design["train"]
    ood_configs = design["ood"]
    _validate_split("train", train_configs)
    _validate_split("ood", ood_configs)
    if set(map(tuple, train_configs.values())) & set(map(tuple, ood_configs.values())):
        raise ValueError("Train and OOD sign vectors must be disjoint")

    with h5py.File(args.cache, "r") as cache:
        demo_names = sorted(cache["demos"])
        frame_counts = [len(cache[f"demos/{name}/rgb"]) for name in demo_names]
    if len(demo_names) != args.expected_demos:
        raise ValueError(
            f"Expected {args.expected_demos} demos, found {len(demo_names)}"
        )
    if sum(frame_counts) != args.expected_physical_steps:
        raise ValueError(
            f"Expected {args.expected_physical_steps} frames, found {sum(frame_counts)}"
        )

    split_rng = np.random.default_rng(args.split_seed)
    val_demo_indexes = np.sort(
        split_rng.choice(len(demo_names), size=args.val_demos, replace=False)
    )
    val_demo_set = set(int(index) for index in val_demo_indexes)
    demo_index_by_physical = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(frame_counts)]
    )
    train_physical = np.flatnonzero(
        ~np.isin(demo_index_by_physical, val_demo_indexes)
    )
    val_physical = np.flatnonzero(
        np.isin(demo_index_by_physical, val_demo_indexes)
    )
    sampling_rng = np.random.default_rng(args.sampling_seed)
    assignments = _balanced_pairs(
        train_physical, list(train_configs), sampling_rng
    )
    assignments.update(
        _balanced_pairs(val_physical, list(train_configs), sampling_rng)
    )
    sampled = [assignments[index] for index in range(sum(frame_counts))]

    train_sampling = _sampling_stats(sampled, train_physical, train_configs)
    val_sampling = _sampling_stats(sampled, val_physical, train_configs)
    return {
        "version": 1,
        "design_path": str(design_path),
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
        "split_seed": args.split_seed,
        "sampling_seed": args.sampling_seed,
        "configs_per_frame": 2,
        "train_configs": train_configs,
        "ood_configs": ood_configs,
        "train_demo_names": [
            name for index, name in enumerate(demo_names) if index not in val_demo_set
        ],
        "val_demo_names": [demo_names[index] for index in val_demo_indexes],
        "train_physical_steps": int(len(train_physical)),
        "val_physical_steps": int(len(val_physical)),
        "num_physical_steps": int(sum(frame_counts)),
        "train_sampling": train_sampling,
        "val_sampling": val_sampling,
        "sampled_train_config_ids_by_physical": sampled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-demos", type=int, default=200)
    parser.add_argument("--expected-physical-steps", type=int, default=17937)
    parser.add_argument("--val-demos", type=int, default=40)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--sampling-seed", type=int, default=20260827)
    args = parser.parse_args()
    manifest = build_manifest(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "num_physical_steps",
                    "train_physical_steps",
                    "val_physical_steps",
                    "train_sampling",
                    "val_sampling",
                    "design_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
