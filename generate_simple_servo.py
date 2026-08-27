#!/usr/bin/env python3
"""Generate paired 3-DoF visual-servo states for sign generalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from action_jacobian.simple_servo import OOD_SIGNS, TRAIN_SIGNS, make_sample, rollout_oracle, sample_state


def downsample_field(field: np.ndarray) -> np.ndarray:
    return field.reshape(7, 16, 4, 16, 4).mean(axis=(2, 4)).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--preview-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = Path(args.preview_dir)
    preview.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    q_values, targets, images, fields, descriptors, actions = [], [], [], [], [], []
    oracle_results = []
    for index in range(args.physical_samples):
        q, target = sample_state(rng)
        q_values.append(q)
        targets.append(target)
        images.append(make_sample(q, target, TRAIN_SIGNS[0]).image)
        config_fields, config_descriptors, config_actions = [], [], []
        for signs in np.concatenate([TRAIN_SIGNS, OOD_SIGNS]):
            sample = make_sample(q, target, signs)
            config_fields.append(downsample_field(sample.pixel_jacobian))
            config_descriptors.append(sample.global_jacobian)
            config_actions.append(sample.action)
        fields.append(config_fields)
        descriptors.append(config_descriptors)
        actions.append(config_actions)
        if index < 100:
            for config_index, signs in enumerate(np.concatenate([TRAIN_SIGNS, OOD_SIGNS])):
                success, error, steps = rollout_oracle(q, target, signs)
                oracle_results.append((config_index, success, error, steps))
        if index < 12:
            imageio.imwrite(preview / f"sample_{index:03d}.png", images[-1])
    with h5py.File(output, "w") as dataset:
        dataset.attrs["seed"] = args.seed
        dataset.attrs["physical_samples"] = args.physical_samples
        dataset.create_dataset("q", data=np.stack(q_values), compression="gzip")
        dataset.create_dataset("target", data=np.stack(targets), compression="gzip")
        dataset.create_dataset("rgb", data=np.stack(images), compression="gzip")
        dataset.create_dataset("pixel_jacobian", data=np.asarray(fields), compression="gzip")
        dataset.create_dataset("global_jacobian", data=np.asarray(descriptors), compression="gzip")
        dataset.create_dataset("actions", data=np.asarray(actions), compression="gzip")
        dataset.create_dataset("train_signs", data=TRAIN_SIGNS)
        dataset.create_dataset("ood_signs", data=OOD_SIGNS)
    success = np.asarray([value[1] for value in oracle_results], dtype=bool)
    metrics = {
        "physical_samples": args.physical_samples,
        "train_labels": args.physical_samples * len(TRAIN_SIGNS),
        "oracle_rollouts": len(oracle_results),
        "oracle_success_rate": float(success.mean()),
        "oracle_max_final_error_px": float(max(value[2] for value in oracle_results)),
    }
    (preview / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if metrics["oracle_success_rate"] < 0.98:
        raise RuntimeError(f"Oracle gate failed: {metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
