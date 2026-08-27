#!/usr/bin/env python3
"""Evaluate one checkpoint on a selected held-out sign configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_jacobian.dataset import JointFlipPairedDataset, JointFlipSource
from action_jacobian.models.policy import DeterministicDinoACTPolicy
from train import action_metrics, load_sign_dr_inputs, rollout_success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-seeds", type=int, default=50)
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--rollout-videos", type=int, default=3)
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config["dinov3_model_path"] = args.dinov3_model_path
    config["device"] = args.device
    model = DeterministicDinoACTPolicy(SimpleNamespace(**config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    source = JointFlipSource(
        args.cache,
        expected_demos=int(config["expected_demos"]),
        expected_physical_steps=int(config["expected_physical_steps"]),
    )
    _, ood_signs, manifest = load_sign_dr_inputs(
        args.design, args.manifest, source
    )
    if args.config_id not in ood_signs:
        raise ValueError(
            f"Unknown OOD config {args.config_id!r}; choose from {sorted(ood_signs)}"
        )
    demo_to_index = {
        demo_name: index for index, demo_name in enumerate(source.demo_names)
    }
    val_demo_indexes = np.asarray(
        [demo_to_index[name] for name in manifest["val_demo_names"]],
        dtype=np.int64,
    )
    val_physical_indexes = np.flatnonzero(
        np.isin(source.demo_index_by_physical, val_demo_indexes)
    )
    condition = str(config["condition"])
    stats = {
        key: np.asarray(checkpoint["stats"][key], dtype=np.float32)
        for key in ("action_mean", "action_std", "action_min", "action_max")
    }
    selected_signs = {args.config_id: ood_signs[args.config_id]}
    evaluation_set = JointFlipPairedDataset(
        source,
        selected_signs,
        int(config["chunk_size"]),
        "val",
        include_structural=condition == "pixel_jacobian",
        include_global_jacobian=condition == "global_token",
        include_sign_array=condition == "sign_array",
        physical_indexes=val_physical_indexes,
        **stats,
    )
    result = {
        "checkpoint_step": int(checkpoint["step"]),
        "condition": condition,
        "config_id": args.config_id,
        "offline": action_metrics(
            model, evaluation_set, condition, device, args.batch_size
        ),
    }
    if not args.skip_rollout:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --skip-rollout is set")
        result["rollout"] = rollout_success(
            model,
            args.dataset,
            source,
            evaluation_set.joint_signs[0],
            stats,
            condition,
            device,
            args.rollout_seeds,
            args.rollout_horizon,
            args.rollout_videos,
            Path(args.output_dir),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
