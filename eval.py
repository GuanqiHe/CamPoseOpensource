#!/usr/bin/env python3
"""Evaluate one checkpoint on a selected held-out sign configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_jacobian.dataset import JointFlipPairedDataset, JointFlipSource
from action_jacobian.models.policy import DeterministicDinoACTPolicy
from train import action_metrics, load_sign_dr_inputs, rollout_success


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    args = SimpleNamespace(
        checkpoint=to_absolute_path(cfg.paths.checkpoint),
        cache=to_absolute_path(cfg.paths.cache),
        dataset=to_absolute_path(cfg.paths.dataset),
        design=to_absolute_path(cfg.paths.design),
        manifest=to_absolute_path(cfg.paths.manifest),
        dinov3_model_path=to_absolute_path(cfg.paths.dinov3_model),
        output_dir=(
            None
            if cfg.paths.output_dir is None
            else to_absolute_path(cfg.paths.output_dir)
        ),
        config_id=str(cfg.config_id),
        device=str(cfg.runtime.device),
        batch_size=int(cfg.evaluation.batch_size),
        rollout_seeds=int(cfg.evaluation.rollout_seeds),
        rollout_horizon=int(cfg.evaluation.rollout_horizon),
        rollout_videos=int(cfg.evaluation.rollout_videos),
        skip_rollout=bool(cfg.evaluation.skip_rollout),
    )

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
            raise ValueError(
                "paths.output_dir is required unless evaluation.skip_rollout=true"
            )
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
