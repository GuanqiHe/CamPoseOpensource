#!/usr/bin/env python3
"""Train matched RGB and RGB+pixel-Jacobian joint-flip policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader

import robosuite as suite
from models.deterministic_dinov3_act import DeterministicDinoACTPolicy
from pixel_action_jacobian import compute_pixel_action_jacobian
from pixel_jacobian_dataset import (
    JointFlipPairedDataset,
    JointFlipSource,
    PairedPhysicalBatchSampler,
)
from robosuite.wrappers.action_wrapper import wrap_env_action_space


HELDOUT_SIGNS = {"cfg5": (-1, 1, -1, 1, 1, 1, 1)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    repo = str(Path(__file__).resolve().parents[1])
    return subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()


def fit_action_stats(
    source: JointFlipSource,
    config_signs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    physical = np.flatnonzero(~source.validation_mask)
    signs = np.asarray(list(config_signs.values()), dtype=np.float32)
    actions = np.repeat(
        source.canonical_actions[physical, None, :], len(signs), axis=1
    )
    actions[..., :7] *= signs[None]
    flat = actions.reshape(-1, 8)
    return {
        "action_mean": flat.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(flat.std(axis=0), 1e-6).astype(np.float32),
        "action_min": flat.min(axis=0).astype(np.float32),
        "action_max": flat.max(axis=0).astype(np.float32),
    }


def model_batch(batch: dict[str, torch.Tensor], condition: str, device):
    output = {
        "image": batch["image"].to(device, non_blocking=True),
    }
    if condition == "pixel_jacobian":
        output["pixel_jacobian"] = batch["pixel_jacobian"].to(
            device, non_blocking=True
        )
    return output


@torch.no_grad()
def action_metrics(
    model: DeterministicDinoACTPolicy,
    dataset: JointFlipPairedDataset,
    condition: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    abs_error = 0.0
    valid_values = 0
    arm_correct = 0
    arm_values = 0
    gripper_correct = 0
    gripper_values = 0
    for batch in loader:
        prediction = model(model_batch(batch, condition, device))
        target = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        valid = (~is_pad).unsqueeze(-1).expand_as(target)
        error = torch.abs(prediction - target)
        abs_error += float(error[valid].sum())
        valid_values += int(valid.sum())

        mean = torch.from_numpy(dataset.action_mean).to(device)
        std = torch.from_numpy(dataset.action_std).to(device)
        raw_prediction = prediction * std[None, None] + mean[None, None]
        raw_target = batch["raw_actions"].to(device, non_blocking=True)
        arm_valid = (
            (~is_pad).unsqueeze(-1)
            & (torch.abs(raw_target[..., :7]) >= 0.002)
        )
        arm_correct += int(
            ((torch.sign(raw_prediction[..., :7]) == torch.sign(raw_target[..., :7]))
             & arm_valid).sum()
        )
        arm_values += int(arm_valid.sum())
        gripper_valid = (~is_pad) & (torch.abs(raw_target[..., 7]) >= 0.5)
        gripper_correct += int(
            ((torch.sign(raw_prediction[..., 7]) == torch.sign(raw_target[..., 7]))
             & gripper_valid).sum()
        )
        gripper_values += int(gripper_valid.sum())
    return {
        "normalized_action_mae": abs_error / max(valid_values, 1),
        "arm_sign_accuracy": arm_correct / max(arm_values, 1),
        "gripper_sign_accuracy": gripper_correct / max(gripper_values, 1),
    }


def build_env(dataset_path: str):
    with h5py.File(dataset_path, "r") as source:
        env_args = json.loads(source["data"].attrs["env_args"])
        action_space = source["data"].attrs["action_space"]
        if isinstance(action_space, bytes):
            action_space = action_space.decode()
    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
    )
    env = suite.make(env_name=env_args["env_name"], **kwargs)
    if action_space in ("eef_delta", "joint_delta"):
        env = wrap_env_action_space(env, action_space)
    return env


def render_agentview(env) -> np.ndarray:
    env.sim.forward()
    image = env.sim.render(camera_name="agentview", height=256, width=256, depth=False)
    return np.flipud(image).copy()


def rollout_success(
    model: DeterministicDinoACTPolicy,
    dataset_path: str,
    source: JointFlipSource,
    signs: np.ndarray,
    stats: dict[str, np.ndarray],
    condition: str,
    device: torch.device,
    seeds: int,
    horizon: int,
    videos: int,
    output_dir: Path,
) -> dict[str, float]:
    env = build_env(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = []
    model.eval()
    try:
        for seed in range(seeds):
            np.random.seed(seed)
            random.seed(seed)
            env.reset()
            env.set_init_action()
            frames = []
            success_step = None
            step = 0
            while step < horizon and success_step is None:
                image = render_agentview(env)
                if seed < videos:
                    frames.append(image)
                image_tensor = (
                    torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
                    .unsqueeze(0).to(device)
                )
                inputs = {"image": image_tensor}
                if condition == "pixel_jacobian":
                    field = compute_pixel_action_jacobian(
                        env, "agentview", grid_height=16, grid_width=16
                    ).field
                    field[:14] = (
                        field[:14].reshape(7, 2, 16, 16)
                        * signs[:, None, None, None]
                    ).reshape(14, 16, 16)
                    field[:14] /= source.jacobian_rms[:, None, None]
                    inputs["pixel_jacobian"] = torch.from_numpy(field).unsqueeze(0).to(device)
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    normalized = model(inputs)[0].float().cpu().numpy()
                action_chunk = normalized * stats["action_std"] + stats["action_mean"]
                action_chunk = np.clip(
                    action_chunk, stats["action_min"], stats["action_max"]
                )
                action_chunk[:, :7] *= signs[None]
                for action in action_chunk:
                    env.step(action)
                    step += 1
                    if bool(env._check_success()):
                        success_step = step
                        break
                    if step >= horizon:
                        break
                    if seed < videos:
                        frames.append(render_agentview(env))
            succeeded = success_step is not None
            successes.append(succeeded)
            if seed < videos:
                imageio.mimsave(
                    output_dir / f"seed_{seed:03d}_success_{int(succeeded)}.mp4",
                    frames,
                    fps=20,
                    codec="libx264",
                    quality=8,
                )
    finally:
        env.close()
    result = {
        "successes": int(sum(successes)),
        "num_seeds": seeds,
        "success_rate": float(np.mean(successes)),
        "horizon": horizon,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    source = JointFlipSource(
        args.cache,
        expected_demos=args.expected_demos,
        expected_physical_steps=args.expected_physical_steps,
    )
    train_signs = {
        config_id: source.cache_config_signs[config_id]
        for config_id in args.train_configs
    }
    heldout_signs = {args.heldout_config: np.asarray(HELDOUT_SIGNS[args.heldout_config], dtype=np.float32)}
    stats = fit_action_stats(source, train_signs)
    include_structural = args.condition == "pixel_jacobian"
    train_set = JointFlipPairedDataset(
        source, train_signs, args.chunk_size, "train", include_structural, **stats
    )
    val_set = JointFlipPairedDataset(
        source, train_signs, args.chunk_size, "val", include_structural, **stats
    )
    heldout_set = JointFlipPairedDataset(
        source, heldout_signs, args.chunk_size, "val", include_structural, **stats
    )
    args.action_dim = 8
    args.matched_jacobian_adapter = True
    model = DeterministicDinoACTPolicy(args).to(device)
    optimizer = model.configure_optimizers()
    sampler = PairedPhysicalBatchSampler(
        train_set, args.physical_batch_size, args.steps, args.seed
    )
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        git_commit=git_commit(),
        cache_sha256=sha256_file(args.cache),
        train_configs=args.train_configs,
        train_signs={key: value.tolist() for key, value in train_signs.items()},
        heldout_signs={key: value.tolist() for key, value in heldout_signs.items()},
        train_samples=len(train_set),
        val_samples=len(val_set),
        heldout_samples=len(heldout_set),
        action_stats_source="train physical frames and train configurations only",
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with open(run_dir / "dataset_stats.json", "w") as handle:
        json.dump({key: value.tolist() for key, value in stats.items()}, handle, indent=2)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=config,
    )
    samples_seen = 0
    best_success = -1.0
    start_time = time.perf_counter()
    try:
        for step, batch in enumerate(loader, start=1):
            inputs = model_batch(batch, args.condition, device)
            target = batch["actions"].to(device, non_blocking=True)
            is_pad = batch["is_pad"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
                prediction = model(inputs)
                valid = (~is_pad).unsqueeze(-1).expand_as(target)
                loss = F.l1_loss(prediction[valid], target[valid])
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            samples_seen += len(batch["image"])
            if step % args.log_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                wandb.log(
                    {
                        "train/l1": float(loss.detach()),
                        "train/gradient_norm": float(gradient_norm),
                        "samples_seen": samples_seen,
                        "throughput_samples_s": samples_seen / elapsed,
                    },
                    step=step,
                )
            if step % args.eval_every == 0 or step == args.steps:
                train_metrics = action_metrics(model, val_set, args.condition, device, args.eval_batch_size)
                heldout_metrics = action_metrics(model, heldout_set, args.condition, device, args.eval_batch_size)
                metrics = {
                    "val_train/normalized_action_mae": train_metrics["normalized_action_mae"],
                    "val_heldout/normalized_action_mae": heldout_metrics["normalized_action_mae"],
                    "val_heldout/arm_sign_accuracy": heldout_metrics["arm_sign_accuracy"],
                    "val_heldout/gripper_sign_accuracy": heldout_metrics["gripper_sign_accuracy"],
                }
                wandb.log(metrics, step=step)
                torch.save(
                    {
                        "step": step,
                        "samples_seen": samples_seen,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "stats": stats,
                        "git_commit": config["git_commit"],
                        "wandb_run_id": run.id,
                    },
                    run_dir / "checkpoints" / f"step_{step:06d}.pth",
                )
            if step % args.rollout_every == 0 or step == args.steps:
                rollout = rollout_success(
                    model,
                    args.dataset,
                    source,
                    heldout_set.joint_signs[0],
                    stats,
                    args.condition,
                    device,
                    args.rollout_seeds,
                    args.rollout_horizon,
                    args.rollout_videos,
                    run_dir / "rollouts" / f"step_{step:06d}",
                )
                wandb.log({"rollout_heldout/success_rate": rollout["success_rate"]}, step=step)
                if rollout["success_rate"] > best_success:
                    best_success = rollout["success_rate"]
                    run.summary["best_heldout_success_rate"] = best_success
                    run.summary["best_step"] = step
    finally:
        run.summary["samples_seen"] = samples_seen
        run.summary["git_commit"] = config["git_commit"]
        run.summary["best_heldout_success_rate"] = best_success
        wandb.finish()
    return {"samples_seen": samples_seen, "best_heldout_success_rate": best_success}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition", choices=("none", "pixel_jacobian"), required=True)
    parser.add_argument("--train-configs", nargs="+", default=["cfg0", "cfg1", "cfg2", "cfg3", "cfg4"])
    parser.add_argument("--heldout-config", choices=("cfg5",), default="cfg5")
    parser.add_argument("--expected-demos", type=int, default=200)
    parser.add_argument("--expected-physical-steps", type=int, default=17937)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--rollout-every", type=int, default=5000)
    parser.add_argument("--rollout-seeds", type=int, default=50)
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--rollout-videos", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--pre-norm", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--wandb-entity", default="wuji-tech")
    parser.add_argument("--wandb-project", default="pixel-action-jacobian")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
