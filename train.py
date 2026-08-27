#!/usr/bin/env python3
"""Train matched joint-flip policies with episode and configuration splits.

Each run consumes the same precomputed 160/40 episode split and balanced
two-configuration-per-physical-frame manifest. ID validation uses the eight
training sign conventions on held-out episodes; OOD validation uses the eight
disjoint OOD sign conventions on those same episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
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

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from action_jacobian.dataset import (
    GLOBAL_JACOBIAN_DIM,
    JointFlipPairedDataset,
    JointFlipSource,
    PairedPhysicalBatchSampler,
    global_jacobian_descriptor,
)
from action_jacobian.models.policy import DeterministicDinoACTPolicy
from action_jacobian.representation import compute_pixel_action_jacobian
from robosuite.wrappers.action_wrapper import wrap_env_action_space


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
    repo = str(REPO_ROOT)
    return subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()


def load_sign_dr_inputs(
    design_path: str,
    manifest_path: str,
    source: JointFlipSource,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    with open(design_path, encoding="utf-8") as handle:
        design = json.load(handle)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    design_sha256 = sha256_file(design_path)
    if manifest["design_sha256"] != design_sha256:
        raise ValueError("Manifest design SHA256 does not match --design")
    if manifest["num_physical_steps"] != source.num_physical_steps:
        raise ValueError("Manifest physical-step count does not match cache")
    if len(manifest["sampled_train_config_ids_by_physical"]) != source.num_physical_steps:
        raise ValueError("Manifest config sampling length does not match cache")
    if set(manifest["train_demo_names"]) & set(manifest["val_demo_names"]):
        raise ValueError("Train and validation demos overlap")
    if set(manifest["train_demo_names"] + manifest["val_demo_names"]) != set(
        source.demo_names
    ):
        raise ValueError("Manifest demos do not exactly cover the cache")
    if design["train"] != manifest["train_configs"]:
        raise ValueError("Manifest training configurations do not match design")
    if design["ood"] != manifest["ood_configs"]:
        raise ValueError("Manifest OOD configurations do not match design")

    train_signs = {
        config_id: np.asarray(signs, dtype=np.float32)
        for config_id, signs in design["train"].items()
    }
    ood_signs = {
        config_id: np.asarray(signs, dtype=np.float32)
        for config_id, signs in design["ood"].items()
    }
    return train_signs, ood_signs, manifest


def fit_action_stats(
    source: JointFlipSource,
    config_signs: dict[str, np.ndarray],
    physical_indexes: np.ndarray,
    sampled_config_ids_by_physical: list[list[str]],
) -> dict[str, np.ndarray]:
    action_parts = []
    for physical_index in physical_indexes:
        config_ids = sampled_config_ids_by_physical[int(physical_index)]
        signs = np.asarray(
            [config_signs[config_id] for config_id in config_ids],
            dtype=np.float32,
        )
        actions = np.repeat(
            source.canonical_actions[int(physical_index)][None],
            len(config_ids),
            axis=0,
        )
        actions[:, :7] *= signs
        action_parts.append(actions)
    flat = np.concatenate(action_parts, axis=0)
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
    elif condition == "global_token":
        output["global_jacobian"] = batch["global_jacobian"].to(
            device, non_blocking=True
        )
    elif condition == "sign_array":
        output["global_sign"] = batch["global_sign"].to(
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
                if condition in ("pixel_jacobian", "global_token"):
                    field = compute_pixel_action_jacobian(
                        env, "agentview", grid_height=16, grid_width=16
                    ).field
                    field[:14] = (
                        field[:14].reshape(7, 2, 16, 16)
                        * signs[:, None, None, None]
                    ).reshape(14, 16, 16)
                    field[:14] /= source.jacobian_rms[:, None, None]
                    if condition == "pixel_jacobian":
                        inputs["pixel_jacobian"] = (
                            torch.from_numpy(field).unsqueeze(0).to(device)
                        )
                    else:
                        inputs["global_jacobian"] = torch.from_numpy(
                            global_jacobian_descriptor(field)
                        ).unsqueeze(0).to(device)
                elif condition == "sign_array":
                    inputs["global_sign"] = torch.from_numpy(signs).unsqueeze(0).to(device)
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
    if args.rollout_seeds_per_config <= 0:
        raise ValueError("--rollout-seeds-per-config must be positive")
    device = torch.device(args.device)
    source = JointFlipSource(
        args.cache,
        expected_demos=args.expected_demos,
        expected_physical_steps=args.expected_physical_steps,
    )
    train_signs, heldout_signs, manifest = load_sign_dr_inputs(
        args.design,
        args.manifest,
        source,
    )
    if args.configs_per_frame != manifest["configs_per_frame"]:
        raise ValueError(
            f"--configs-per-frame={args.configs_per_frame} does not match "
            f"manifest value {manifest['configs_per_frame']}"
        )
    demo_to_index = {
        demo_name: index for index, demo_name in enumerate(source.demo_names)
    }
    train_demo_indexes = np.asarray(
        [demo_to_index[name] for name in manifest["train_demo_names"]],
        dtype=np.int64,
    )
    val_demo_indexes = np.asarray(
        [demo_to_index[name] for name in manifest["val_demo_names"]],
        dtype=np.int64,
    )
    train_physical_indexes = np.flatnonzero(
        np.isin(source.demo_index_by_physical, train_demo_indexes)
    )
    val_physical_indexes = np.flatnonzero(
        np.isin(source.demo_index_by_physical, val_demo_indexes)
    )
    if not 1 <= args.ood_eval_samples <= len(val_physical_indexes):
        raise ValueError(
            "ood_eval_samples must be in [1, val_physical_steps], got "
            f"{args.ood_eval_samples} for {len(val_physical_indexes)} val steps"
        )
    ood_eval_rng = np.random.default_rng(args.ood_eval_seed)
    ood_eval_physical_indexes = np.sort(
        ood_eval_rng.choice(
            val_physical_indexes,
            size=args.ood_eval_samples,
            replace=False,
        )
    )
    manifest["ood_eval_seed"] = int(args.ood_eval_seed)
    manifest["ood_eval_physical_indexes"] = ood_eval_physical_indexes.tolist()
    stats = fit_action_stats(
        source,
        train_signs,
        train_physical_indexes,
        manifest["sampled_train_config_ids_by_physical"],
    )
    include_structural = args.condition == "pixel_jacobian"
    include_global_jacobian = args.condition == "global_token"
    include_sign_array = args.condition == "sign_array"
    train_set = JointFlipPairedDataset(
        source,
        train_signs,
        args.chunk_size,
        "train",
        include_structural=include_structural,
        include_global_jacobian=include_global_jacobian,
        include_sign_array=include_sign_array,
        physical_indexes=train_physical_indexes,
        sampled_config_ids_by_physical=manifest["sampled_train_config_ids_by_physical"],
        **stats,
    )
    val_set = JointFlipPairedDataset(
        source,
        train_signs,
        args.chunk_size,
        "val",
        include_structural=include_structural,
        include_global_jacobian=include_global_jacobian,
        include_sign_array=include_sign_array,
        physical_indexes=val_physical_indexes,
        sampled_config_ids_by_physical=manifest["sampled_train_config_ids_by_physical"],
        **stats,
    )
    heldout_set = JointFlipPairedDataset(
        source,
        heldout_signs,
        args.chunk_size,
        "val",
        include_structural=include_structural,
        include_global_jacobian=include_global_jacobian,
        include_sign_array=include_sign_array,
        physical_indexes=val_physical_indexes,
        **stats,
    )
    heldout_eval_set = JointFlipPairedDataset(
        source,
        heldout_signs,
        args.chunk_size,
        "val",
        include_structural=include_structural,
        include_global_jacobian=include_global_jacobian,
        include_sign_array=include_sign_array,
        physical_indexes=ood_eval_physical_indexes,
        **stats,
    )
    args.action_dim = 8
    args.matched_jacobian_adapter = True
    args.global_jacobian_dim = GLOBAL_JACOBIAN_DIM
    args.sign_array_dim = 7
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
        design_sha256=sha256_file(args.design),
        input_manifest_sha256=sha256_file(args.manifest),
        train_configs=list(train_signs),
        heldout_configs=list(heldout_signs),
        train_signs={key: value.tolist() for key, value in train_signs.items()},
        heldout_signs={key: value.tolist() for key, value in heldout_signs.items()},
        train_samples=len(train_set),
        val_samples=len(val_set),
        heldout_samples=len(heldout_set),
        ood_eval_physical_steps=len(ood_eval_physical_indexes),
        ood_eval_samples=len(heldout_eval_set),
        train_physical_steps=len(train_physical_indexes),
        val_physical_steps=len(val_physical_indexes),
        samples_seen_target=(
            args.steps * args.physical_batch_size * args.configs_per_frame
        ),
        manifest_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        action_stats_source="train physical frames and train configurations only",
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with open(run_dir / "dataset_stats.json", "w") as handle:
        json.dump({key: value.tolist() for key, value in stats.items()}, handle, indent=2)
    with open(run_dir / "split_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

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
                id_metrics = action_metrics(
                    model, val_set, args.condition, device, args.eval_batch_size
                )
                ood_metrics = action_metrics(
                    model,
                    heldout_eval_set,
                    args.condition,
                    device,
                    args.eval_batch_size,
                )
                metrics = {
                    "val_id/normalized_action_mae": id_metrics["normalized_action_mae"],
                    "val_id/arm_sign_accuracy": id_metrics["arm_sign_accuracy"],
                    "val_id/gripper_sign_accuracy": id_metrics["gripper_sign_accuracy"],
                    "val_ood/normalized_action_mae": ood_metrics["normalized_action_mae"],
                    "val_ood/arm_sign_accuracy": ood_metrics["arm_sign_accuracy"],
                    "val_ood/gripper_sign_accuracy": ood_metrics["gripper_sign_accuracy"],
                    # Keep explicit test aliases visible in W&B.  These are
                    # fixed held-out evaluation sets, not training metrics.
                    "test_id/normalized_action_mae": id_metrics["normalized_action_mae"],
                    "test_id/arm_sign_accuracy": id_metrics["arm_sign_accuracy"],
                    "test_id/gripper_sign_accuracy": id_metrics["gripper_sign_accuracy"],
                    "test_ood/normalized_action_mae": ood_metrics["normalized_action_mae"],
                    "test_ood/arm_sign_accuracy": ood_metrics["arm_sign_accuracy"],
                    "test_ood/gripper_sign_accuracy": ood_metrics["gripper_sign_accuracy"],
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
                rollout_metrics = {}
                for config_id, signs in heldout_signs.items():
                    rollout_metrics[config_id] = rollout_success(
                        model,
                        args.dataset,
                        source,
                        signs,
                        stats,
                        args.condition,
                        device,
                        args.rollout_seeds_per_config,
                        args.rollout_horizon,
                        args.rollout_videos,
                        run_dir
                        / "rollouts"
                        / f"step_{step:06d}"
                        / config_id,
                    )
                macro_success = float(
                    np.mean(
                        [
                            metrics["success_rate"]
                            for metrics in rollout_metrics.values()
                        ]
                    )
                )
                logged_rollouts = {
                    "rollout_ood/macro_success_rate": macro_success,
                    **{
                        f"rollout_ood/{config_id}_success_rate": metrics[
                            "success_rate"
                        ]
                        for config_id, metrics in rollout_metrics.items()
                    },
                }
                wandb.log(logged_rollouts, step=step)
                if macro_success > best_success:
                    best_success = macro_success
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
    parser.add_argument("--design", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--condition",
        choices=("none", "sign_array", "global_token", "pixel_jacobian"),
        required=True,
    )
    parser.add_argument("--expected-demos", type=int, default=200)
    parser.add_argument("--expected-physical-steps", type=int, default=17937)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--physical-batch-size", type=int, default=10)
    parser.add_argument("--configs-per-frame", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--rollout-every", type=int, default=5000)
    parser.add_argument("--rollout-seeds-per-config", type=int, required=True)
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--rollout-videos", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ood-eval-samples", type=int, default=512)
    parser.add_argument("--ood-eval-seed", type=int, default=20260827)
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
