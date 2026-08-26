#!/usr/bin/env python3
"""Train and evaluate the approved canonical official-Lift ACT baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

import robosuite as suite
from robosuite.wrappers.action_wrapper import wrap_env_action_space

from models.act import ACTPolicy


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


def git_commit(repo: str) -> str:
    return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()


def autocast_context(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


class LiftFrameDataset(Dataset):
    """Frame-indexed ACT samples backed by the deterministic cloud RGB cache."""

    def __init__(
        self,
        dataset_path: str,
        rgb_cache_path: str,
        chunk_size: int,
        split: str,
        expected_demos: int = 200,
    ):
        self.dataset_path = dataset_path
        self.rgb_cache_path = rgb_cache_path
        self.chunk_size = chunk_size
        self.split = split
        self._cache_handle = None
        self.actions = {}
        self.indices = []
        all_actions = []
        with h5py.File(dataset_path, "r") as source, h5py.File(rgb_cache_path, "r") as cache:
            if not bool(cache["data"].attrs.get("complete", False)):
                raise RuntimeError("RGB cache is incomplete")
            source_hash = sha256_file(dataset_path)
            cached_hash = cache["data"].attrs.get("source_sha256", "")
            if cached_hash != source_hash:
                raise RuntimeError(f"RGB cache source hash mismatch: {cached_hash} != {source_hash}")
            demo_names = sorted(
                (name for name in source["data"] if name.startswith("demo_")),
                key=lambda name: int(name.split("_")[-1]),
            )
            if len(demo_names) != expected_demos:
                raise RuntimeError(f"Expected exactly {expected_demos} demos, found {len(demo_names)}")
            for demo_name in demo_names:
                actions = source["data"][demo_name]["actions"][()].astype(np.float32)
                rgb = cache["data"][demo_name]["agentview_rgb"]
                if len(rgb) != len(actions):
                    raise RuntimeError(f"Length mismatch for {demo_name}: rgb={len(rgb)}, actions={len(actions)}")
                self.actions[demo_name] = actions
                all_actions.append(actions)
                for timestep in range(len(actions)):
                    is_val = timestep % 10 == 0
                    if (split == "val" and is_val) or (split == "train" and not is_val):
                        self.indices.append((demo_name, timestep))
        action_array = np.concatenate(all_actions, axis=0)
        self.action_mean = action_array.mean(axis=0).astype(np.float32)
        self.action_std = np.maximum(action_array.std(axis=0), 1e-4).astype(np.float32)
        self.action_min = action_array.min(axis=0).astype(np.float32)
        self.action_max = action_array.max(axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _cache(self):
        if self._cache_handle is None:
            self._cache_handle = h5py.File(self.rgb_cache_path, "r", swmr=True)
        return self._cache_handle

    def __getitem__(self, index: int):
        demo_name, timestep = self.indices[index]
        image = self._cache()["data"][demo_name]["agentview_rgb"][timestep]
        image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
        actions = self.actions[demo_name]
        chunk = np.zeros((self.chunk_size, actions.shape[1]), dtype=np.float32)
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        available = min(self.chunk_size, len(actions) - timestep)
        chunk[:available] = actions[timestep : timestep + available]
        is_pad[:available] = False
        chunk = (chunk - self.action_mean) / self.action_std
        return {
            "image": image.unsqueeze(0),
            "qpos": torch.zeros(7, dtype=torch.float32),
            "actions": torch.from_numpy(chunk),
            "is_pad": torch.from_numpy(is_pad),
            "cam_extrinsics": torch.zeros(2, 4, 4, dtype=torch.float32),
        }


def build_env(dataset_path: str):
    with h5py.File(dataset_path, "r") as source:
        env_args = json.loads(source["data"].attrs["env_args"])
        action_space = source["data"].attrs["action_space"]
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview"],
    )
    env = suite.make(env_name=env_args["env_name"], **env_kwargs)
    if action_space in ("eef_delta", "joint_delta"):
        env = wrap_env_action_space(env, action_space)
    return env


def render_agentview(env) -> np.ndarray:
    env.sim.forward()
    image = env.sim.render(camera_name="agentview", height=256, width=256, depth=False)
    return np.flipud(image).copy()


def policy_action_chunk(policy, image: np.ndarray, stats: dict, use_bf16: bool) -> np.ndarray:
    image_tensor = (
        torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).unsqueeze(0).cuda()
    )
    model_input = {
        "image": image_tensor,
        "qpos": torch.zeros(1, 7, device="cuda"),
        "cam_extrinsics": torch.zeros(1, 2, 4, 4, device="cuda"),
    }
    with torch.inference_mode(), autocast_context(use_bf16):
        normalized = policy(model_input)[0].float().cpu().numpy()
    actions = normalized * stats["action_std"] + stats["action_mean"]
    return np.clip(actions, stats["action_min"], stats["action_max"])


def evaluate(policy, dataset_path: str, stats: dict, output_dir: Path, seeds: int, horizon: int, videos: int, use_bf16: bool):
    env = build_env(dataset_path)
    successes = []
    episode_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
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
                action_chunk = policy_action_chunk(policy, image, stats, use_bf16)
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
            episode_rows.append({"seed": seed, "success": succeeded, "success_step": success_step})
            if seed < videos:
                name = f"seed_{seed:03d}_success_{int(succeeded)}.mp4"
                imageio.mimsave(output_dir / name, frames, fps=20, codec="libx264", quality=8)
            print(f"eval seed={seed:02d} success={succeeded} step={success_step}")
    finally:
        env.close()
    summary = {
        "num_seeds": seeds,
        "horizon": horizon,
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "episodes": episode_rows,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def mean_metrics(rows):
    return {key: torch.stack([row[key].detach() for row in rows]).mean().item() for key in rows[0]}


def save_checkpoint(path: Path, policy, optimizer, step: int, samples_seen: int, config: dict, stats: dict, commit: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "samples_seen": samples_seen,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stats": stats,
            "git_commit": commit,
            "wandb_run_id": wandb.run.id,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    repo = str(Path(__file__).resolve().parents[1])
    commit = git_commit(repo)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_set = LiftFrameDataset(args.dataset, args.rgb_cache, args.chunk_size, "train", args.expected_demos)
    val_set = LiftFrameDataset(args.dataset, args.rgb_cache, args.chunk_size, "val", args.expected_demos)
    stats = {
        "action_mean": train_set.action_mean,
        "action_std": train_set.action_std,
        "action_min": train_set.action_min,
        "action_max": train_set.action_max,
    }
    config = vars(args).copy()
    config.update(
        git_commit=commit,
        dataset_sha256=sha256_file(args.dataset),
        rgb_cache_sha256=sha256_file(args.rgb_cache),
        dinov3_weights_sha256=sha256_file(os.path.join(args.dinov3_model_path, "model.safetensors")),
        train_frames=len(train_set),
        val_frames=len(val_set),
        action_dim=8,
        obs_dim=7,
        configuration="canonical_cfg0",
        structural_condition="none",
        proprio_input="zeroed",
        validation_split="every 10th frame within all 200 trajectories; optimization sanity only",
    )
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    with open(run_dir / "dataset_stats.json", "w") as handle:
        json.dump({key: value.tolist() for key, value in stats.items()}, handle, indent=2)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_iterator = iter(train_loader)

    args.action_dim = 8
    args.obs_dim = 7
    args.use_plucker = False
    args.use_cam_pose = False
    args.prob_drop_proprio = 1.0
    args.num_side_cam = 1
    policy = ACTPolicy(args).cuda()
    optimizer = policy.configure_optimizers()

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.run_name,
        config=config,
        mode=args.wandb_mode,
    )
    wandb.run.summary["git_commit"] = commit
    wandb.run.summary["dataset_sha256"] = config["dataset_sha256"]
    wandb.run.summary["rgb_cache_sha256"] = config["rgb_cache_sha256"]

    samples_seen = 0
    last_log_time = time.time()
    last_log_samples = 0
    best_success = -1.0
    use_bf16 = bool(args.use_bf16)
    try:
        for step in range(1, args.max_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            policy.train()
            with autocast_context(use_bf16):
                losses = policy(batch)
                loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += len(batch["image"])

            if step % args.log_every == 0:
                now = time.time()
                throughput = (samples_seen - last_log_samples) / max(now - last_log_time, 1e-6)
                metrics = {f"train/{key}": value.detach().item() for key, value in losses.items()}
                metrics.update(samples_seen=samples_seen, throughput_samples_s=throughput, lr=optimizer.param_groups[0]["lr"])
                wandb.log(metrics, step=step)
                print(f"step={step} samples={samples_seen} loss={loss.item():.6f} throughput={throughput:.1f}/s")
                last_log_time, last_log_samples = now, samples_seen

            if step % args.val_every == 0:
                policy.eval()
                rows = []
                with torch.inference_mode():
                    for batch_index, batch in enumerate(val_loader):
                        if batch_index >= args.val_batches:
                            break
                        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                        with autocast_context(use_bf16):
                            rows.append(policy(batch))
                values = mean_metrics(rows)
                wandb.log({f"val/{key}": value for key, value in values.items()}, step=step)

            if step % args.save_every == 0 or step == args.max_steps:
                checkpoint = run_dir / "checkpoints" / f"step_{step:06d}.pth"
                save_checkpoint(checkpoint, policy, optimizer, step, samples_seen, config, stats, commit)
                wandb.run.summary["latest_checkpoint"] = str(checkpoint)

            if step % args.eval_every == 0 or (step == args.max_steps and args.eval_final):
                eval_seeds = args.eval_seeds if step % args.full_eval_every == 0 or step == args.max_steps else args.quick_eval_seeds
                summary = evaluate(
                    policy,
                    args.dataset,
                    stats,
                    run_dir / "eval" / f"step_{step:06d}",
                    eval_seeds,
                    args.eval_horizon,
                    args.eval_videos,
                    use_bf16,
                )
                wandb.log({f"rollout/success_rate_{eval_seeds}": summary["success_rate"]}, step=step)
                if eval_seeds == args.eval_seeds and summary["success_rate"] > best_success:
                    best_success = summary["success_rate"]
                    wandb.run.summary["best_success_rate"] = best_success
                    wandb.run.summary["best_step"] = step
    finally:
        wandb.run.summary["samples_seen"] = samples_seen
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--dinov3-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-entity", default="wuji-tech")
    parser.add_argument("--wandb-project", default="official-lift-baseline")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-demos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--full-eval-every", type=int, default=5_000)
    parser.add_argument("--quick-eval-seeds", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, default=50)
    parser.add_argument("--eval-horizon", type=int, default=400)
    parser.add_argument("--eval-videos", type=int, default=3)
    parser.add_argument("--eval-final", type=int, default=1)
    parser.add_argument("--use-bf16", type=int, default=1)
    parser.add_argument("--backbone", default="dinov3_vitb16")
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--pre-norm", type=int, default=1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-drop-prob", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
