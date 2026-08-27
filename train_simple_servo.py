#!/usr/bin/env python3
"""Train one matched baseline on the analytic visual-servo benchmark."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from action_jacobian.simple_servo import keypoints, make_sample, world_to_pixel


class ServoPolicy(nn.Module):
    def __init__(self, condition: str):
        super().__init__()
        self.condition = condition
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        if condition == "pixel_jacobian":
            self.pixel = nn.Sequential(nn.Conv2d(7, 32, 1), nn.GELU())
            feature_channels = 96
        elif condition in ("sign_array", "global_token"):
            condition_dim = {"sign_array": 3, "global_token": 12}[condition]
            self.condition_film = nn.Sequential(
                nn.Linear(condition_dim, 128), nn.GELU(), nn.Linear(128, 128)
            )
            feature_channels = 64
        else:
            feature_channels = 64
        self.visual = nn.Sequential(
            nn.Conv2d(feature_channels, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3), nn.Tanh(),
        )

    def forward(self, image, structural=None):
        feature = self.rgb(image)
        if self.condition == "pixel_jacobian":
            feature = torch.cat([feature, self.pixel(structural)], dim=1)
        elif self.condition in ("sign_array", "global_token"):
            gamma, beta = self.condition_film(structural).chunk(2, dim=1)
            feature = feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        feature = self.visual(feature).flatten(1)
        return 0.12 * self.head(feature)


def structural_input(condition, fields, descriptors, signs, physical, configs, device):
    if condition == "none":
        return None
    if condition == "pixel_jacobian":
        return torch.from_numpy(fields[physical, configs].astype(np.float32)).to(device)
    if condition == "global_token":
        return torch.from_numpy(descriptors[physical, configs].astype(np.float32)).to(device)
    return torch.from_numpy(signs[configs].astype(np.float32)).to(device)


def offline_metrics(model, condition, arrays, physical_indexes, config_indexes, device):
    images, fields, descriptors, actions, signs = arrays
    errors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(physical_indexes), 256):
            physical = physical_indexes[start:start + 256]
            configs = config_indexes[start:start + 256]
            image = torch.from_numpy(images[physical]).permute(0, 3, 1, 2).float().div(255).to(device)
            structural = structural_input(condition, fields, descriptors, signs, physical, configs, device)
            prediction = model(image, structural).cpu().numpy()
            errors.append(np.abs(prediction - actions[physical, configs]))
    error = np.concatenate(errors)
    return {"mae": float(error.mean())}


def rollout(model, condition, q_values, targets, signs, device, output_dir, limit=25):
    successes, final_errors = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for config_index, sign in enumerate(signs):
        for sample_index in range(min(limit, len(q_values))):
            q = q_values[sample_index].copy()
            target = targets[sample_index]
            frames = []
            for step in range(26):
                error = float(np.linalg.norm(world_to_pixel(target) - world_to_pixel(keypoints(q)[-1])))
                if error <= 2.5:
                    break
                sample = make_sample(q, target, sign)
                field = sample.pixel_jacobian.reshape(7, 16, 4, 16, 4).mean(axis=(2, 4))
                image = torch.from_numpy(sample.image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                if condition == "pixel_jacobian":
                    structural = torch.from_numpy(field).float().unsqueeze(0).to(device)
                elif condition == "global_token":
                    structural = torch.from_numpy(sample.global_jacobian).float().unsqueeze(0).to(device)
                elif condition == "sign_array":
                    structural = torch.from_numpy(sign).float().unsqueeze(0).to(device)
                else:
                    structural = None
                with torch.inference_mode():
                    action = model(image, structural)[0].cpu().numpy()
                q += sign * action
                if sample_index == 0:
                    frames.append(sample.image)
            successes.append(error <= 2.5)
            final_errors.append(error)
            if sample_index == 0 and frames:
                imageio.mimsave(output_dir / f"config_{config_index}.mp4", frames, fps=8)
    return {"success_rate": float(np.mean(successes)), "mean_final_error_px": float(np.mean(final_errors)), "cases": len(successes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True, choices=("none", "sign_array", "global_token", "pixel_jacobian"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-entity", default="wuji-tech")
    parser.add_argument("--wandb-project", default="pixel-action-jacobian-simple-servo")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    with h5py.File(args.dataset, "r") as data:
        q = data["q"][()]; target = data["target"][()]; images = data["rgb"][()]
        fields = data["pixel_jacobian"][()]; descriptors = data["global_jacobian"][()]
        actions = data["actions"][()]; train_signs = data["train_signs"][()]; ood_signs = data["ood_signs"][()]
    signs = np.concatenate([train_signs, ood_signs])
    if len(q) != 2000:
        raise ValueError(f"Expected exactly 2000 physical samples, found {len(q)}")
    if args.batch_size % len(train_signs) != 0:
        raise ValueError("batch-size must be divisible by the four train configurations")
    train_physical = np.arange(0, 1600); id_val_physical = np.arange(1600, 1800); test_physical = np.arange(1800, 2000)
    train_configs = np.arange(4); ood_configs = np.arange(4, 8)
    arrays = (images, fields, descriptors, actions, signs)
    model = ServoPolicy(args.condition).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = vars(args) | {"git_commit": commit, "train_physical": 1600, "id_val_physical": 200, "test_physical": 200, "train_configs": 4, "ood_configs": 4}
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=f"{args.condition}_s{args.seed}", config=config)
    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        selected_physical = rng.choice(
            train_physical, args.batch_size // len(train_signs), replace=True
        )
        physical = np.repeat(selected_physical, len(train_signs))
        configs = np.tile(train_configs, len(selected_physical))
        image = torch.from_numpy(images[physical]).permute(0, 3, 1, 2).float().div(255).to(device)
        structural = structural_input(args.condition, fields, descriptors, signs, physical, configs, device)
        target_action = torch.from_numpy(actions[physical, configs]).float().to(device)
        model.train(); prediction = model(image, structural); loss = F.mse_loss(prediction, target_action)
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        wandb.log({"train/mse": float(loss), "train/grad_norm": float(grad), "samples_seen": step * args.batch_size, "throughput_samples_s": step * args.batch_size / (time.perf_counter() - start_time)}, step=step)
        if step % 250 == 0 or step == args.steps:
            id_p = np.repeat(id_val_physical, 4); id_c = np.tile(train_configs, len(id_val_physical))
            ood_p = np.repeat(id_val_physical, 4); ood_c = np.tile(ood_configs, len(id_val_physical))
            id_metrics = offline_metrics(model, args.condition, arrays, id_p, id_c, device)
            ood_metrics = offline_metrics(model, args.condition, arrays, ood_p, ood_c, device)
            wandb.log({"val_id/action_mae": id_metrics["mae"], "val_ood/action_mae": ood_metrics["mae"]}, step=step)
        if step % 1000 == 0 or step == args.steps:
            result = rollout(model, args.condition, q[test_physical], target[test_physical], ood_signs, device, output / "rollouts" / f"step_{step}")
            wandb.log({"rollout_ood/success_rate": result["success_rate"], "rollout_ood/mean_final_error_px": result["mean_final_error_px"]}, step=step)
            torch.save({"model": model.state_dict(), "config": config, "step": step, "rollout": result}, output / f"step_{step}.pth")
    run.summary["git_commit"] = commit
    wandb.finish()


if __name__ == "__main__":
    main()
