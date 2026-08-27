#!/usr/bin/env python3
"""Learn image-space intent while a fixed DLS layer produces raw actions."""

from __future__ import annotations

import argparse
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

from action_jacobian.simple_servo import (
    eef_pixel_jacobian,
    keypoints,
    make_sample,
    world_to_pixel,
)


class PixelIntentDLS(nn.Module):
    def __init__(self, damping: float = 0.5):
        super().__init__()
        self.damping = damping
        self.heatmaps = nn.Sequential(
            nn.Conv2d(3, 32, 1), nn.GELU(), nn.Conv2d(32, 2, 1)
        )
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij"), -1
        )[..., [1, 0]].reshape(-1, 2).float()
        self.register_buffer("coordinates", coordinates, persistent=False)

    def forward(self, image: torch.Tensor, jacobian: torch.Tensor):
        logits = self.heatmaps(image).flatten(2)
        probabilities = logits.softmax(dim=-1)
        points = probabilities @ self.coordinates
        pixel_error = points[:, 0] - points[:, 1]
        gram = jacobian @ jacobian.transpose(1, 2)
        identity = torch.eye(2, device=jacobian.device, dtype=jacobian.dtype)[None]
        inverse = jacobian.transpose(1, 2) @ torch.linalg.inv(
            gram + self.damping**2 * identity
        )
        action = (inverse @ pixel_error.unsqueeze(-1)).squeeze(-1).clamp(-0.12, 0.12)
        return {"action": action, "pixel_error": pixel_error, "logits": logits}


def build_jacobians(q: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[eef_pixel_jacobian(state, sign) for sign in signs] for state in q],
        dtype=np.float32,
    )


def keypoint_labels(q: np.ndarray, target: np.ndarray):
    target_pixel = np.rint(world_to_pixel(target)).astype(np.int64).clip(0, 63)
    eef_pixel = np.rint(
        np.asarray([world_to_pixel(keypoints(state)[-1]) for state in q])
    ).astype(np.int64).clip(0, 63)
    labels = np.stack(
        [target_pixel[:, 1] * 64 + target_pixel[:, 0], eef_pixel[:, 1] * 64 + eef_pixel[:, 0]],
        axis=1,
    )
    exact_error = world_to_pixel(target) - np.asarray(
        [world_to_pixel(keypoints(state)[-1]) for state in q]
    )
    return labels, exact_error.astype(np.float32)


def rollout(model, q_values, targets, signs, device, output_dir, limit=25):
    output_dir.mkdir(parents=True, exist_ok=True)
    successes, errors = [], []
    model.eval()
    for config_index, sign in enumerate(signs):
        for sample_index in range(min(limit, len(q_values))):
            q = q_values[sample_index].copy()
            target = targets[sample_index]
            frames = []
            for _ in range(26):
                error = float(np.linalg.norm(
                    world_to_pixel(target) - world_to_pixel(keypoints(q)[-1])
                ))
                if error <= 2.5:
                    break
                sample = make_sample(q, target, sign)
                image = torch.from_numpy(sample.image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                jacobian = torch.from_numpy(eef_pixel_jacobian(q, sign)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    action = model(image, jacobian)["action"][0].cpu().numpy()
                q += sign * action
                if sample_index == 0:
                    frames.append(sample.image)
            successes.append(error <= 2.5)
            errors.append(error)
            if sample_index == 0 and frames:
                imageio.mimsave(output_dir / f"config_{config_index}.mp4", frames, fps=8)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_final_error_px": float(np.mean(errors)),
        "cases": len(successes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    with h5py.File(args.dataset, "r") as data:
        q = data["q"][()]
        target = data["target"][()]
        images = data["rgb"][()]
        actions = data["actions"][()]
        train_signs = data["train_signs"][()]
        ood_signs = data["ood_signs"][()]
    signs = np.concatenate([train_signs, ood_signs])
    jacobians = build_jacobians(q, signs)
    labels, exact_error = keypoint_labels(q, target)
    train = np.arange(1600)
    val = np.arange(1600, 1800)
    test = np.arange(1800, 2000)
    model = PixelIntentDLS().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = vars(args) | {"git_commit": commit, "method": "learned_pixel_intent_fixed_dls"}
    run = wandb.init(
        entity="wuji-tech",
        project="pixel-action-jacobian-simple-servo",
        name="pixel_intent_fixed_dls_s0",
        config=config,
    )
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        physical = rng.choice(train, args.batch_size // 4, replace=True)
        physical = np.repeat(physical, 4)
        configs = np.tile(np.arange(4), args.batch_size // 4)
        image = torch.from_numpy(images[physical]).permute(0, 3, 1, 2).float().div(255).to(device)
        jacobian = torch.from_numpy(jacobians[physical, configs]).to(device)
        result = model(image, jacobian)
        label = torch.from_numpy(labels[physical]).to(device)
        desired_error = torch.from_numpy(exact_error[physical]).to(device)
        target_action = torch.from_numpy(actions[physical, configs]).to(device)
        heatmap_loss = F.cross_entropy(result["logits"].reshape(-1, 4096), label.reshape(-1))
        intent_loss = F.mse_loss(result["pixel_error"], desired_error)
        action_loss = F.mse_loss(result["action"], target_action)
        loss = heatmap_loss + 0.05 * intent_loss + action_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log(
            {
                "train/loss": float(loss),
                "train/heatmap_ce": float(heatmap_loss),
                "train/intent_mse": float(intent_loss),
                "train/action_mse": float(action_loss),
                "samples_seen": step * args.batch_size,
                "throughput_samples_s": step * args.batch_size / (time.perf_counter() - start),
            },
            step=step,
        )
        if step % 250 == 0 or step == args.steps:
            with torch.inference_mode():
                physical_eval = np.repeat(val, 4)
                config_eval = np.tile(np.arange(4, 8), len(val))
                image_eval = torch.from_numpy(images[physical_eval]).permute(0, 3, 1, 2).float().div(255).to(device)
                prediction = model(
                    image_eval,
                    torch.from_numpy(jacobians[physical_eval, config_eval]).to(device),
                )
                action_mae = float(
                    (prediction["action"].cpu() - torch.from_numpy(actions[physical_eval, config_eval])).abs().mean()
                )
                intent_rmse = float(torch.sqrt(F.mse_loss(
                    prediction["pixel_error"].cpu(), torch.from_numpy(exact_error[physical_eval])
                )))
            wandb.log(
                {"val_ood/action_mae": action_mae, "val_ood/intent_rmse_px": intent_rmse},
                step=step,
            )
        if step % 500 == 0 or step == args.steps:
            metrics = rollout(
                model, q[test], target[test], ood_signs, device, output / f"step_{step}"
            )
            wandb.log(
                {
                    "rollout_ood/success_rate": metrics["success_rate"],
                    "rollout_ood/mean_final_error_px": metrics["mean_final_error_px"],
                },
                step=step,
            )
            torch.save(
                {"model": model.state_dict(), "config": config, "metrics": metrics},
                output / f"step_{step}.pth",
            )
    wandb.finish()


if __name__ == "__main__":
    main()
