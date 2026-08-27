#!/usr/bin/env python3
"""Learn Panda image-space intent and execute it through a fixed DLS inverse."""

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
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from action_jacobian.models.dinov3 import FrozenDinoV3Backbone
from action_jacobian.representation import _continuous_project
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from train import build_env, render_agentview


class PandaPixelIntent(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.backbone = FrozenDinoV3Backbone(model_path, hidden_dim=256, num_cameras=1)
        self.heatmap_head = nn.Linear(256, 2)
        self.offset_head = nn.Linear(256, 4)
        centers = torch.stack(
            torch.meshgrid(
                torch.arange(16, dtype=torch.float32) * 16 + 8,
                torch.arange(16, dtype=torch.float32) * 16 + 8,
                indexing="ij",
            ),
            -1,
        )[..., [1, 0]].reshape(256, 2)
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, image: torch.Tensor):
        tokens, _ = self.backbone(image.unsqueeze(1))
        logits = self.heatmap_head(tokens).transpose(1, 2)
        offsets = self.offset_head(tokens).reshape(-1, 256, 2, 2).permute(0, 2, 1, 3)
        offsets = offsets.tanh() * 8.0
        candidates = self.centers[None, None] + offsets
        points = (logits.softmax(dim=-1).unsqueeze(-1) * candidates).sum(dim=2)
        return {
            "points": points,
            "pixel_error": points[:, 0] - points[:, 1],
            "logits": logits,
        }


def project(world: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray):
    return _continuous_project(np.asarray(world), world_to_camera, intrinsic).astype(np.float32)


def draw_feature_markers(
    image: np.ndarray,
    cube_pixel: np.ndarray,
    eef_pixel: np.ndarray,
) -> np.ndarray:
    """Render observable visual-servo features without changing scene geometry."""
    marked = image.copy()
    yy, xx = np.ogrid[: marked.shape[0], : marked.shape[1]]
    cube_radius_sq = (xx - cube_pixel[0]) ** 2 + (yy - cube_pixel[1]) ** 2
    cube_ring = (cube_radius_sq >= 8**2) & (cube_radius_sq <= 12**2)
    marked[cube_ring] = (0, 255, 0)
    eef_disk = (xx - eef_pixel[0]) ** 2 + (yy - eef_pixel[1]) ** 2 <= 4**2
    marked[eef_disk] = (255, 0, 255)
    return marked


def overlay_dataset_features(
    images: np.ndarray, cube_pixels: np.ndarray, eef_pixels: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            draw_feature_markers(image, cube_pixel, eef_pixel)
            for image, cube_pixel, eef_pixel in zip(images, cube_pixels, eef_pixels)
        ]
    )


def load_data(cache_path: str, raw_path: str):
    images, cube_pixels, eef_pixels, demo_indexes = [], [], [], []
    with h5py.File(cache_path, "r") as cache, h5py.File(raw_path, "r") as raw:
        intrinsic = np.asarray(json.loads(cache.attrs["camera_intrinsic_16x16"]), dtype=np.float64)
        intrinsic[:2] *= 16
        world_to_camera = np.linalg.inv(
            np.asarray(json.loads(cache.attrs["camera_to_world"]), dtype=np.float64)
        )
        names = sorted(cache["demos"])
        for demo_index, name in enumerate(names):
            rgb = cache[f"demos/{name}/rgb"][()]
            source = raw[f"data/{name}/obs"]
            eef = source["robot0_eef_pos"][()]
            cube = source["object"][:, :3]
            if not (len(rgb) == len(eef) == len(cube)):
                raise ValueError(f"Unaligned demo {name}")
            images.append(rgb)
            eef_pixels.append(project(eef, world_to_camera, intrinsic))
            cube_pixels.append(project(cube, world_to_camera, intrinsic))
            demo_indexes.append(np.full(len(rgb), demo_index, dtype=np.int64))
    return (
        np.concatenate(images),
        np.concatenate(cube_pixels),
        np.concatenate(eef_pixels),
        np.concatenate(demo_indexes),
    )


def dls_action(jacobian: np.ndarray, pixel_error: np.ndarray):
    return np.clip(
        jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + 25.0 * np.eye(2)) @ pixel_error,
        -0.02,
        0.02,
    )


def panda_eef_jacobian(env, site_id, qvel_indexes, world_to_camera, intrinsic):
    sim = env.sim
    jacobian_world = np.zeros((3, sim.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        sim.model._model, sim.data._data, jacobian_world, None, site_id
    )
    rotation = world_to_camera[:3, :3]
    point_camera = rotation @ sim.data.site_xpos[site_id] + world_to_camera[:3, 3]
    velocity_camera = rotation @ jacobian_world[:, qvel_indexes]
    x, y, z = point_camera
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    return np.stack(
        [
            fx / z * velocity_camera[0] - fx * x / (z * z) * velocity_camera[2],
            fy / z * velocity_camera[1] - fy * y / (z * z) * velocity_camera[2],
        ]
    )


def rollout(
    model,
    raw_path,
    signs_by_config,
    device,
    seeds,
    horizon,
    output_dir,
    feature_overlay=False,
):
    env = build_env(raw_path)
    intrinsic = get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
    world_to_camera = np.linalg.inv(get_camera_extrinsic_matrix(env.sim, "agentview"))
    site_id = env.sim.model.site_name2id("gripper0_right_grip_site")
    qvel_indexes = np.asarray(env.robots[0]._ref_joint_vel_indexes)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    model.eval()
    try:
        for config_id, signs in signs_by_config.items():
            signs = np.asarray(signs, dtype=np.float64)
            for seed in range(seeds):
                np.random.seed(seed)
                random.seed(seed)
                env.reset()
                env.set_init_action()
                frames = []
                trace = []
                for step in range(horizon):
                    image = render_agentview(env)
                    cube_pixel = project(
                        env.sim.data.body_xpos[env.cube_body_id][None],
                        world_to_camera,
                        intrinsic,
                    )[0]
                    eef_pixel = project(
                        env.sim.data.site_xpos[site_id][None], world_to_camera, intrinsic
                    )[0]
                    if feature_overlay:
                        image = draw_feature_markers(image, cube_pixel, eef_pixel)
                    if seed == 0:
                        frames.append(image)
                    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        predicted_error = model(tensor)["pixel_error"][0].float().cpu().numpy()
                    true_error = cube_pixel - eef_pixel
                    true_norm = float(np.linalg.norm(true_error))
                    trace.append(
                        {
                            "step": step,
                            "true_error": true_error.tolist(),
                            "predicted_error": predicted_error.tolist(),
                            "true_norm": true_norm,
                            "prediction_error_norm": float(
                                np.linalg.norm(predicted_error - true_error)
                            ),
                        }
                    )
                    if true_norm <= 5.0:
                        break
                    canonical_jacobian = panda_eef_jacobian(
                        env, site_id, qvel_indexes, world_to_camera, intrinsic
                    )
                    raw_jacobian = canonical_jacobian * signs[None]
                    raw_action = dls_action(raw_jacobian, predicted_error)
                    env.step(np.concatenate([raw_action * signs, [-1.0]]))
                success = true_norm <= 5.0
                results.append((config_id, seed, success, true_norm, step))
                with open(output_dir / f"{config_id}_seed_{seed:02d}.json", "w") as handle:
                    json.dump(
                        {
                            "config_id": config_id,
                            "seed": seed,
                            "success": bool(success),
                            "final_error_px": true_norm,
                            "trace": trace,
                        },
                        handle,
                    )
                if seed == 0:
                    imageio.mimsave(
                        output_dir / f"{config_id}_success_{int(success)}.mp4",
                        frames,
                        fps=20,
                    )
    finally:
        env.close()
    return {
        "success_rate": float(np.mean([item[2] for item in results])),
        "mean_final_error_px": float(np.mean([item[3] for item in results])),
        "cases": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--raw-dataset", required=True)
    parser.add_argument("--design", default="configs/joint_sign_dr_v1.json")
    parser.add_argument("--dinov3-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-seeds", type=int, default=2)
    parser.add_argument("--feature-overlay", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    images, cube_pixels, eef_pixels, demo_indexes = load_data(args.cache, args.raw_dataset)
    if args.feature_overlay:
        images = overlay_dataset_features(images, cube_pixels, eef_pixels)
    train_indexes = np.flatnonzero(demo_indexes < 160)
    val_indexes = np.flatnonzero(demo_indexes >= 160)
    labels = np.stack(
        [
            np.rint(cube_pixels[:, 1] / 16 - 0.5).clip(0, 15) * 16
            + np.rint(cube_pixels[:, 0] / 16 - 0.5).clip(0, 15),
            np.rint(eef_pixels[:, 1] / 16 - 0.5).clip(0, 15) * 16
            + np.rint(eef_pixels[:, 0] / 16 - 0.5).clip(0, 15),
        ],
        axis=1,
    ).astype(np.int64)
    desired_error = cube_pixels - eef_pixels
    with open(args.design) as handle:
        ood_signs = json.load(handle)["ood"]
    model = PandaPixelIntent(args.dinov3_model).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = vars(args) | {
        "git_commit": commit,
        "train_frames": len(train_indexes),
        "val_frames": len(val_indexes),
        "method": "panda_pixel_intent_fixed_dls",
    }
    run = wandb.init(
        entity="wuji-tech",
        project="pixel-action-jacobian-panda-servo",
        name="panda_pixel_intent_dls_s0",
        config=config,
    )
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indexes = rng.choice(train_indexes, args.batch_size, replace=True)
        image = torch.from_numpy(images[indexes]).permute(0, 3, 1, 2).float().div(255).to(device)
        result = model(image)
        label = torch.from_numpy(labels[indexes]).to(device)
        target_points = torch.from_numpy(
            np.stack([cube_pixels[indexes], eef_pixels[indexes]], axis=1)
        ).to(device)
        target_error = torch.from_numpy(desired_error[indexes]).to(device)
        heatmap_loss = F.cross_entropy(result["logits"].reshape(-1, 256), label.reshape(-1))
        point_loss = F.smooth_l1_loss(result["points"], target_points)
        intent_loss = F.smooth_l1_loss(result["pixel_error"], target_error)
        loss = heatmap_loss + 0.1 * point_loss + 0.05 * intent_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log(
            {
                "train/loss": float(loss),
                "train/heatmap_ce": float(heatmap_loss),
                "train/point_l1_px": float((result["points"] - target_points).abs().mean()),
                "train/intent_l1_px": float((result["pixel_error"] - target_error).abs().mean()),
                "samples_seen": step * args.batch_size,
                "throughput_samples_s": step * args.batch_size / (time.perf_counter() - started),
            },
            step=step,
        )
        if step % 250 == 0 or step == args.steps:
            sample = rng.choice(val_indexes, min(512, len(val_indexes)), replace=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    torch.from_numpy(images[sample]).permute(0, 3, 1, 2).float().div(255).to(device)
                )["pixel_error"].float().cpu()
            error = prediction - torch.from_numpy(desired_error[sample])
            wandb.log(
                {
                    "val/intent_mae_px": float(error.abs().mean()),
                    "val/intent_rmse_px": float(torch.sqrt((error**2).mean())),
                },
                step=step,
            )
        if step % 1000 == 0 or step == args.steps:
            metrics = rollout(
                model,
                args.raw_dataset,
                ood_signs,
                device,
                args.rollout_seeds,
                80,
                output / f"step_{step}",
                args.feature_overlay,
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
