"""Validate and visualize paired equivalent-URDF trajectories."""

from __future__ import annotations

import argparse
import json
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

from equivalent_panda_configs import CONFIG_SPECS  # noqa: F401: registers robot classes
from play_dataset import create_replay_env_from_dataset
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)


CAMERA_NAME = "agentview"
FRAME_SIZE = 256
BODY_NAMES = [f"link{i}" for i in range(8)]
EEF_SITE_NAME = "gripper0_right_grip_site"
KEYPOINT_COLORS = [
    (255, 64, 64),
    (255, 160, 64),
    (255, 230, 64),
    (120, 255, 64),
    (64, 255, 180),
    (64, 190, 255),
    (100, 100, 255),
    (210, 90, 255),
    (255, 255, 255),
]


def _keypoints_world(env) -> np.ndarray:
    prefix = env.robots[0].robot_model.naming_prefix
    points = [env.sim.data.get_body_xpos(f"{prefix}{name}").copy() for name in BODY_NAMES]
    site_id = env.sim.model.site_name2id(EEF_SITE_NAME)
    points.append(env.sim.data.site_xpos[site_id].copy())
    return np.asarray(points)


def _render_state(env, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    frame = env.sim.render(camera_name=CAMERA_NAME, height=FRAME_SIZE, width=FRAME_SIZE)
    frame = np.flipud(frame).astype(np.uint8)
    points = _keypoints_world(env)
    transform = get_camera_transform_matrix(env.sim, CAMERA_NAME, FRAME_SIZE, FRAME_SIZE)
    pixels = project_points_from_world_to_camera(points, transform, FRAME_SIZE, FRAME_SIZE)
    return frame, points, pixels


def _draw_action_panel(frame: np.ndarray, config_id: str, action: np.ndarray, pixels: np.ndarray, scale: float) -> np.ndarray:
    # 352 px total height is codec-friendly (divisible by 16).
    canvas = np.zeros((FRAME_SIZE + 96, FRAME_SIZE, 3), dtype=np.uint8)
    canvas[:FRAME_SIZE] = frame
    for index, (row, col) in enumerate(pixels):
        cv2.circle(canvas, (int(col), int(row)), 4, KEYPOINT_COLORS[index], -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(col), int(row)), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (FRAME_SIZE - 1, 25), (0, 0, 0), -1)
    sign_text = "".join("+" if value > 0 else "-" for value in CONFIG_SPECS[config_id]["joint_signs"])
    cv2.putText(canvas, f"{config_id}  S={sign_text}", (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    origin_y = FRAME_SIZE + 43
    cv2.putText(canvas, "joint delta action", (7, FRAME_SIZE + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(canvas, (7, origin_y), (FRAME_SIZE - 7, origin_y), (90, 90, 90), 1)
    bar_width = 24
    for joint_index, value in enumerate(action[:7]):
        x0 = 12 + joint_index * 34
        height = int(np.clip(value / scale, -1.0, 1.0) * 31)
        color = (80, 210, 255) if value >= 0 else (255, 120, 90)
        cv2.rectangle(canvas, (x0, origin_y - max(height, 0)), (x0 + bar_width, origin_y - min(height, 0)), color, -1)
        cv2.putText(canvas, str(joint_index + 1), (x0 + 7, FRAME_SIZE + 81), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def _tile_config_panels(panels: list[np.ndarray]) -> np.ndarray:
    columns = min(4, len(panels))
    rows = []
    for start in range(0, len(panels), columns):
        row = list(panels[start : start + columns])
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _load_dataset(dataset_path: str):
    with h5py.File(dataset_path, "r") as dataset:
        demos = {}
        for demo_name in sorted(dataset["data"].keys()):
            demos[demo_name] = {
                "states": dataset["data"][demo_name]["states"][()],
                "actions": dataset["data"][demo_name]["actions"][()],
                "success": bool(np.max(dataset["data"][demo_name]["rewards"][()]) > 0),
            }
    return demos


def visualize_unit_test(input_dir: str, output_dir: str) -> None:
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(input_dir, "datasets")

    datasets = {
        config_id: _load_dataset(os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5"))
        for config_id in CONFIG_SPECS
    }

    metrics = {
        "num_configs": len(CONFIG_SPECS),
        "num_trajectories_per_config": len(datasets["cfg0"]),
        "body_keypoints": BODY_NAMES + ["eef"],
        "camera": CAMERA_NAME,
        "trajectories": {},
        "max_body_keypoint_error_m": 0.0,
        "max_rgb_abs_error": 0,
        "mean_rgb_abs_error": 0.0,
        "fraction_rgb_values_abs_error_gt_5": 0.0,
        "max_canonicalized_action_error": 0.0,
        "all_successful": True,
    }
    rgb_error_sum = 0.0
    rgb_error_count = 0
    rgb_error_gt_5_count = 0

    for demo_name in datasets["cfg0"]:
        lengths = {config_id: len(data[demo_name]["states"]) for config_id, data in datasets.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Unpaired trajectory lengths for {demo_name}: {lengths}")
        trajectory_length = next(iter(lengths.values()))
        action_scale = max(
            float(np.max(np.abs(datasets[config_id][demo_name]["actions"][:, :7])))
            for config_id in CONFIG_SPECS
        )
        action_scale = max(action_scale, 1e-6)

        rendered = {config_id: [] for config_id in CONFIG_SPECS}
        keypoints = {config_id: [] for config_id in CONFIG_SPECS}
        pixels = {config_id: [] for config_id in CONFIG_SPECS}
        for config_id in CONFIG_SPECS:
            # robosuite's OSMesa renderer does not make its GL context current
            # on every frame. Keep only one live context so cross-model pixel
            # comparisons cannot be contaminated by a later-created context.
            dataset_path = os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5")
            env = create_replay_env_from_dataset(dataset_path)[0]
            env.reset()
            env.sim.render(camera_name=CAMERA_NAME, height=FRAME_SIZE, width=FRAME_SIZE)
            for state in datasets[config_id][demo_name]["states"]:
                frame, points, projected = _render_state(env, state)
                rendered[config_id].append(frame)
                keypoints[config_id].append(points)
                pixels[config_id].append(projected)
            env.close()
            rendered[config_id] = np.asarray(rendered[config_id])
            keypoints[config_id] = np.asarray(keypoints[config_id])
            pixels[config_id] = np.asarray(pixels[config_id])

        canonical_frames = rendered["cfg0"]
        canonical_points = keypoints["cfg0"]
        canonical_actions = datasets["cfg0"][demo_name]["actions"]
        max_keypoint_error = 0.0
        max_rgb_error = 0
        max_action_error = 0.0
        trajectory_rgb_error_sum = 0.0
        trajectory_rgb_error_count = 0
        trajectory_rgb_error_gt_5_count = 0
        worst_diff = None
        for config_id, spec in CONFIG_SPECS.items():
            point_error = np.linalg.norm(keypoints[config_id] - canonical_points, axis=-1)
            max_keypoint_error = max(max_keypoint_error, float(np.max(point_error)))
            rgb_error = np.abs(rendered[config_id].astype(np.int16) - canonical_frames.astype(np.int16))
            max_rgb_error = max(max_rgb_error, int(np.max(rgb_error)))
            rgb_error_sum += float(np.sum(rgb_error))
            rgb_error_count += int(rgb_error.size)
            rgb_error_gt_5_count += int(np.count_nonzero(rgb_error > 5))
            trajectory_rgb_error_sum += float(np.sum(rgb_error))
            trajectory_rgb_error_count += int(rgb_error.size)
            trajectory_rgb_error_gt_5_count += int(np.count_nonzero(rgb_error > 5))
            frame_means = rgb_error.reshape(trajectory_length, -1).mean(axis=1)
            frame_index = int(np.argmax(frame_means))
            candidate = (float(frame_means[frame_index]), config_id, frame_index, rgb_error[frame_index])
            if worst_diff is None or candidate[0] > worst_diff[0]:
                worst_diff = candidate

            signs = np.asarray(spec["joint_signs"])
            canonicalized = datasets[config_id][demo_name]["actions"][:, :7] * signs
            action_error = np.max(np.abs(canonicalized - canonical_actions[:, :7]))
            max_action_error = max(max_action_error, float(action_error))

        combined_frames = []
        for frame_index in range(trajectory_length):
            panels = []
            for config_id in CONFIG_SPECS:
                panels.append(
                    _draw_action_panel(
                        rendered[config_id][frame_index],
                        config_id,
                        datasets[config_id][demo_name]["actions"][frame_index],
                        pixels[config_id][frame_index],
                        action_scale,
                    )
                )
            combined_frames.append(_tile_config_panels(panels))

        video_path = os.path.join(
            output_dir, f"{demo_name}_{len(CONFIG_SPECS)}_configs.mp4"
        )
        with imageio.get_writer(video_path, fps=20, codec="libx264", quality=8) as writer:
            for frame in combined_frames:
                writer.append_data(frame)
        sheet_indexes = [0, trajectory_length // 2, trajectory_length - 1]
        sheet = np.concatenate([combined_frames[index] for index in sheet_indexes], axis=0)
        imageio.imwrite(os.path.join(output_dir, f"{demo_name}_contact_sheet.jpg"), sheet)
        _, worst_config, worst_frame_index, worst_error = worst_diff
        heatmap = np.clip(worst_error * 8, 0, 255).astype(np.uint8)
        diff_review = np.concatenate(
            [
                canonical_frames[worst_frame_index],
                rendered[worst_config][worst_frame_index],
                heatmap,
            ],
            axis=1,
        )
        imageio.imwrite(os.path.join(output_dir, f"{demo_name}_worst_rgb_diff.png"), diff_review)

        successes = {config_id: datasets[config_id][demo_name]["success"] for config_id in CONFIG_SPECS}
        metrics["trajectories"][demo_name] = {
            "length": trajectory_length,
            "success": successes,
            "max_body_keypoint_error_m": max_keypoint_error,
            "max_rgb_abs_error": max_rgb_error,
            "mean_rgb_abs_error": trajectory_rgb_error_sum / max(trajectory_rgb_error_count, 1),
            "fraction_rgb_values_abs_error_gt_5": trajectory_rgb_error_gt_5_count
            / max(trajectory_rgb_error_count, 1),
            "worst_rgb_diff_config": worst_config,
            "worst_rgb_diff_frame": worst_frame_index,
            "max_canonicalized_action_error": max_action_error,
            "video": os.path.basename(video_path),
        }
        metrics["max_body_keypoint_error_m"] = max(metrics["max_body_keypoint_error_m"], max_keypoint_error)
        metrics["max_rgb_abs_error"] = max(metrics["max_rgb_abs_error"], max_rgb_error)
        metrics["max_canonicalized_action_error"] = max(metrics["max_canonicalized_action_error"], max_action_error)
        metrics["all_successful"] = metrics["all_successful"] and all(successes.values())

    metrics["mean_rgb_abs_error"] = rgb_error_sum / max(rgb_error_count, 1)
    metrics["fraction_rgb_values_abs_error_gt_5"] = rgb_error_gt_5_count / max(rgb_error_count, 1)
    metrics["acceptance"] = {
        "all_successful": bool(metrics["all_successful"]),
        "body_keypoint_error_below_1e-9_m": metrics["max_body_keypoint_error_m"] < 1e-9,
        "rgb_mean_abs_error_below_0_1": metrics["mean_rgb_abs_error"] < 0.1,
        "rgb_changed_fraction_below_1e-3": metrics["fraction_rgb_values_abs_error_gt_5"] < 1e-3,
        "canonicalized_actions_exact": metrics["max_canonicalized_action_error"] < 1e-12,
    }
    metrics["rgb_bitwise_exact"] = metrics["max_rgb_abs_error"] == 0
    metrics["accepted"] = all(metrics["acceptance"].values())
    metrics_path = os.path.join(output_dir, "validation.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    visualize_unit_test(args.input_dir, args.output_dir)
