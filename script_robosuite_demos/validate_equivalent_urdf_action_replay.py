"""Validate equivalent robot coordinates through controller action replay.

Unlike ``visualize_equivalent_urdf_unit_test.py``, which renders recorded
MuJoCo states, this script resets each configured robot to the paired initial
state and executes the configuration-specific joint-delta actions through the
real robosuite controller.  It then compares the resulting physical robot
trajectories across configurations.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import h5py
import imageio.v2 as imageio
import numpy as np

from equivalent_panda_configs import CONFIG_SPECS  # registers robot classes
from play_dataset import create_replay_env_from_dataset
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)
from robosuite.wrappers.action_wrapper import wrap_env_action_space
from visualize_equivalent_urdf_unit_test import (
    BODY_NAMES,
    CAMERA_NAME,
    FRAME_SIZE,
    _draw_action_panel,
    _keypoints_world,
    _tile_config_panels,
)


KEYPOINT_ERROR_THRESHOLD_M = 2e-4
EEF_ERROR_THRESHOLD_M = 2e-4
QPOS_ERROR_THRESHOLD_RAD = 5e-4
RESET_SEED = 0


@dataclass
class ReplayResult:
    keypoints: np.ndarray
    qpos: np.ndarray
    frames: np.ndarray
    pixels: np.ndarray
    success: bool
    success_step: int | None


def _load_demo(dataset_path: str, demo_name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, "r") as dataset:
        demo = dataset[f"data/{demo_name}"]
        return demo["states"][()], demo["actions"][()]


def _demo_names(dataset_path: str) -> list[str]:
    with h5py.File(dataset_path, "r") as dataset:
        return sorted(name for name in dataset["data"] if name.startswith("demo_"))


def _render(env) -> tuple[np.ndarray, np.ndarray]:
    frame = env.sim.render(
        camera_name=CAMERA_NAME,
        height=FRAME_SIZE,
        width=FRAME_SIZE,
    )
    frame = np.flipud(frame).astype(np.uint8)
    transform = get_camera_transform_matrix(
        env.sim,
        CAMERA_NAME,
        FRAME_SIZE,
        FRAME_SIZE,
    )
    pixels = project_points_from_world_to_camera(
        _keypoints_world(env),
        transform,
        FRAME_SIZE,
        FRAME_SIZE,
    )
    return frame, pixels


def _restore_deterministic_state(env, state: np.ndarray) -> None:
    """Restore recorded state and clear dynamics omitted by MjSimState.flatten."""
    env.sim.set_state_from_flattened(state)
    data = env.sim.data
    for field in (
        "qacc_warmstart",
        "qacc",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
    ):
        values = getattr(data, field, None)
        if values is not None:
            values[...] = 0
    if getattr(data, "act", None) is not None:
        data.act[...] = 0
    env.sim.forward()
    # mj_forward may update acceleration-related buffers.  Warm-start must be
    # zero immediately before the first controller step for paired rollouts.
    data.qacc_warmstart[...] = 0


def _replay_demo(dataset_path: str, demo_name: str) -> ReplayResult:
    states, actions = _load_demo(dataset_path, demo_name)
    np.random.seed(RESET_SEED)
    env, _, action_space = create_replay_env_from_dataset(dataset_path)
    if action_space != "joint_delta":
        env.close()
        raise ValueError(f"Expected joint_delta dataset, got {action_space!r}")

    np.random.seed(RESET_SEED)
    env.reset()
    _restore_deterministic_state(env, states[0])
    env = wrap_env_action_space(env, action_space)
    env.set_init_action()

    joint_indexes = env.robots[0]._ref_joint_pos_indexes
    keypoints = [_keypoints_world(env)]
    qpos = [env.sim.data.qpos[joint_indexes].copy()]
    first_frame, first_pixels = _render(env)
    frames = [first_frame]
    pixels = [first_pixels]
    success = bool(env._check_success())
    success_step = 0 if success else None

    for step, action in enumerate(actions, start=1):
        _, reward, _, _ = env.step(action)
        current_success = bool(reward == 1 or env._check_success())
        if current_success and not success:
            success_step = step
        success = success or current_success
        keypoints.append(_keypoints_world(env))
        qpos.append(env.sim.data.qpos[joint_indexes].copy())
        frame, projected = _render(env)
        frames.append(frame)
        pixels.append(projected)

    env.close()
    return ReplayResult(
        keypoints=np.asarray(keypoints),
        qpos=np.asarray(qpos),
        frames=np.asarray(frames),
        pixels=np.asarray(pixels),
        success=success,
        success_step=success_step,
    )


def _trajectory_metrics(
    replays: dict[str, ReplayResult],
) -> tuple[
    dict[str, dict[str, float | bool | int | None]],
    float,
    float,
    float,
    bool,
]:
    canonical = replays["cfg0"]
    per_config = {}
    max_keypoint_error = 0.0
    max_eef_error = 0.0
    max_qpos_error = 0.0
    all_success_steps_match = True
    for config_id, replay in replays.items():
        if replay.keypoints.shape != canonical.keypoints.shape:
            raise RuntimeError(
                f"Unpaired replay shape for {config_id}: "
                f"{replay.keypoints.shape} != {canonical.keypoints.shape}"
            )
        signs = np.asarray(CONFIG_SPECS[config_id]["joint_signs"])
        keypoint_error = np.linalg.norm(
            replay.keypoints - canonical.keypoints,
            axis=-1,
        )
        canonicalized_qpos = replay.qpos * signs
        qpos_error = np.abs(canonicalized_qpos - canonical.qpos)
        config_max_keypoint_error = float(np.max(keypoint_error))
        config_rmse_keypoint_error = float(np.sqrt(np.mean(np.square(keypoint_error))))
        config_max_eef_error = float(np.max(keypoint_error[:, -1]))
        config_max_qpos_error = float(np.max(qpos_error))
        max_keypoint_error = max(max_keypoint_error, config_max_keypoint_error)
        max_eef_error = max(max_eef_error, config_max_eef_error)
        max_qpos_error = max(max_qpos_error, config_max_qpos_error)
        all_success_steps_match &= replay.success_step == canonical.success_step
        per_config[config_id] = {
            "success": replay.success,
            "success_step": replay.success_step,
            "max_body_keypoint_error_m": config_max_keypoint_error,
            "body_keypoint_rmse_m": config_rmse_keypoint_error,
            "max_eef_position_error_m": config_max_eef_error,
            "max_canonicalized_qpos_error_rad": config_max_qpos_error,
        }
    return (
        per_config,
        max_keypoint_error,
        max_eef_error,
        max_qpos_error,
        all_success_steps_match,
    )


def _write_video(
    output_dir: str,
    demo_name: str,
    replays: dict[str, ReplayResult],
    actions_by_config: dict[str, np.ndarray],
) -> str:
    num_frames = len(replays["cfg0"].frames)
    action_scale = max(
        float(np.max(np.abs(actions[:, :7])))
        for actions in actions_by_config.values()
    )
    action_scale = max(action_scale, 1e-6)
    combined_frames = []
    for frame_index in range(num_frames):
        panels = []
        for config_id in CONFIG_SPECS:
            action = (
                np.zeros(8, dtype=np.float64)
                if frame_index == 0
                else actions_by_config[config_id][frame_index - 1]
            )
            panels.append(
                _draw_action_panel(
                    replays[config_id].frames[frame_index],
                    config_id,
                    action,
                    replays[config_id].pixels[frame_index],
                    action_scale,
                )
            )
        combined_frames.append(_tile_config_panels(panels))

    video_path = os.path.join(
        output_dir,
        f"{demo_name}_action_replay_{len(CONFIG_SPECS)}_configs.mp4",
    )
    with imageio.get_writer(video_path, fps=20, codec="libx264", quality=8) as writer:
        for frame in combined_frames:
            writer.append_data(frame)

    indexes = [0, num_frames // 2, num_frames - 1]
    contact_sheet = np.concatenate([combined_frames[index] for index in indexes], axis=0)
    imageio.imwrite(
        os.path.join(output_dir, f"{demo_name}_action_replay_contact_sheet.jpg"),
        contact_sheet,
    )
    return os.path.basename(video_path)


def validate_action_replay(input_dir: str, output_dir: str) -> dict:
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(input_dir, "datasets")
    dataset_paths = {
        config_id: os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5")
        for config_id in CONFIG_SPECS
    }
    demo_names = _demo_names(dataset_paths["cfg0"])

    metrics = {
        "validation_mode": "controller action replay",
        "input_data_modified": False,
        "num_configs": len(CONFIG_SPECS),
        "num_trajectories_per_config": len(demo_names),
        "body_keypoints": BODY_NAMES + ["eef"],
        "trajectories": {},
        "all_action_replays_successful": True,
        "all_success_steps_match": True,
        "max_body_keypoint_error_m": 0.0,
        "max_eef_position_error_m": 0.0,
        "max_canonicalized_qpos_error_rad": 0.0,
    }

    for demo_name in demo_names:
        replays = {}
        actions_by_config = {}
        for config_id, dataset_path in dataset_paths.items():
            _, actions_by_config[config_id] = _load_demo(dataset_path, demo_name)
            replays[config_id] = _replay_demo(dataset_path, demo_name)

        (
            per_config,
            max_keypoint_error,
            max_eef_error,
            max_qpos_error,
            success_steps_match,
        ) = _trajectory_metrics(replays)
        video = _write_video(
            output_dir,
            demo_name,
            replays,
            actions_by_config,
        )
        all_successful = all(result.success for result in replays.values())
        metrics["trajectories"][demo_name] = {
            "num_actions": len(actions_by_config["cfg0"]),
            "configs": per_config,
            "all_action_replays_successful": all_successful,
            "all_success_steps_match": success_steps_match,
            "max_body_keypoint_error_m": max_keypoint_error,
            "max_eef_position_error_m": max_eef_error,
            "max_canonicalized_qpos_error_rad": max_qpos_error,
            "video": video,
        }
        metrics["all_action_replays_successful"] &= all_successful
        metrics["all_success_steps_match"] &= success_steps_match
        metrics["max_body_keypoint_error_m"] = max(
            metrics["max_body_keypoint_error_m"],
            max_keypoint_error,
        )
        metrics["max_canonicalized_qpos_error_rad"] = max(
            metrics["max_canonicalized_qpos_error_rad"],
            max_qpos_error,
        )
        metrics["max_eef_position_error_m"] = max(
            metrics["max_eef_position_error_m"],
            max_eef_error,
        )

    metrics["acceptance"] = {
        "all_action_replays_successful": metrics["all_action_replays_successful"],
        "all_success_steps_match": metrics["all_success_steps_match"],
        "body_keypoint_error_below_2e-4_m": (
            metrics["max_body_keypoint_error_m"] < KEYPOINT_ERROR_THRESHOLD_M
        ),
        "eef_position_error_below_2e-4_m": (
            metrics["max_eef_position_error_m"] < EEF_ERROR_THRESHOLD_M
        ),
        "canonicalized_qpos_error_below_5e-4_rad": (
            metrics["max_canonicalized_qpos_error_rad"] < QPOS_ERROR_THRESHOLD_RAD
        ),
    }
    metrics["accepted"] = all(metrics["acceptance"].values())
    metrics_path = os.path.join(output_dir, "action_replay_validation.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    result = validate_action_replay(args.input_dir, args.output_dir)
    raise SystemExit(0 if result["accepted"] else 1)
