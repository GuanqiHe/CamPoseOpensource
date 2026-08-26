"""Validate the pixel action Jacobian against finite differences."""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from equivalent_panda_configs import CONFIG_SPECS  # registers robot classes
from play_dataset import create_replay_env_from_dataset
from policy_robosuite.pixel_action_jacobian import (
    compute_pixel_action_jacobian,
    finite_difference_pixel_action_jacobian,
)


CAMERA_NAME = "agentview"
MAGNITUDE_THRESHOLD = 0.05


def _demo_names(dataset_path: str) -> list[str]:
    with h5py.File(dataset_path, "r") as dataset:
        return sorted(dataset["data"].keys())


def _frame_indexes(num_states: int, frames_per_demo: int) -> np.ndarray:
    return np.unique(
        np.linspace(0, num_states - 1, frames_per_demo).round().astype(int)
    )


def _load_states(dataset_path: str, demo_name: str) -> np.ndarray:
    with h5py.File(dataset_path, "r") as dataset:
        return dataset[f"data/{demo_name}/states"][()]


def _set_state(env, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def validate(
    dataset_dir: str,
    frames_per_demo: int,
    epsilon: float,
) -> dict:
    dataset_paths = {
        config_id: os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5")
        for config_id in CONFIG_SPECS
    }
    canonical_path = dataset_paths["cfg0"]
    errors = []
    relative_errors = []
    valid_cells_per_frame = []
    evaluated_entries_per_frame = []
    sign_transform_errors = []

    canonical_env, _, _ = create_replay_env_from_dataset(canonical_path)
    canonical_env.reset()
    for demo_name in _demo_names(canonical_path):
        states = _load_states(canonical_path, demo_name)
        for frame_index in _frame_indexes(len(states), frames_per_demo):
            _set_state(canonical_env, states[frame_index])
            analytic = compute_pixel_action_jacobian(
                canonical_env, CAMERA_NAME
            )
            finite_difference = finite_difference_pixel_action_jacobian(
                canonical_env,
                analytic,
                CAMERA_NAME,
                epsilon=epsilon,
            )
            valid_cells_per_frame.append(int(analytic.field[14].sum()))
            evaluated = np.abs(finite_difference) >= MAGNITUDE_THRESHOLD
            evaluated_entries_per_frame.append(int(evaluated.sum()))
            absolute_error = np.abs(
                analytic.field[:14].astype(np.float64)
                - finite_difference.astype(np.float64)
            )[evaluated]
            relative_error = absolute_error / np.abs(
                finite_difference.astype(np.float64)[evaluated]
            )
            errors.extend(absolute_error.tolist())
            relative_errors.extend(relative_error.tolist())
    canonical_env.close()

    canonical_states = {
        demo_name: _load_states(canonical_path, demo_name)
        for demo_name in _demo_names(canonical_path)
    }
    canonical_fields = {}
    canonical_env, _, _ = create_replay_env_from_dataset(canonical_path)
    canonical_env.reset()
    for demo_name, states in canonical_states.items():
        frame_index = int(_frame_indexes(len(states), frames_per_demo)[0])
        _set_state(canonical_env, states[frame_index])
        canonical_fields[demo_name] = compute_pixel_action_jacobian(
            canonical_env, CAMERA_NAME
        ).field
    canonical_env.close()

    for config_id, dataset_path in dataset_paths.items():
        env, _, _ = create_replay_env_from_dataset(dataset_path)
        env.reset()
        signs = np.asarray(CONFIG_SPECS[config_id]["joint_signs"])
        for demo_name in _demo_names(dataset_path):
            states = _load_states(dataset_path, demo_name)
            frame_index = int(_frame_indexes(len(states), frames_per_demo)[0])
            _set_state(env, states[frame_index])
            actual = compute_pixel_action_jacobian(env, CAMERA_NAME).field
            expected = canonical_fields[demo_name].copy()
            expected[:14] = (
                expected[:14].reshape(7, 2, 16, 16)
                * signs[:, None, None, None]
            ).reshape(14, 16, 16)
            sign_transform_errors.append(float(np.max(np.abs(actual - expected))))
        env.close()

    relative_errors = np.asarray(relative_errors)
    errors = np.asarray(errors)
    metrics = {
        "camera_name": CAMERA_NAME,
        "grid_shape": [16, 16],
        "channels": 15,
        "frames_checked": len(valid_cells_per_frame),
        "finite_difference_epsilon": epsilon,
        "magnitude_threshold_grid_pixel_per_action": MAGNITUDE_THRESHOLD,
        "min_valid_robot_cells_per_frame": int(min(valid_cells_per_frame)),
        "min_evaluated_entries_per_frame": int(min(evaluated_entries_per_frame)),
        "median_relative_error": float(np.median(relative_errors)),
        "p95_relative_error": float(np.percentile(relative_errors, 95)),
        "max_absolute_error_grid_pixel_per_action": float(np.max(errors)),
        "max_sign_transform_error": float(max(sign_transform_errors)),
    }
    metrics["gate_passed"] = bool(
        metrics["min_valid_robot_cells_per_frame"] >= 10
        and metrics["min_evaluated_entries_per_frame"] > 0
        and metrics["median_relative_error"] < 0.02
        and metrics["p95_relative_error"] < 0.10
        and metrics["max_sign_transform_error"] < 1e-6
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--frames-per-demo", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    metrics = validate(
        os.path.abspath(os.path.expanduser(args.dataset_dir)),
        args.frames_per_demo,
        args.epsilon,
    )
    output = json.dumps(metrics, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        output_path = os.path.abspath(os.path.expanduser(args.output_json))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(output + "\n")


if __name__ == "__main__":
    main()
