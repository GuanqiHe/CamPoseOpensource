"""Validate success-only official Lift demonstration datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from play_dataset import create_replay_env_from_dataset
from robosuite.wrappers.action_wrapper import wrap_env_action_space


def _demo_sort_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _render_cube_distribution(xy: np.ndarray, output_path: Path) -> None:
    size = 800
    margin = 80
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    plot_min, plot_max = -0.035, 0.035

    def project(values: np.ndarray) -> np.ndarray:
        normalized = (values - plot_min) / (plot_max - plot_min)
        return margin + normalized * (size - 2 * margin)

    lo = int(project(np.array([-0.03]))[0])
    hi = int(project(np.array([0.03]))[0])
    cv2.rectangle(canvas, (lo, size - hi), (hi, size - lo), (40, 40, 40), 2)

    projected = project(xy)
    for index, (px, py) in enumerate(projected):
        point = (int(round(px)), int(round(size - py)))
        cv2.circle(canvas, point, 7, (20, 90, 220), -1, lineType=cv2.LINE_AA)
        if len(xy) <= 20:
            cv2.putText(
                canvas,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                lineType=cv2.LINE_AA,
            )

    cv2.putText(canvas, "Official Lift initial cube XY", (margin, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(canvas, "outlined range: [-0.03, 0.03] m", (margin, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(output_path), canvas)


def validate_dataset(
    dataset_path: Path,
    expected_demos: int,
    replay_actions: bool,
) -> dict:
    with h5py.File(dataset_path, "r") as dataset:
        data = dataset["data"]
        demo_names = sorted(
            (name for name in data if name.startswith("demo_")),
            key=_demo_sort_key,
        )
        if len(demo_names) != expected_demos:
            raise AssertionError(f"Expected {expected_demos} demos, found {len(demo_names)}")

        env_args = json.loads(data.attrs["env_args"])
        if env_args["env_name"] != "Lift":
            raise AssertionError(f"Expected official Lift, got {env_args['env_name']!r}")
        if str(data.attrs["action_space"]) != "joint_delta":
            raise AssertionError(f"Expected joint_delta, got {data.attrs['action_space']!r}")

        cube_positions = []
        recorded_success = []
        success_steps = []
        episode_lengths = []
        for demo_name in demo_names:
            demo = data[demo_name]
            cube_positions.append(np.asarray(demo.attrs["initial_cube_pos"], dtype=np.float64))
            success = bool(demo.attrs["success"])
            success_step = int(demo.attrs["success_step"])
            reward_success = bool(np.any(np.isclose(demo["rewards"][()], 1.0)))
            recorded_success.append(success and reward_success and success_step >= 0)
            success_steps.append(success_step)
            episode_lengths.append(len(demo["actions"]))

    cube_positions_array = np.stack(cube_positions)
    xy = cube_positions_array[:, :2]
    rounded_unique_xy = np.unique(np.round(xy, decimals=6), axis=0)
    within_official_range = bool(np.all((xy >= -0.030001) & (xy <= 0.030001)))

    replay_success = []
    if replay_actions:
        env, _, action_space = create_replay_env_from_dataset(
            str(dataset_path),
            enable_rendering=False,
        )
        if action_space in ("eef_delta", "joint_delta"):
            env = wrap_env_action_space(env, action_space)
        try:
            with h5py.File(dataset_path, "r") as dataset:
                for demo_name in demo_names:
                    demo = dataset["data"][demo_name]
                    env.reset()
                    env.sim.set_state_from_flattened(demo["states"][0])
                    env.sim.forward()
                    if action_space in ("eef_delta", "joint_delta"):
                        env.set_init_action()
                    succeeded = bool(env._check_success())
                    for action in demo["actions"]:
                        env.step(action)
                        succeeded = succeeded or bool(env._check_success())
                    replay_success.append(succeeded)
        finally:
            env.close()

    metrics = {
        "dataset_path": str(dataset_path),
        "expected_demos": expected_demos,
        "num_demos": len(demo_names),
        "recorded_success_count": int(np.sum(recorded_success)),
        "action_replay_checked": replay_actions,
        "action_replay_success_count": int(np.sum(replay_success)) if replay_actions else None,
        "unique_initial_cube_xy": int(len(rounded_unique_xy)),
        "initial_cube_xy_min_m": xy.min(axis=0).tolist(),
        "initial_cube_xy_max_m": xy.max(axis=0).tolist(),
        "initial_cube_xy_std_m": xy.std(axis=0).tolist(),
        "initial_cube_xy_within_official_range": within_official_range,
        "success_step_min": int(np.min(success_steps)),
        "success_step_median": float(np.median(success_steps)),
        "success_step_max": int(np.max(success_steps)),
        "episode_length_min": int(np.min(episode_lengths)),
        "episode_length_median": float(np.median(episode_lengths)),
        "episode_length_max": int(np.max(episode_lengths)),
    }

    if metrics["recorded_success_count"] != expected_demos:
        raise AssertionError("Not every stored demo is marked successful with a success reward")
    if replay_actions and metrics["action_replay_success_count"] != expected_demos:
        raise AssertionError("Not every stored demo succeeds under joint-delta action replay")
    if metrics["unique_initial_cube_xy"] != expected_demos:
        raise AssertionError("Initial cube XY positions are not unique")
    if not within_official_range:
        raise AssertionError("Initial cube XY exceeds the official Lift randomization range")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--expected_demos", type=int, required=True)
    parser.add_argument("--replay_actions", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = validate_dataset(
        dataset_path=dataset_path,
        expected_demos=args.expected_demos,
        replay_actions=args.replay_actions,
    )
    metrics_path = output_dir / "quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    with h5py.File(dataset_path, "r") as dataset:
        demo_names = sorted(
            (name for name in dataset["data"] if name.startswith("demo_")),
            key=_demo_sort_key,
        )
        cube_xy = np.stack(
            [np.asarray(dataset["data"][name].attrs["initial_cube_pos"])[:2] for name in demo_names]
        )
    _render_cube_distribution(cube_xy, output_dir / "initial_cube_xy.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
