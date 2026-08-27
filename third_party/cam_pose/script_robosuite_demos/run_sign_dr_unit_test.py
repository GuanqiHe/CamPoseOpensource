#!/usr/bin/env python3
"""Build and validate the eight-config multi-flip unit-test artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

import validate_equivalent_urdf_action_replay as action_replay
import validate_pixel_action_jacobian as jacobian_validation
import visualize_equivalent_urdf_unit_test as trajectory_visualization
from equivalent_panda_configs import SIGN_DR_TRAIN_CONFIG_SPECS
from generate_equivalent_urdf_unit_test import _write_variant_dataset
from play_dataset import create_replay_env_from_dataset
from action_jacobian.representation import compute_pixel_action_jacobian


CANONICAL_ID = "sign_train_00"


def _unit_test_specs() -> dict[str, dict]:
    """Alias the canonical entry to cfg0 for the established validators."""

    return {
        "cfg0" if config_id == CANONICAL_ID else config_id: spec
        for config_id, spec in SIGN_DR_TRAIN_CONFIG_SPECS.items()
    }


def _derive_datasets(
    canonical_path: Path,
    output_dir: Path,
    specs: dict[str, dict],
) -> dict[str, Path]:
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    env, _, action_space = create_replay_env_from_dataset(str(canonical_path))
    if action_space != "joint_delta":
        env.close()
        raise ValueError(f"Expected joint_delta, got {action_space!r}")
    qpos_indexes = np.asarray(env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)
    qvel_indexes = np.asarray(env.robots[0]._ref_joint_vel_indexes, dtype=np.int64)
    nq = env.sim.model.nq
    env.close()

    paths = {}
    for config_id, spec in specs.items():
        path = dataset_dir / f"{config_id}_joint_delta.hdf5"
        if path.exists():
            path.unlink()
        _write_variant_dataset(
            canonical_path=str(canonical_path),
            variant_path=str(path),
            config_id=config_id,
            robot_name=spec["robot"],
            signs=np.asarray(spec["joint_signs"], dtype=np.float64),
            robot_qpos_indexes=qpos_indexes,
            robot_qvel_indexes=qvel_indexes,
            nq=nq,
        )
        paths[config_id] = path
    return paths


def _render_signed_layer(layer: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(layer / max(scale, 1e-6), -1.0, 1.0)
    image = np.zeros((*layer.shape, 3), dtype=np.uint8)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    image[..., 0] = (positive * 255).astype(np.uint8)
    image[..., 1] = ((positive + negative) * 90).astype(np.uint8)
    image[..., 2] = (negative * 255).astype(np.uint8)
    return cv2.resize(image, (112, 112), interpolation=cv2.INTER_NEAREST)


def _jacobian_layer_image(
    field: np.ndarray,
    channel_scales: np.ndarray,
    title: str,
) -> np.ndarray:
    cell_height, cell_width = 140, 128
    canvas = np.zeros((3 * cell_height + 28, 5 * cell_width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for channel in range(15):
        row, column = divmod(channel, 5)
        y0 = 28 + row * cell_height
        x0 = column * cell_width
        if channel == 14:
            layer = np.repeat((field[channel] * 255).astype(np.uint8)[..., None], 3, axis=2)
            layer = cv2.resize(layer, (112, 112), interpolation=cv2.INTER_NEAREST)
            label = "robot_mask"
        else:
            layer = _render_signed_layer(field[channel], channel_scales[channel])
            label = f"{'du' if channel % 2 == 0 else 'dv'}/da{channel // 2}"
        canvas[y0 : y0 + 112, x0 + 8 : x0 + 120] = layer
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _render_jacobian_layers(
    dataset_paths: dict[str, Path],
    specs: dict[str, dict],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_paths["cfg0"], "r") as source:
        demo_name = sorted(source["data"])[0]
        states = source[f"data/{demo_name}/states"][()]
    frame_index = len(states) // 2
    fields = {}
    for config_id, path in dataset_paths.items():
        with h5py.File(path, "r") as source:
            state = source[f"data/{demo_name}/states"][frame_index]
        env, _, _ = create_replay_env_from_dataset(str(path))
        env.reset()
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        fields[config_id] = compute_pixel_action_jacobian(
            env, "agentview", grid_height=16, grid_width=16
        ).field
        env.close()

    canonical = fields["cfg0"]
    channel_scales = np.maximum(np.max(np.abs(canonical), axis=(1, 2)), 1e-6)
    max_transform_error = 0.0
    rendered = []
    for config_id, field in fields.items():
        signs = np.asarray(specs[config_id]["joint_signs"])
        expected = canonical.copy()
        expected[:14] = (
            expected[:14].reshape(7, 2, 16, 16)
            * signs[:, None, None, None]
        ).reshape(14, 16, 16)
        max_transform_error = max(
            max_transform_error, float(np.max(np.abs(field - expected)))
        )
        image = _jacobian_layer_image(field, channel_scales, config_id)
        cv2.imwrite(str(output_dir / f"{config_id}_all_15_layers.png"), image)
        rendered.append(image)
    contact_sheet = trajectory_visualization._tile_config_panels(rendered)
    cv2.imwrite(str(output_dir / "all_configs_all_layers_contact_sheet.png"), contact_sheet)
    result = {
        "demo_name": demo_name,
        "frame_index": frame_index,
        "max_direct_vs_sign_derived_error": max_transform_error,
        "accepted": max_transform_error < 1e-6,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run(args: argparse.Namespace) -> dict:
    canonical_path = Path(args.canonical_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _unit_test_specs()
    dataset_paths = _derive_datasets(canonical_path, output_dir, specs)

    trajectory_visualization.visualize_unit_test(
        str(output_dir),
        str(output_dir / "trajectory_visualization"),
        specs,
    )
    trajectory_metrics = json.loads(
        (output_dir / "trajectory_visualization" / "validation.json").read_text()
    )
    replay_metrics = action_replay.validate_action_replay(
        str(output_dir),
        str(output_dir / "action_replay"),
        specs,
    )
    jacobian_metrics = jacobian_validation.validate(
        str(output_dir / "datasets"),
        frames_per_demo=args.frames_per_demo,
        epsilon=args.epsilon,
        config_specs=specs,
    )
    (output_dir / "jacobian_validation.json").write_text(
        json.dumps(jacobian_metrics, indent=2, sort_keys=True)
    )
    jacobian_visualization = _render_jacobian_layers(
        dataset_paths, specs, output_dir / "jacobian_layers"
    )
    summary = {
        "canonical_dataset": str(canonical_path),
        "num_configs": len(specs),
        "num_trajectories": trajectory_metrics["num_trajectories_per_config"],
        "trajectory_visualization_accepted": trajectory_metrics["accepted"],
        "action_replay_accepted": replay_metrics["accepted"],
        "jacobian_validation_accepted": jacobian_metrics["gate_passed"],
        "jacobian_visualization_accepted": jacobian_visualization["accepted"],
    }
    summary["accepted"] = all(
        value for key, value in summary.items() if key.endswith("_accepted")
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-demo", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
