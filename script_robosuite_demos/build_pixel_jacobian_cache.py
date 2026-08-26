"""Build the exact-paired RGB / action / pixel-Jacobian training cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import h5py
import numpy as np
from tqdm import tqdm

from equivalent_panda_configs import CONFIG_SPECS  # registers robot classes
from play_dataset import create_replay_env_from_dataset
from policy_robosuite.pixel_action_jacobian import compute_pixel_action_jacobian
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)


CAMERA_NAME = "agentview"
IMAGE_SIZE = 256
GRID_SIZE = 16


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _demo_names(dataset_path: str) -> list[str]:
    with h5py.File(dataset_path, "r") as dataset:
        return sorted(dataset["data"].keys())


def _load_demo(dataset_path: str, demo_name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, "r") as dataset:
        demo = dataset[f"data/{demo_name}"]
        return demo["states"][()], demo["actions"][()].astype(np.float32)


def build_cache(dataset_dir: str, output_path: str) -> dict:
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {output_path}"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".incomplete"
    if os.path.exists(temporary_path):
        raise FileExistsError(
            f"Remove or inspect incomplete cache first: {temporary_path}"
        )

    config_ids = list(CONFIG_SPECS)
    dataset_paths = {
        config_id: os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5")
        for config_id in config_ids
    }
    for path in dataset_paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    canonical_path = dataset_paths["cfg0"]
    demo_names = _demo_names(canonical_path)
    total_physical_steps = sum(
        len(_load_demo(canonical_path, demo_name)[0])
        for demo_name in demo_names
    )
    all_actions = []
    actions_by_demo = {}
    for demo_name in demo_names:
        per_config = []
        expected_length = None
        for config_id in config_ids:
            _, actions = _load_demo(dataset_paths[config_id], demo_name)
            if expected_length is None:
                expected_length = len(actions)
            elif len(actions) != expected_length:
                raise ValueError(f"Unpaired length for {config_id}/{demo_name}")
            per_config.append(actions)
            all_actions.append(actions)
        actions_by_demo[demo_name] = np.stack(per_config)

    action_array = np.concatenate(all_actions, axis=0)
    action_mean = action_array.mean(axis=0).astype(np.float32)
    action_std = np.maximum(action_array.std(axis=0), 1e-4).astype(np.float32)

    env, _, action_space = create_replay_env_from_dataset(canonical_path)
    if action_space != "joint_delta":
        env.close()
        raise ValueError(f"Expected joint_delta, got {action_space!r}")
    env.reset()
    intrinsic = get_camera_intrinsic_matrix(
        env.sim, CAMERA_NAME, GRID_SIZE, GRID_SIZE
    )
    extrinsic = get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)

    jacobian_square_sum = np.zeros(14, dtype=np.float64)
    jacobian_nonzero_count = np.zeros(14, dtype=np.int64)
    num_physical_steps = 0

    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["schema_version"] = 1
            output.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["git_commit"] = _git_commit()
            output.attrs["camera_name"] = CAMERA_NAME
            output.attrs["image_size"] = IMAGE_SIZE
            output.attrs["grid_size"] = GRID_SIZE
            output.attrs["channel_order"] = (
                "du_da1,dv_da1,...,du_da7,dv_da7,robot_mask"
            )
            output.attrs["input_includes_q"] = False
            output.attrs["pairing"] = "canonical RGB duplicated across configs"
            output.attrs["camera_intrinsic_16x16"] = json.dumps(intrinsic.tolist())
            output.attrs["camera_to_world"] = json.dumps(extrinsic.tolist())
            output.attrs["source_sha256"] = json.dumps(
                {key: _sha256(value) for key, value in dataset_paths.items()},
                sort_keys=True,
            )
            output.create_dataset(
                "config_ids",
                data=np.asarray(config_ids, dtype=h5py.string_dtype()),
            )
            output.create_dataset(
                "joint_signs",
                data=np.asarray(
                    [CONFIG_SPECS[key]["joint_signs"] for key in config_ids],
                    dtype=np.int8,
                ),
            )
            output.create_dataset("action_mean", data=action_mean)
            output.create_dataset("action_std", data=action_std)

            demos_group = output.create_group("demos")
            progress = tqdm(
                total=total_physical_steps,
                desc="Building pixel-Jacobian cache",
                unit="frame",
            )
            for demo_name in demo_names:
                states, _ = _load_demo(canonical_path, demo_name)
                group = demos_group.create_group(demo_name)
                rgb_dataset = group.create_dataset(
                    "rgb",
                    shape=(len(states), IMAGE_SIZE, IMAGE_SIZE, 3),
                    dtype=np.uint8,
                    chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                    compression="gzip",
                    compression_opts=1,
                )
                jacobian_dataset = group.create_dataset(
                    "canonical_pixel_jacobian",
                    shape=(len(states), 15, GRID_SIZE, GRID_SIZE),
                    dtype=np.float32,
                    chunks=(1, 15, GRID_SIZE, GRID_SIZE),
                    compression="gzip",
                    compression_opts=1,
                )
                group.create_dataset("actions", data=actions_by_demo[demo_name])
                for frame_index, state in enumerate(states):
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    rgb_dataset[frame_index] = env.sim.render(
                        camera_name=CAMERA_NAME,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                    )[::-1]
                    field = compute_pixel_action_jacobian(
                        env, CAMERA_NAME, GRID_SIZE, GRID_SIZE
                    ).field
                    jacobian_dataset[frame_index] = field
                    mask = field[14].astype(bool)
                    values = field[:14, mask]
                    jacobian_square_sum += np.square(values).sum(axis=1)
                    jacobian_nonzero_count += mask.sum()
                    num_physical_steps += 1
                    progress.update(1)

            progress.close()

            jacobian_rms = np.sqrt(
                jacobian_square_sum / np.maximum(jacobian_nonzero_count, 1)
            ).astype(np.float32)
            output.create_dataset(
                "jacobian_channel_rms",
                data=np.maximum(jacobian_rms, 1e-4),
            )
            output.attrs["num_physical_steps"] = num_physical_steps
            output.attrs["num_paired_samples"] = (
                num_physical_steps * len(config_ids)
            )
        os.replace(temporary_path, output_path)
    finally:
        env.close()

    return {
        "output_path": output_path,
        "num_demos": len(demo_names),
        "num_configs": len(config_ids),
        "num_physical_steps": num_physical_steps,
        "num_paired_samples": num_physical_steps * len(config_ids),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "jacobian_channel_rms": jacobian_rms.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = build_cache(
        os.path.abspath(os.path.expanduser(args.dataset_dir)),
        os.path.abspath(os.path.expanduser(args.output)),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
