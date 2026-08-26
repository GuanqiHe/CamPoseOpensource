"""Derive exactly paired joint-sign variants from a canonical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from equivalent_panda_configs import CONFIG_SPECS
from generate_equivalent_urdf_unit_test import _write_variant_dataset
from play_dataset import create_replay_env_from_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demo_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as dataset:
        return sorted(dataset["data"].keys())


def _verify_transforms(
    dataset_paths: dict[str, Path],
    qpos_indexes: np.ndarray,
    qvel_indexes: np.ndarray,
    nq: int,
) -> dict:
    max_action_error = 0.0
    max_state_error = 0.0
    canonical_path = dataset_paths["cfg0"]
    with h5py.File(canonical_path, "r") as canonical:
        for config_id, path in dataset_paths.items():
            signs = np.asarray(CONFIG_SPECS[config_id]["joint_signs"])
            with h5py.File(path, "r") as variant:
                for demo_name in canonical["data"]:
                    canonical_demo = canonical[f"data/{demo_name}"]
                    variant_demo = variant[f"data/{demo_name}"]

                    expected_actions = canonical_demo["actions"][()].copy()
                    expected_actions[:, :7] *= signs
                    max_action_error = max(
                        max_action_error,
                        float(
                            np.max(
                                np.abs(
                                    variant_demo["actions"][()]
                                    - expected_actions
                                )
                            )
                        ),
                    )

                    expected_states = canonical_demo["states"][()].copy()
                    expected_states[:, 1 + qpos_indexes] *= signs
                    expected_states[:, 1 + nq + qvel_indexes] *= signs
                    max_state_error = max(
                        max_state_error,
                        float(
                            np.max(
                                np.abs(
                                    variant_demo["states"][()]
                                    - expected_states
                                )
                            )
                        ),
                    )
    return {
        "max_action_transform_error": max_action_error,
        "max_state_transform_error": max_state_error,
    }


def derive(canonical_path: Path, output_dir: Path) -> dict:
    canonical_path = canonical_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)

    dataset_paths = {
        config_id: output_dir / f"{config_id}_joint_delta.hdf5"
        for config_id in CONFIG_SPECS
    }
    for path in dataset_paths.values():
        if path.exists() or Path(str(path) + ".incomplete").exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    env, _, action_space = create_replay_env_from_dataset(str(canonical_path))
    if action_space != "joint_delta":
        env.close()
        raise ValueError(f"Expected joint_delta, got {action_space!r}")
    qpos_indexes = np.asarray(
        env.robots[0]._ref_joint_pos_indexes,
        dtype=np.int64,
    )
    qvel_indexes = np.asarray(
        env.robots[0]._ref_joint_vel_indexes,
        dtype=np.int64,
    )
    nq = env.sim.model.nq
    env.close()

    for config_id, target_path in dataset_paths.items():
        spec = CONFIG_SPECS[config_id]
        incomplete_path = Path(str(target_path) + ".incomplete")
        _write_variant_dataset(
            canonical_path=str(canonical_path),
            variant_path=str(incomplete_path),
            config_id=config_id,
            robot_name=spec["robot"],
            signs=np.asarray(spec["joint_signs"], dtype=np.float64),
            robot_qpos_indexes=qpos_indexes,
            robot_qvel_indexes=qvel_indexes,
            nq=nq,
        )
        os.replace(incomplete_path, target_path)

    verification = _verify_transforms(
        dataset_paths,
        qpos_indexes,
        qvel_indexes,
        nq,
    )
    manifest = {
        "canonical_dataset": str(canonical_path),
        "canonical_sha256": _sha256(canonical_path),
        "num_physical_episodes": len(_demo_names(canonical_path)),
        "num_configs": len(CONFIG_SPECS),
        "num_logical_episodes": len(_demo_names(canonical_path))
        * len(CONFIG_SPECS),
        "pairing": "same physical trajectory; q_cfg=S_cfg*q0; a_cfg=S_cfg*a0",
        "configs": {
            config_id: {
                **CONFIG_SPECS[config_id],
                "dataset": str(path),
                "sha256": _sha256(path),
            }
            for config_id, path in dataset_paths.items()
        },
        **verification,
    }
    manifest_path = output_dir / "paired_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    derive(args.canonical_dataset, args.output_dir)
