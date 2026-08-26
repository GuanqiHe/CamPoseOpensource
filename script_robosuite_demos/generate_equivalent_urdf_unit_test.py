"""Generate paired LiftRand data for equivalent robot-coordinate models."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import h5py
import numpy as np

from equivalent_panda_configs import CONFIG_SPECS
from gen_robosuite_format_demo import create_demo_env, generate_demos


def _transform_joint_observations(group: h5py.Group, signs: np.ndarray) -> None:
    for subgroup_name in ("obs", "next_obs"):
        subgroup = group[subgroup_name]
        for key in ("robot0_joint_pos", "robot0_joint_pos_sin", "robot0_joint_vel"):
            values = subgroup[key][()]
            subgroup[key][...] = values * signs
        # cos(sign * q) == cos(q), so robot0_joint_pos_cos is unchanged.


def _write_variant_dataset(
    canonical_path: str,
    variant_path: str,
    config_id: str,
    robot_name: str,
    signs: np.ndarray,
    robot_qpos_indexes: np.ndarray,
    robot_qvel_indexes: np.ndarray,
    nq: int,
) -> None:
    if os.path.abspath(canonical_path) != os.path.abspath(variant_path):
        shutil.copy2(canonical_path, variant_path)

    with h5py.File(variant_path, "r+") as dataset:
        data = dataset["data"]
        env_args = json.loads(data.attrs["env_args"])
        env_args["env_kwargs"]["robots"] = [robot_name]
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["config_id"] = config_id
        data.attrs["robot_name"] = robot_name
        data.attrs["joint_signs"] = json.dumps(signs.astype(int).tolist())
        data.attrs["paired_source"] = os.path.basename(canonical_path)

        if np.all(signs == 1):
            return

        for demo_name in sorted(data.keys()):
            demo = data[demo_name]
            actions = demo["actions"][()]
            actions[:, :7] *= signs
            demo["actions"][...] = actions

            states = demo["states"][()]
            states[:, 1 + robot_qpos_indexes] *= signs
            states[:, 1 + nq + robot_qvel_indexes] *= signs
            demo["states"][...] = states
            _transform_joint_observations(demo, signs)


def generate_unit_test(output_dir: str, num_trajectories: int, seed: int) -> None:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    dataset_dir = os.path.join(output_dir, "datasets")
    model_dir = os.path.join(output_dir, "robot_models")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    canonical_path = os.path.join(dataset_dir, "cfg0_joint_delta.hdf5")
    generate_demos(
        num_demos=num_trajectories,
        output_files=[os.path.basename(canonical_path)],
        action_spaces=["joint_delta"],
        seed=seed,
        task="liftrand",
        output_dir=dataset_dir,
        robot=CONFIG_SPECS["cfg0"]["robot"],
    )

    canonical_env = create_demo_env("liftrand", robot=CONFIG_SPECS["cfg0"]["robot"])
    canonical_env.reset()
    qpos_indexes = np.asarray(canonical_env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)
    qvel_indexes = np.asarray(canonical_env.robots[0]._ref_joint_vel_indexes, dtype=np.int64)
    nq = canonical_env.sim.model.nq
    canonical_env.close()

    manifest = {
        "task": "LiftRand",
        "action_space": "joint_delta",
        "num_trajectories_per_config": num_trajectories,
        "seed": seed,
        "pairing": "same canonical state trajectory transformed by q_cfg = S_cfg q_cfg0",
        "configs": {},
    }

    for config_id, spec in CONFIG_SPECS.items():
        robot_name = spec["robot"]
        signs = np.asarray(spec["joint_signs"], dtype=np.float64)
        dataset_path = os.path.join(dataset_dir, f"{config_id}_joint_delta.hdf5")
        _write_variant_dataset(
            canonical_path=canonical_path,
            variant_path=dataset_path,
            config_id=config_id,
            robot_name=robot_name,
            signs=signs,
            robot_qpos_indexes=qpos_indexes,
            robot_qvel_indexes=qvel_indexes,
            nq=nq,
        )

        env = create_demo_env("liftrand", robot=robot_name)
        env.reset()
        model_path = os.path.join(model_dir, f"{robot_name}.xml")
        env.robots[0].robot_model.save_model(model_path, pretty=True)
        env.close()

        manifest["configs"][config_id] = {
            **spec,
            "dataset": os.path.relpath(dataset_path, output_dir),
            "robot_model": os.path.relpath(model_path, output_dir),
        }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Generated paired unit-test data: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_trajectories", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_unit_test(args.output_dir, args.num_trajectories, args.seed)

