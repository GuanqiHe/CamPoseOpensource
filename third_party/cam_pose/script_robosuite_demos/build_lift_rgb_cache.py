#!/usr/bin/env python3
"""Render a deterministic agentview RGB cache for a robosuite dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_env(dataset_path: str, camera_name: str, height: int, width: int):
    with h5py.File(dataset_path, "r") as source:
        env_args = json.loads(source["data"].attrs["env_args"])
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_heights=height,
        camera_widths=width,
        camera_names=[camera_name],
    )
    return suite.make(env_name=env_args["env_name"], **env_kwargs)


def render_cache(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    env = build_env(args.dataset, args.camera_name, args.height, args.width)
    started = time.time()
    rendered = 0

    try:
        with h5py.File(args.dataset, "r") as source, h5py.File(output, mode) as cache:
            source_data = source["data"]
            cache_data = cache.require_group("data")
            cache_data.attrs.update(
                source_dataset=os.path.abspath(args.dataset),
                source_sha256=sha256_file(args.dataset),
                camera_name=args.camera_name,
                height=args.height,
                width=args.width,
                vertical_flip=True,
                renderer="robosuite.sim.render",
            )
            cache_data.attrs["complete"] = False
            demo_names = sorted(
                (name for name in source_data if name.startswith("demo_")),
                key=lambda name: int(name.split("_")[-1]),
            )
            for demo_index, demo_name in enumerate(demo_names):
                states = source_data[demo_name]["states"]
                demo_group = cache_data.require_group(demo_name)
                if "agentview_rgb" in demo_group:
                    existing = demo_group["agentview_rgb"]
                    if (
                        existing.shape == (len(states), args.height, args.width, 3)
                        and bool(demo_group.attrs.get("complete", False))
                    ):
                        print(f"[{demo_index + 1}/{len(demo_names)}] {demo_name}: cached")
                        continue
                    del demo_group["agentview_rgb"]
                demo_group.attrs["complete"] = False
                rgb = demo_group.create_dataset(
                    "agentview_rgb",
                    shape=(len(states), args.height, args.width, 3),
                    dtype=np.uint8,
                    chunks=(1, args.height, args.width, 3),
                    compression="lzf",
                )
                env.reset()
                for frame_index, state in enumerate(states):
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    image = env.sim.render(
                        camera_name=args.camera_name,
                        height=args.height,
                        width=args.width,
                        depth=False,
                    )
                    rgb[frame_index] = np.flipud(image).copy()
                    rendered += 1
                demo_group.attrs["num_frames"] = len(states)
                demo_group.attrs["complete"] = True
                cache.flush()
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[{demo_index + 1}/{len(demo_names)}] {demo_name}: "
                    f"{len(states)} frames, total={rendered}, fps={rendered / elapsed:.1f}"
                )
            cache_data.attrs["complete"] = True
            cache_data.attrs["num_demos"] = len(demo_names)
            cache_data.attrs["rendered_frames_this_run"] = rendered
            cache_data.attrs["elapsed_seconds_this_run"] = time.time() - started
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    render_cache(parse_args())
