#!/usr/bin/env python3
"""Validate an official-Lift RGB cache and emit a cloud-side contact sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampled = []
    demo_rows = []
    frame_total = 0
    with h5py.File(args.dataset, "r") as source, h5py.File(args.rgb_cache, "r") as cache:
        cache_data = cache["data"]
        source_hash = sha256_file(args.dataset)
        if cache_data.attrs.get("source_sha256", "") != source_hash:
            raise RuntimeError("RGB cache source SHA256 does not match dataset")
        if not bool(cache_data.attrs.get("complete", False)):
            raise RuntimeError("RGB cache is not marked complete")
        demo_names = sorted(
            (name for name in source["data"] if name.startswith("demo_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if len(demo_names) != args.expected_demos:
            raise RuntimeError(f"Expected {args.expected_demos} demos, found {len(demo_names)}")
        for demo_index, demo_name in enumerate(demo_names):
            actions = source["data"][demo_name]["actions"]
            cache_demo = cache_data[demo_name]
            rgb = cache_demo["agentview_rgb"]
            if not bool(cache_demo.attrs.get("complete", False)) or len(rgb) != len(actions):
                raise RuntimeError(f"Incomplete or length-mismatched cache entry: {demo_name}")
            initial = rgb[0]
            frame_total += len(rgb)
            row = {
                "demo": demo_name,
                "frames": len(rgb),
                "initial_mean": float(initial.mean()),
                "initial_std": float(initial.std()),
            }
            if row["initial_std"] < 5.0:
                raise RuntimeError(f"Near-constant rendered image: {demo_name}")
            demo_rows.append(row)
            if demo_index < args.contact_sheet_demos:
                sampled.append((demo_name, initial))

    columns = 5
    tile = 256
    rows = (len(sampled) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (demo_name, frame) in enumerate(sampled):
        x = (index % columns) * tile
        y = (index // columns) * tile
        sheet.paste(Image.fromarray(frame), (x, y))
        draw.rectangle((x, y, x + 85, y + 16), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), demo_name, fill=(255, 255, 255))
    sheet.save(output / "initial_frames_contact_sheet.png")
    metrics = {
        "accepted": True,
        "num_demos": len(demo_rows),
        "num_frames": frame_total,
        "source_sha256": source_hash,
        "initial_frame_mean_range": [
            min(row["initial_mean"] for row in demo_rows),
            max(row["initial_mean"] for row in demo_rows),
        ],
        "initial_frame_std_range": [
            min(row["initial_std"] for row in demo_rows),
            max(row["initial_std"] for row in demo_rows),
        ],
        "demos": demo_rows,
    }
    with open(output / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps({key: value for key, value in metrics.items() if key != "demos"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-demos", type=int, default=200)
    parser.add_argument("--contact-sheet-demos", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
