#!/usr/bin/env python3
"""Single entrypoint for collection, cache building, and dataset validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
UPSTREAM_SCRIPTS = REPO_ROOT / "third_party" / "cam_pose" / "script_robosuite_demos"
STAGES = {
    "collect": "gen_robosuite_format_demo.py",
    "build-cache": "build_pixel_jacobian_cache.py",
    "build-manifest": "build_sign_dr_manifest.py",
    "validate": "run_sign_dr_unit_test.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one reproducible dataset-pipeline stage."
    )
    parser.add_argument("stage", choices=STAGES)
    args, forwarded = parser.parse_known_args()
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_path, environment.get("PYTHONPATH")])
    )
    subprocess.run(
        [sys.executable, str(UPSTREAM_SCRIPTS / STAGES[args.stage]), *forwarded],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
