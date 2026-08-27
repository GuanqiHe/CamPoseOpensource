#!/usr/bin/env python3
"""Hydra entrypoint for collection, cache building, and dataset validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
UPSTREAM_SCRIPTS = REPO_ROOT / "third_party" / "cam_pose" / "script_robosuite_demos"


def _command(stage: DictConfig) -> list[str]:
    command = [sys.executable, str(UPSTREAM_SCRIPTS / str(stage.script))]
    params = OmegaConf.to_container(stage.params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("stage.params must be a mapping")
    for key, value in params.items():
        if value is None or value is False:
            continue
        flag = f"--{key if stage.name == 'collect' else key.replace('_', '-')}"
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *(str(item) for item in value)])
        else:
            command.extend([flag, str(value)])
    return command


@hydra.main(version_base="1.3", config_path="configs", config_name="generate")
def main(cfg: DictConfig) -> None:
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_path, environment.get("PYTHONPATH")])
    )
    subprocess.run(
        _command(cfg.stage),
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
