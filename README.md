# Pixel Action Jacobian

This repository tests whether a vision policy can generalize across robot
action conventions when it receives the local map from raw joint actions to
visible robot motion.

## Review surface

```text
generate_dataset.py              # Hydra data-pipeline entrypoint
train.py                         # Hydra matched-training entrypoint
eval.py                          # Hydra checkpoint evaluation entrypoint
configs/                         # data stages, methods, train/eval parameters
src/action_jacobian/
  representation.py             # 15x16x16 pixel action Jacobian
  dataset.py                    # paired physical-state dataset
  models/                       # DINOv3 + deterministic ACT
third_party/                    # upstream CamPose, robosuite, and ManiSkill code
```

The model conditions are `none`, `sign_array`, `global_token`, and
`pixel_jacobian`. They share the same RGB, action chunks, episode split,
optimizer, and evaluation code.

## Setup

Clone with submodules, then create the Python environment from any working
directory:

```bash
git clone --recursive <repository-url>
cd CamPoseOpensource
bash third_party/cam_pose/setup.sh
```

The setup script installs the simulator submodules from `third_party/` and all
runtime dependencies required by the top-level data, training, and evaluation
entrypoints.

## Data

Select one Hydra stage and override only its paths:

```bash
python generate_dataset.py stage=collect stage.params.output_dir=/path/to/raw

python generate_dataset.py stage=build_cache \
  stage.params.dataset_dir=/path/to/config_datasets \
  stage.params.output=/path/to/cache.hdf5

python generate_dataset.py stage=build_manifest \
  stage.params.cache=/path/to/cache.hdf5 \
  stage.params.output=/path/to/manifest.json

python generate_dataset.py stage=validate --cfg job
```

## Train

The four matched methods are selected with `method=none`, `method=sign_array`,
`method=global_token`, or `method=pixel_jacobian`. Inspect the resolved config
without launching training:

```bash
python train.py method=pixel_jacobian --cfg job
```

Formal runs require W&B online tracking. Training is never part of unit tests.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python generate_dataset.py stage=validate \
  stage.params.canonical_dataset=/path/to/canonical.hdf5 \
  stage.params.output_dir=/tmp/jacobian-unit-test
```

## Evaluate

```bash
python eval.py \
  paths.checkpoint=/path/to/step.pth \
  paths.cache=/path/to/cache.hdf5 \
  paths.dataset=/path/to/raw.hdf5 \
  paths.manifest=/path/to/manifest.json \
  paths.dinov3_model=/path/to/dinov3 \
  config_id=sign_ood_00 paths.output_dir=/path/to/eval
```

The historical camera-conditioning implementation and simulator submodules are
kept under `third_party/` for provenance; they are outside the experiment
review surface.
