# Pixel Action Jacobian

This repository tests whether a vision policy can generalize across robot
action conventions when it receives the local map from raw joint actions to
visible robot motion.

## Review surface

```text
generate_dataset.py              # collection, cache, manifest, validation
train.py                         # matched training entrypoint
eval.py                          # checkpoint-only offline and rollout evaluation
configs/                         # approved sign configurations and experiment spec
src/action_jacobian/
  representation.py             # 15x16x16 pixel action Jacobian
  dataset.py                    # paired physical-state dataset
  models/                       # DINOv3 + deterministic ACT
third_party/                    # upstream CamPose, robosuite, and ManiSkill code
```

The model conditions are `none`, `sign_array`, `global_token`, and
`pixel_jacobian`. They share the same RGB, action chunks, episode split,
optimizer, and evaluation code.

## Data

Run one stage through the single dispatcher:

```bash
python generate_dataset.py collect --task liftrand --num_demos 200 \
  --action_spaces joint_delta --successful_only --output_dir /path/to/raw

python generate_dataset.py build-cache \
  --dataset-dir /path/to/config_datasets --output /path/to/cache.hdf5

python generate_dataset.py build-manifest \
  --cache /path/to/cache.hdf5 --design configs/joint_sign_dr_v1.json \
  --output /path/to/manifest.json

python generate_dataset.py validate --help
```

## Train

Formal runs require W&B online tracking. Inspect `python train.py --help` for
the complete fixed-budget configuration. Training is never part of unit tests.

## Evaluate

```bash
python eval.py \
  --checkpoint /path/to/step.pth \
  --cache /path/to/cache.hdf5 \
  --dataset /path/to/raw.hdf5 \
  --design configs/joint_sign_dr_v1.json \
  --manifest /path/to/manifest.json \
  --dinov3-model-path /path/to/dinov3 \
  --config-id sign_ood_00 --output-dir /path/to/eval
```

The historical camera-conditioning implementation and simulator submodules are
kept under `third_party/` for provenance; they are outside the experiment
review surface.
