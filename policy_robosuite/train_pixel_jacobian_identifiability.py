"""Train deterministic ACT for the pixel-Jacobian identifiability stages."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.deterministic_act import CONDITION_MODES, DeterministicACT
from pixel_jacobian_dataset import (
    PairedPhysicalBatchSampler,
    PixelJacobianPairedDataset,
)


ARM_SIGN_THRESHOLD_RAD = 0.002


@dataclass
class EvaluationMetrics:
    normalized_action_mae: float
    normalized_l1_bayes_bound: float
    per_config_normalized_action_mae: dict[str, float]
    per_config_arm_sign_accuracy: dict[str, float]
    per_config_gripper_sign_accuracy: dict[str, float]
    per_joint_arm_sign_accuracy: dict[str, float | None]
    arm_sign_entry_count: int


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
) -> torch.Tensor:
    valid = (~is_pad).unsqueeze(-1).expand_as(target)
    return F.l1_loss(prediction[valid], target[valid])


@torch.no_grad()
def evaluate(
    model: DeterministicACT,
    dataset: PixelJacobianPairedDataset,
    device: torch.device,
    batch_size: int,
) -> EvaluationMetrics:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    absolute_error_sum = 0.0
    valid_value_count = 0
    config_error_sum = {index: 0.0 for index in dataset.config_indexes}
    config_value_count = {index: 0 for index in dataset.config_indexes}
    config_arm_correct = {index: 0 for index in dataset.config_indexes}
    config_arm_count = {index: 0 for index in dataset.config_indexes}
    config_gripper_correct = {index: 0 for index in dataset.config_indexes}
    config_gripper_count = {index: 0 for index in dataset.config_indexes}
    joint_correct = np.zeros(7, dtype=np.int64)
    joint_count = np.zeros(7, dtype=np.int64)

    action_mean = torch.from_numpy(dataset.action_mean).to(device)
    action_std = torch.from_numpy(dataset.action_std).to(device)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        jacobian = batch["pixel_jacobian"].to(device, non_blocking=True)
        signs = batch["global_sign"].to(device, non_blocking=True)
        target = batch["actions"].to(device, non_blocking=True)
        raw_target = batch["raw_actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        config_indexes = batch["config_index"].to(device)
        prediction = model(image, jacobian, signs)

        valid = (~is_pad).unsqueeze(-1).expand_as(target)
        error = torch.abs(prediction - target)
        absolute_error_sum += float(error[valid].sum())
        valid_value_count += int(valid.sum())
        raw_prediction = prediction * action_std[None, None] + action_mean[
            None, None
        ]

        for config_index in config_error_sum:
            sample_mask = config_indexes == int(config_index)
            if not bool(sample_mask.any()):
                continue
            config_valid = valid[sample_mask]
            config_error_sum[config_index] += float(
                error[sample_mask][config_valid].sum()
            )
            config_value_count[config_index] += int(config_valid.sum())

            config_pad = is_pad[sample_mask]
            config_raw_target = raw_target[sample_mask]
            config_raw_prediction = raw_prediction[sample_mask]
            arm_valid = (
                (~config_pad).unsqueeze(-1)
                & (torch.abs(config_raw_target[..., :7]) >= ARM_SIGN_THRESHOLD_RAD)
            )
            arm_correct = (
                torch.sign(config_raw_prediction[..., :7])
                == torch.sign(config_raw_target[..., :7])
            ) & arm_valid
            config_arm_correct[config_index] += int(arm_correct.sum())
            config_arm_count[config_index] += int(arm_valid.sum())
            joint_correct += arm_correct.sum(dim=(0, 1)).cpu().numpy()
            joint_count += arm_valid.sum(dim=(0, 1)).cpu().numpy()

            gripper_valid = (~config_pad) & (
                torch.abs(config_raw_target[..., 7]) >= 0.5
            )
            gripper_correct = (
                torch.sign(config_raw_prediction[..., 7])
                == torch.sign(config_raw_target[..., 7])
            ) & gripper_valid
            config_gripper_correct[config_index] += int(gripper_correct.sum())
            config_gripper_count[config_index] += int(gripper_valid.sum())

    id_by_index = {
        int(index): config_id
        for index, config_id in zip(dataset.config_indexes, dataset.config_ids)
    }
    return EvaluationMetrics(
        normalized_action_mae=absolute_error_sum / valid_value_count,
        normalized_l1_bayes_bound=dataset.normalized_l1_bayes_bound(),
        per_config_normalized_action_mae={
            id_by_index[int(index)]: config_error_sum[index]
            / config_value_count[index]
            for index in config_error_sum
        },
        per_config_arm_sign_accuracy={
            id_by_index[int(index)]: config_arm_correct[index]
            / max(config_arm_count[index], 1)
            for index in config_arm_correct
        },
        per_config_gripper_sign_accuracy={
            id_by_index[int(index)]: config_gripper_correct[index]
            / max(config_gripper_count[index], 1)
            for index in config_gripper_correct
        },
        per_joint_arm_sign_accuracy={
            f"joint_{index + 1}": (
                float(joint_correct[index] / joint_count[index])
                if joint_count[index]
                else None
            )
            for index in range(7)
        },
        arm_sign_entry_count=int(joint_count.sum()),
    )


def _flatten_metrics(metrics: EvaluationMetrics) -> dict[str, float]:
    output = {
        "eval/normalized_action_mae": metrics.normalized_action_mae,
        "eval/normalized_l1_bayes_bound": metrics.normalized_l1_bayes_bound,
        "eval/arm_sign_entry_count": metrics.arm_sign_entry_count,
    }
    for name, values in (
        ("config_mae", metrics.per_config_normalized_action_mae),
        ("config_arm_sign", metrics.per_config_arm_sign_accuracy),
        ("config_gripper_sign", metrics.per_config_gripper_sign_accuracy),
        ("joint_arm_sign", metrics.per_joint_arm_sign_accuracy),
    ):
        for key, value in values.items():
            if value is not None:
                output[f"eval/{name}/{key}"] = value
    return output


def train(args: argparse.Namespace) -> dict:
    import wandb

    _seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = PixelJacobianPairedDataset(
        args.cache, args.configs, args.chunk_size
    )
    model = DeterministicACT(
        condition_mode=args.condition,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        ffn_dim=args.ffn_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        imagenet=args.imagenet,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    resume_checkpoint = None
    start_step = 0
    samples_seen = 0
    previous_elapsed_seconds = 0.0
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        resume_config = resume_checkpoint["config"]
        immutable_keys = (
            "condition",
            "configs",
            "chunk_size",
            "hidden_dim",
            "nheads",
            "ffn_dim",
            "enc_layers",
            "dec_layers",
            "dropout",
            "imagenet",
        )
        mismatches = {
            key: (resume_config.get(key), getattr(args, key))
            for key in immutable_keys
            if resume_config.get(key) != getattr(args, key)
        }
        if mismatches:
            raise ValueError(f"Resume configuration mismatch: {mismatches}")
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        samples_seen = int(resume_checkpoint["samples_seen"])
        start_step = int(
            resume_checkpoint.get(
                "step",
                samples_seen
                // (args.physical_batch_size * dataset.num_configs),
            )
        )
        previous_elapsed_seconds = float(
            resume_checkpoint.get("metrics", {}).get("elapsed_seconds", 0.0)
        )
        if start_step >= args.steps:
            raise ValueError(
                f"Checkpoint step {start_step} is not below target {args.steps}"
            )

    sampler = PairedPhysicalBatchSampler(
        dataset,
        args.physical_batch_size,
        args.steps - start_step,
        args.seed,
        start_batch=start_step,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "git_commit": _git_commit(),
            "parameter_count": parameter_count,
            "num_physical_steps": dataset.num_physical_steps,
            "num_paired_samples": len(dataset),
            "normalized_l1_bayes_bound": dataset.normalized_l1_bayes_bound(),
        }
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as output:
        json.dump(run_config, output, indent=2, sort_keys=True)

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        config=run_config,
    )
    history_path = os.path.join(args.output_dir, "history.jsonl")
    start_time = time.perf_counter()
    last_metrics = None

    model.train()
    for step, batch in enumerate(loader, start=start_step + 1):
        image = batch["image"].to(device, non_blocking=True)
        jacobian = batch["pixel_jacobian"].to(device, non_blocking=True)
        signs = batch["global_sign"].to(device, non_blocking=True)
        target = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.bf16 and device.type == "cuda",
        ):
            prediction = model(image, jacobian, signs)
            loss = _masked_l1(prediction, target, is_pad)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()
        samples_seen += len(image)

        log_record = {
            "step": step,
            "samples_seen": samples_seen,
            "train/l1": float(loss.detach()),
            "train/gradient_norm": float(gradient_norm),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
        }
        should_evaluate = step == 1 or step % args.eval_every == 0 or step == args.steps
        if should_evaluate:
            last_metrics = evaluate(
                model, dataset, device, args.eval_batch_size
            )
            log_record.update(_flatten_metrics(last_metrics))
            model.train()
        wandb.log(log_record, step=step)
        with open(history_path, "a", encoding="utf-8") as history:
            history.write(json.dumps(log_record, sort_keys=True) + "\n")

    elapsed_seconds = (
        previous_elapsed_seconds + time.perf_counter() - start_time
    )
    if last_metrics is None:
        last_metrics = evaluate(model, dataset, device, args.eval_batch_size)
    final_metrics = asdict(last_metrics)
    final_metrics.update(
        {
            "samples_seen": samples_seen,
            "elapsed_seconds": elapsed_seconds,
            "samples_per_second": samples_seen / elapsed_seconds,
            "parameter_count": parameter_count,
            "wandb_run_id": wandb_run.id,
            "wandb_run_url": wandb_run.url,
            "git_commit": run_config["git_commit"],
        }
    )
    with open(
        os.path.join(args.output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(final_metrics, output, indent=2, sort_keys=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": run_config,
            "metrics": final_metrics,
            "samples_seen": samples_seen,
            "step": args.steps,
            "wandb_run_id": wandb_run.id,
        },
        os.path.join(args.output_dir, "checkpoint_final.pt"),
    )
    wandb.finish()
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--condition", choices=CONDITION_MODES, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="pixel-action-jacobian")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
