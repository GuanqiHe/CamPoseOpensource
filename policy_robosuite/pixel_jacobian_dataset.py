"""In-memory exact-paired dataset for the identifiability experiment."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class PixelJacobianPairedDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        config_ids: list[str],
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.cache_path = cache_path
        self.chunk_size = chunk_size
        with h5py.File(cache_path, "r") as cache:
            all_config_ids = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in cache["config_ids"][()]
            ]
            unknown = sorted(set(config_ids) - set(all_config_ids))
            if unknown:
                raise ValueError(f"Unknown configurations: {unknown}")
            self.config_indexes = np.asarray(
                [all_config_ids.index(config_id) for config_id in config_ids],
                dtype=np.int64,
            )
            self.config_ids = list(config_ids)
            self.joint_signs = cache["joint_signs"][()].astype(np.float32)
            self.action_mean = cache["action_mean"][()].astype(np.float32)
            self.action_std = cache["action_std"][()].astype(np.float32)
            self.jacobian_rms = cache["jacobian_channel_rms"][()].astype(
                np.float32
            )
            rgb_parts = []
            jacobian_parts = []
            action_parts = []
            episode_end_parts = []
            offset = 0
            for demo_name in sorted(cache["demos"]):
                group = cache[f"demos/{demo_name}"]
                rgb = group["rgb"][()]
                jacobian = group["canonical_pixel_jacobian"][()]
                actions = group["actions"][()].transpose(1, 0, 2)
                if not (len(rgb) == len(jacobian) == len(actions)):
                    raise ValueError(f"Unpaired cache group: {demo_name}")
                rgb_parts.append(rgb)
                jacobian_parts.append(jacobian)
                action_parts.append(actions)
                episode_end_parts.append(
                    np.full(len(rgb), offset + len(rgb), dtype=np.int64)
                )
                offset += len(rgb)

        self.rgb = np.concatenate(rgb_parts, axis=0)
        self.canonical_jacobian = np.concatenate(jacobian_parts, axis=0)
        self.actions = np.concatenate(action_parts, axis=0)
        self.episode_end = np.concatenate(episode_end_parts, axis=0)
        self.num_physical_steps = len(self.rgb)
        self.num_configs = len(self.config_indexes)

    def __len__(self) -> int:
        return self.num_physical_steps * self.num_configs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        physical_index = index // self.num_configs
        local_config_index = index % self.num_configs
        config_index = int(self.config_indexes[local_config_index])
        episode_end = int(self.episode_end[physical_index])
        chunk_end = min(physical_index + self.chunk_size, episode_end)
        chunk_length = chunk_end - physical_index

        raw_actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
        raw_actions[:chunk_length] = self.actions[
            physical_index:chunk_end, config_index
        ]
        normalized_actions = (
            raw_actions - self.action_mean[None]
        ) / self.action_std[None]
        is_pad = np.arange(self.chunk_size) >= chunk_length

        rgb = self.rgb[physical_index].astype(np.float32) / 255.0
        rgb = (rgb - IMAGE_MEAN[None, None]) / IMAGE_STD[None, None]
        rgb = np.transpose(rgb, (2, 0, 1)).copy()

        jacobian = self.canonical_jacobian[physical_index].copy()
        signs = self.joint_signs[config_index]
        jacobian[:14] = (
            jacobian[:14].reshape(7, 2, 16, 16)
            * signs[:, None, None, None]
        ).reshape(14, 16, 16)
        jacobian[:14] /= self.jacobian_rms[:, None, None]

        return {
            "image": torch.from_numpy(rgb),
            "pixel_jacobian": torch.from_numpy(jacobian),
            "global_sign": torch.from_numpy(signs.copy()),
            "actions": torch.from_numpy(normalized_actions),
            "raw_actions": torch.from_numpy(raw_actions),
            "is_pad": torch.from_numpy(is_pad),
            "config_index": torch.tensor(config_index, dtype=torch.long),
            "physical_index": torch.tensor(physical_index, dtype=torch.long),
        }

    def normalized_l1_bayes_bound(self) -> float:
        total_error = 0.0
        denominator = 0
        selected = self.config_indexes
        for physical_index in range(self.num_physical_steps):
            episode_end = int(self.episode_end[physical_index])
            chunk_end = min(physical_index + self.chunk_size, episode_end)
            targets = self.actions[physical_index:chunk_end, selected]
            targets = targets.transpose(1, 0, 2)
            targets = (
                targets - self.action_mean[None, None]
            ) / self.action_std[None, None]
            median = np.median(targets, axis=0, keepdims=True)
            total_error += np.abs(targets - median).sum()
            denominator += targets.size
        return float(total_error / denominator)


class PairedPhysicalBatchSampler(Sampler[list[int]]):
    """Sample physical states, then include every selected configuration."""

    def __init__(
        self,
        dataset: PixelJacobianPairedDataset,
        physical_batch_size: int,
        num_batches: int,
        seed: int,
        start_batch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.physical_batch_size = physical_batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.start_batch = start_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.start_batch):
            rng.integers(
                0,
                self.dataset.num_physical_steps,
                size=self.physical_batch_size,
            )
        for _ in range(self.num_batches):
            physical_indexes = rng.integers(
                0,
                self.dataset.num_physical_steps,
                size=self.physical_batch_size,
            )
            yield [
                int(physical_index * self.dataset.num_configs + config_index)
                for physical_index in physical_indexes
                for config_index in range(self.dataset.num_configs)
            ]

    def __len__(self) -> int:
        return self.num_batches
