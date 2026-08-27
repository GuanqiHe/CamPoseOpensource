"""In-memory exact-paired dataset for the identifiability experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
        include_structural: bool = True,
        normalize_rgb: bool = True,
        expected_demos: int | None = None,
        expected_physical_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.cache_path = cache_path
        self.chunk_size = chunk_size
        self.include_structural = include_structural
        self.normalize_rgb = normalize_rgb
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
            validation_parts = []
            offset = 0
            demo_names = sorted(cache["demos"])
            if expected_demos is not None and len(demo_names) != expected_demos:
                raise ValueError(
                    f"Expected {expected_demos} demos, found {len(demo_names)}"
                )
            for demo_name in demo_names:
                group = cache[f"demos/{demo_name}"]
                rgb = group["rgb"][()]
                actions = group["actions"][()].transpose(1, 0, 2)
                if len(rgb) != len(actions):
                    raise ValueError(f"Unpaired cache group: {demo_name}")
                rgb_parts.append(rgb)
                if include_structural:
                    jacobian = group["canonical_pixel_jacobian"][()]
                    if len(jacobian) != len(rgb):
                        raise ValueError(f"Unpaired Jacobian group: {demo_name}")
                    jacobian_parts.append(jacobian)
                action_parts.append(actions)
                episode_end_parts.append(
                    np.full(len(rgb), offset + len(rgb), dtype=np.int64)
                )
                validation_parts.append(np.arange(len(rgb)) % 10 == 0)
                offset += len(rgb)

        self.rgb = np.concatenate(rgb_parts, axis=0)
        self.canonical_jacobian = (
            np.concatenate(jacobian_parts, axis=0)
            if include_structural
            else None
        )
        self.actions = np.concatenate(action_parts, axis=0)
        self.episode_end = np.concatenate(episode_end_parts, axis=0)
        self.validation_mask = np.concatenate(validation_parts, axis=0)
        self.num_physical_steps = len(self.rgb)
        self.num_configs = len(self.config_indexes)
        if (
            expected_physical_steps is not None
            and self.num_physical_steps != expected_physical_steps
        ):
            raise ValueError(
                f"Expected {expected_physical_steps} physical steps, "
                f"found {self.num_physical_steps}"
            )

    def __len__(self) -> int:
        return self.num_physical_steps * self.num_configs

    def split_indices(self, split: str) -> list[int]:
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")
        selected = ~self.validation_mask if split == "train" else self.validation_mask
        return [
            physical_index * self.num_configs + config_offset
            for physical_index in np.flatnonzero(selected)
            for config_offset in range(self.num_configs)
        ]

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
        if self.normalize_rgb:
            rgb = (rgb - IMAGE_MEAN[None, None]) / IMAGE_STD[None, None]
        rgb = np.transpose(rgb, (2, 0, 1)).copy()
        result = {
            "image": torch.from_numpy(rgb),
            "actions": torch.from_numpy(normalized_actions),
            "raw_actions": torch.from_numpy(raw_actions),
            "is_pad": torch.from_numpy(is_pad),
            "config_index": torch.tensor(config_index, dtype=torch.long),
            "physical_index": torch.tensor(physical_index, dtype=torch.long),
        }
        if self.include_structural:
            jacobian = self.canonical_jacobian[physical_index].copy()
            signs = self.joint_signs[config_index]
            jacobian[:14] = (
                jacobian[:14].reshape(7, 2, 16, 16)
                * signs[:, None, None, None]
            ).reshape(14, 16, 16)
            jacobian[:14] /= self.jacobian_rms[:, None, None]
            result["pixel_jacobian"] = torch.from_numpy(jacobian)
            result["global_sign"] = torch.from_numpy(signs.copy())
        return result

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


class JointFlipSource:
    """Canonical RGB, action, and pixel-Jacobian arrays shared by all views."""

    def __init__(
        self,
        cache_path: str,
        canonical_config_id: str = "cfg0",
        expected_demos: int | None = None,
        expected_physical_steps: int | None = None,
    ) -> None:
        self.cache_path = cache_path
        with h5py.File(cache_path, "r") as cache:
            cache_config_ids = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in cache["config_ids"][()]
            ]
            if canonical_config_id not in cache_config_ids:
                raise ValueError(
                    f"Canonical configuration {canonical_config_id!r} not in cache"
                )
            canonical_index = cache_config_ids.index(canonical_config_id)
            cache_signs = cache["joint_signs"][()].astype(np.float32)
            self.cache_config_signs = {
                config_id: cache_signs[index].copy()
                for index, config_id in enumerate(cache_config_ids)
            }
            self.jacobian_rms = cache["jacobian_channel_rms"][()].astype(
                np.float32
            )

            rgb_parts = []
            jacobian_parts = []
            action_parts = []
            episode_end_parts = []
            validation_parts = []
            offset = 0
            demo_names = sorted(cache["demos"])
            if expected_demos is not None and len(demo_names) != expected_demos:
                raise ValueError(
                    f"Expected {expected_demos} demos, found {len(demo_names)}"
                )
            for demo_name in demo_names:
                group = cache[f"demos/{demo_name}"]
                rgb = group["rgb"][()]
                jacobian = group["canonical_pixel_jacobian"][()]
                cached_actions = group["actions"][()].astype(np.float32)
                canonical_actions = cached_actions[canonical_index]
                if not (
                    len(rgb) == len(jacobian) == len(canonical_actions)
                ):
                    raise ValueError(f"Unpaired cache group: {demo_name}")

                expected_actions = np.repeat(
                    canonical_actions[None], len(cache_config_ids), axis=0
                )
                expected_actions[..., :7] *= cache_signs[:, None, :]
                if not np.array_equal(expected_actions, cached_actions):
                    max_error = float(
                        np.max(np.abs(expected_actions - cached_actions))
                    )
                    raise ValueError(
                        f"Stored action convention mismatch in {demo_name}: "
                        f"max_abs={max_error}"
                    )

                rgb_parts.append(rgb)
                jacobian_parts.append(jacobian)
                action_parts.append(canonical_actions)
                episode_end_parts.append(
                    np.full(len(rgb), offset + len(rgb), dtype=np.int64)
                )
                validation_parts.append(np.arange(len(rgb)) % 10 == 0)
                offset += len(rgb)

        self.rgb = np.concatenate(rgb_parts, axis=0)
        self.canonical_jacobian = np.concatenate(jacobian_parts, axis=0)
        self.canonical_actions = np.concatenate(action_parts, axis=0)
        self.episode_end = np.concatenate(episode_end_parts, axis=0)
        self.validation_mask = np.concatenate(validation_parts, axis=0)
        self.num_physical_steps = len(self.rgb)
        if (
            expected_physical_steps is not None
            and self.num_physical_steps != expected_physical_steps
        ):
            raise ValueError(
                f"Expected {expected_physical_steps} physical steps, "
                f"found {self.num_physical_steps}"
            )


class JointFlipPairedDataset(Dataset):
    """Derive raw-action conventions from one canonical physical trajectory."""

    def __init__(
        self,
        source: JointFlipSource,
        config_signs: Mapping[str, Sequence[int]],
        chunk_size: int,
        split: str,
        include_structural: bool = True,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "all"):
            raise ValueError(f"Unknown split: {split}")
        if not config_signs:
            raise ValueError("At least one configuration is required")
        self.source = source
        self.chunk_size = chunk_size
        self.split = split
        self.include_structural = include_structural
        self.config_ids = list(config_signs)
        self.joint_signs = np.asarray(
            [config_signs[config_id] for config_id in self.config_ids],
            dtype=np.float32,
        )
        if self.joint_signs.shape != (len(self.config_ids), 7):
            raise ValueError("Each joint-sign configuration must have shape (7,)")
        if not np.all(np.isin(self.joint_signs, (-1.0, 1.0))):
            raise ValueError("Joint signs must contain only -1 or +1")

        if split == "train":
            selected = ~source.validation_mask
        elif split == "val":
            selected = source.validation_mask
        else:
            selected = np.ones(source.num_physical_steps, dtype=bool)
        self.physical_indexes = np.flatnonzero(selected)
        self.num_physical_steps = len(self.physical_indexes)
        self.num_configs = len(self.config_ids)

        convention_actions = np.repeat(
            source.canonical_actions[:, None, :], self.num_configs, axis=1
        )
        convention_actions[..., :7] *= self.joint_signs[None]
        flattened_actions = convention_actions.reshape(-1, 8)
        computed_mean = flattened_actions.mean(axis=0).astype(np.float32)
        computed_std = flattened_actions.std(axis=0).astype(np.float32)
        computed_std = np.maximum(computed_std, 1e-6)
        computed_min = flattened_actions.min(axis=0).astype(np.float32)
        computed_max = flattened_actions.max(axis=0).astype(np.float32)

        supplied = (action_mean, action_std, action_min, action_max)
        if any(value is None for value in supplied) and not all(
            value is None for value in supplied
        ):
            raise ValueError("Action statistics must be supplied together")
        if action_mean is None:
            self.action_mean = computed_mean
            self.action_std = computed_std
            self.action_min = computed_min
            self.action_max = computed_max
        else:
            self.action_mean = np.asarray(action_mean, dtype=np.float32)
            self.action_std = np.asarray(action_std, dtype=np.float32)
            self.action_min = np.asarray(action_min, dtype=np.float32)
            self.action_max = np.asarray(action_max, dtype=np.float32)
            for name, value in (
                ("action_mean", self.action_mean),
                ("action_std", self.action_std),
                ("action_min", self.action_min),
                ("action_max", self.action_max),
            ):
                if value.shape != (8,):
                    raise ValueError(f"{name} must have shape (8,)")

    def __len__(self) -> int:
        return self.num_physical_steps * self.num_configs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        physical_offset = index // self.num_configs
        config_index = index % self.num_configs
        physical_index = int(self.physical_indexes[physical_offset])
        signs = self.joint_signs[config_index]
        episode_end = int(self.source.episode_end[physical_index])
        chunk_end = min(physical_index + self.chunk_size, episode_end)
        chunk_length = chunk_end - physical_index

        raw_actions = np.zeros((self.chunk_size, 8), dtype=np.float32)
        raw_actions[:chunk_length] = self.source.canonical_actions[
            physical_index:chunk_end
        ]
        raw_actions[:chunk_length, :7] *= signs[None]
        normalized_actions = (
            raw_actions - self.action_mean[None]
        ) / self.action_std[None]
        is_pad = np.arange(self.chunk_size) >= chunk_length

        rgb = self.source.rgb[physical_index].astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1)).copy()
        result = {
            "image": torch.from_numpy(rgb),
            "actions": torch.from_numpy(normalized_actions),
            "raw_actions": torch.from_numpy(raw_actions),
            "is_pad": torch.from_numpy(is_pad),
            "config_index": torch.tensor(config_index, dtype=torch.long),
            "physical_index": torch.tensor(physical_index, dtype=torch.long),
        }
        if self.include_structural:
            jacobian = self.source.canonical_jacobian[physical_index].copy()
            jacobian[:14] = (
                jacobian[:14].reshape(7, 2, 16, 16)
                * signs[:, None, None, None]
            ).reshape(14, 16, 16)
            jacobian[:14] /= self.source.jacobian_rms[:, None, None]
            result["pixel_jacobian"] = torch.from_numpy(jacobian)
        return result

    def normalized_l1_bayes_bound(self) -> float:
        total_error = 0.0
        denominator = 0
        for physical_index in self.physical_indexes:
            episode_end = int(self.source.episode_end[physical_index])
            chunk_end = min(physical_index + self.chunk_size, episode_end)
            canonical = self.source.canonical_actions[
                physical_index:chunk_end
            ]
            targets = np.repeat(
                canonical[None], self.num_configs, axis=0
            )
            targets[..., :7] *= self.joint_signs[:, None]
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
        dataset: PixelJacobianPairedDataset | JointFlipPairedDataset,
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
