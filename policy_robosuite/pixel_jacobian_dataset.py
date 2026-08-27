"""In-memory exact-paired dataset for the identifiability experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
GLOBAL_JACOBIAN_DIM = 30


def global_jacobian_descriptor(field: np.ndarray) -> np.ndarray:
    """Pool a normalized ``[15, H, W]`` Jacobian map without pixel locations."""

    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] != 15:
        raise ValueError(f"Expected [15,H,W] Jacobian field, got {field.shape}")
    mean = field.mean(axis=(1, 2))
    rms = np.sqrt(np.mean(np.square(field), axis=(1, 2)))
    return np.concatenate([mean, rms]).astype(np.float32)


def build_joint_flip_manifest(
    source: "JointFlipSource",
    train_config_ids: Sequence[str],
    val_demos: int = 40,
    split_seed: int = 20260827,
    sampling_seed: int = 20260827,
    configs_per_frame: int = 2,
) -> dict:
    """Create one reproducible episode split and config-pair sampling map."""

    train_config_ids = list(train_config_ids)
    if len(set(train_config_ids)) != len(train_config_ids):
        raise ValueError("train_config_ids must be unique")
    unknown = sorted(set(train_config_ids) - set(source.cache_config_signs))
    if unknown:
        raise ValueError(f"Unknown training configurations: {unknown}")
    if not 1 <= val_demos < source.num_demos:
        raise ValueError(
            f"val_demos must be in [1,{source.num_demos - 1}], got {val_demos}"
        )
    if not 2 <= configs_per_frame <= len(train_config_ids):
        raise ValueError(
            "configs_per_frame must be at least 2 and no larger than the "
            f"number of training configurations ({len(train_config_ids)})"
        )

    split_rng = np.random.default_rng(split_seed)
    val_demo_indexes = np.sort(
        split_rng.choice(source.num_demos, size=val_demos, replace=False)
    )
    val_demo_set = set(int(index) for index in val_demo_indexes)
    val_demo_names = [
        source.demo_names[index]
        for index in val_demo_indexes
    ]
    train_demo_names = [
        name
        for index, name in enumerate(source.demo_names)
        if index not in val_demo_set
    ]

    sampling_rng = np.random.default_rng(sampling_seed)
    config_rank = {config_id: index for index, config_id in enumerate(train_config_ids)}
    sampled_config_ids_by_physical = []
    for _ in range(source.num_physical_steps):
        sampled = sampling_rng.choice(
            train_config_ids, size=configs_per_frame, replace=False
        )
        sampled = sorted(sampled.tolist(), key=config_rank.__getitem__)
        sampled_config_ids_by_physical.append(sampled)

    return {
        "version": 1,
        "split_seed": int(split_seed),
        "sampling_seed": int(sampling_seed),
        "val_demos": int(val_demos),
        "configs_per_frame": int(configs_per_frame),
        "train_config_ids": train_config_ids,
        "train_demo_names": train_demo_names,
        "val_demo_names": val_demo_names,
        "sampled_config_ids_by_physical": sampled_config_ids_by_physical,
    }


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
            demo_index_parts = []
            offset = 0
            demo_names = sorted(cache["demos"])
            if expected_demos is not None and len(demo_names) != expected_demos:
                raise ValueError(
                    f"Expected {expected_demos} demos, found {len(demo_names)}"
                )
            for demo_index, demo_name in enumerate(demo_names):
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
                demo_index_parts.append(
                    np.full(len(rgb), demo_index, dtype=np.int64)
                )
                offset += len(rgb)

        self.rgb = np.concatenate(rgb_parts, axis=0)
        self.canonical_jacobian = np.concatenate(jacobian_parts, axis=0)
        self.canonical_actions = np.concatenate(action_parts, axis=0)
        self.episode_end = np.concatenate(episode_end_parts, axis=0)
        self.validation_mask = np.concatenate(validation_parts, axis=0)
        self.demo_names = demo_names
        self.demo_index_by_physical = np.concatenate(demo_index_parts, axis=0)
        self.num_demos = len(self.demo_names)
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
        include_global_jacobian: bool = False,
        include_sign_array: bool = False,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        physical_indexes: Sequence[int] | None = None,
        sampled_config_ids_by_physical: Sequence[Sequence[str]] | None = None,
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
        self.include_global_jacobian = include_global_jacobian
        self.include_sign_array = include_sign_array
        self.config_ids = list(config_signs)
        self.joint_signs = np.asarray(
            [config_signs[config_id] for config_id in self.config_ids],
            dtype=np.float32,
        )
        if self.joint_signs.shape != (len(self.config_ids), 7):
            raise ValueError("Each joint-sign configuration must have shape (7,)")
        if not np.all(np.isin(self.joint_signs, (-1.0, 1.0))):
            raise ValueError("Joint signs must contain only -1 or +1")

        if physical_indexes is None:
            if split == "train":
                selected = ~source.validation_mask
            elif split == "val":
                selected = source.validation_mask
            else:
                selected = np.ones(source.num_physical_steps, dtype=bool)
            self.physical_indexes = np.flatnonzero(selected)
        else:
            self.physical_indexes = np.asarray(physical_indexes, dtype=np.int64)
            if np.any(
                (self.physical_indexes < 0)
                | (self.physical_indexes >= source.num_physical_steps)
            ):
                raise ValueError("physical_indexes contains an out-of-range index")
        self.num_physical_steps = len(self.physical_indexes)
        self.config_id_to_index = {
            config_id: index for index, config_id in enumerate(self.config_ids)
        }

        self.sampled_config_ids_by_physical = {}
        self.pair_indices_by_physical = {}
        pair_parts = []
        for physical_index in self.physical_indexes:
            physical_index = int(physical_index)
            if sampled_config_ids_by_physical is None:
                sampled_config_ids = list(self.config_ids)
            else:
                sampled_config_ids = list(
                    sampled_config_ids_by_physical[physical_index]
                )
            if not sampled_config_ids:
                raise ValueError(
                    f"No configurations selected for physical index {physical_index}"
                )
            if len(set(sampled_config_ids)) != len(sampled_config_ids):
                raise ValueError(
                    f"Duplicate sampled configurations at physical index "
                    f"{physical_index}: {sampled_config_ids}"
                )
            unknown = sorted(set(sampled_config_ids) - set(self.config_ids))
            if unknown:
                raise ValueError(
                    f"Unknown sampled configurations at physical index "
                    f"{physical_index}: {unknown}"
                )
            self.sampled_config_ids_by_physical[physical_index] = sampled_config_ids
            pair_indices = []
            for config_id in sampled_config_ids:
                pair_indices.append(len(pair_parts))
                pair_parts.append(
                    (physical_index, self.config_id_to_index[config_id])
                )
            self.pair_indices_by_physical[physical_index] = pair_indices
        self.pairs = np.asarray(pair_parts, dtype=np.int64)

        flattened_actions = source.canonical_actions[
            self.pairs[:, 0]
        ].copy()
        flattened_actions[:, :7] *= self.joint_signs[self.pairs[:, 1]]
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
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        physical_index, config_index = self.pairs[index]
        physical_index = int(physical_index)
        config_index = int(config_index)
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
        if self.include_structural or self.include_global_jacobian:
            jacobian = self.source.canonical_jacobian[physical_index].copy()
            jacobian[:14] = (
                jacobian[:14].reshape(7, 2, 16, 16)
                * signs[:, None, None, None]
            ).reshape(14, 16, 16)
            jacobian[:14] /= self.source.jacobian_rms[:, None, None]
            if self.include_structural:
                result["pixel_jacobian"] = torch.from_numpy(jacobian)
            if self.include_global_jacobian:
                result["global_jacobian"] = torch.from_numpy(
                    global_jacobian_descriptor(jacobian)
                )
        if self.include_sign_array:
            result["global_sign"] = torch.from_numpy(signs.copy())
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
            sampled_config_ids = self.sampled_config_ids_by_physical[
                int(physical_index)
            ]
            targets = np.repeat(
                canonical[None], len(sampled_config_ids), axis=0
            )
            config_indexes = np.asarray(
                [self.config_id_to_index[config_id] for config_id in sampled_config_ids],
                dtype=np.int64,
            )
            targets[..., :7] *= self.joint_signs[config_indexes][:, None, :]
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
                pair_index
                for physical_index in physical_indexes
                for pair_index in self.dataset.pair_indices_by_physical[
                    int(self.dataset.physical_indexes[physical_index])
                ]
            ]

    def __len__(self) -> int:
        return self.num_batches
