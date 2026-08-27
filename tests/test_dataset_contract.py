from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from action_jacobian.dataset import (
    JointFlipPairedDataset,
    JointFlipSource,
    PairedPhysicalBatchSampler,
)


SIGNS = {
    "cfg0": np.ones(7, dtype=np.float32),
    "flip0": np.asarray([-1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
}


def _write_cache(path: Path) -> None:
    with h5py.File(path, "w") as cache:
        cache.create_dataset("config_ids", data=np.asarray([b"cfg0", b"flip0"]))
        cache.create_dataset("joint_signs", data=np.stack(list(SIGNS.values())))
        cache.create_dataset("jacobian_channel_rms", data=np.ones(14, dtype=np.float32))
        demos = cache.create_group("demos")
        action_offset = 0
        for demo_name, length in (("demo_0", 3), ("demo_1", 2)):
            group = demos.create_group(demo_name)
            rgb = np.full((length, 16, 16, 3), 128 + action_offset, dtype=np.uint8)
            canonical = np.arange(
                action_offset * 8,
                (action_offset + length) * 8,
                dtype=np.float32,
            ).reshape(length, 8)
            canonical[:, :7] = canonical[:, :7] / 100.0 + 0.01
            canonical[:, 7] = 1.0
            actions = np.repeat(canonical[None], 2, axis=0)
            actions[1, :, :7] *= SIGNS["flip0"]
            jacobian = np.ones((length, 15, 16, 16), dtype=np.float32)
            jacobian[:, 14] = 1.0
            group.create_dataset("rgb", data=rgb)
            group.create_dataset("actions", data=actions)
            group.create_dataset("canonical_pixel_jacobian", data=jacobian)
            action_offset += length


class DatasetContractTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.temp_dir.name) / "paired.hdf5"
        _write_cache(self.cache_path)
        self.source = JointFlipSource(
            str(self.cache_path), expected_demos=2, expected_physical_steps=5
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def dataset(self, **kwargs):
        return JointFlipPairedDataset(
            self.source,
            SIGNS,
            chunk_size=4,
            split="all",
            **kwargs,
        )

    def test_flip_and_episode_padding(self):
        dataset = self.dataset(include_structural=True)
        sample = dataset[5]
        canonical = self.source.canonical_actions[2]
        expected = canonical.copy()
        expected[:7] *= SIGNS["flip0"]
        np.testing.assert_array_equal(sample["raw_actions"][0].numpy(), expected)
        np.testing.assert_array_equal(sample["is_pad"].numpy(), [False, True, True, True])
        self.assertEqual(tuple(sample["image"].shape), (3, 16, 16))
        self.assertGreaterEqual(float(sample["image"].min()), 0.0)
        self.assertLessEqual(float(sample["image"].max()), 1.0)

    def test_four_condition_inputs(self):
        none = self.dataset(include_structural=False)[0]
        sign = self.dataset(include_structural=False, include_sign_array=True)[0]
        global_token = self.dataset(
            include_structural=False, include_global_jacobian=True
        )[0]
        pixel = self.dataset(include_structural=True)[1]
        self.assertNotIn("pixel_jacobian", none)
        self.assertNotIn("global_sign", none)
        self.assertEqual(tuple(sign["global_sign"].shape), (7,))
        self.assertEqual(tuple(global_token["global_jacobian"].shape), (30,))
        self.assertEqual(tuple(pixel["pixel_jacobian"].shape), (15, 16, 16))
        torch.testing.assert_close(
            pixel["pixel_jacobian"][0], -torch.ones(16, 16)
        )
        torch.testing.assert_close(
            pixel["pixel_jacobian"][14], torch.ones(16, 16)
        )

    def test_sampler_keeps_paired_configurations(self):
        dataset = self.dataset(include_structural=False)
        sampler = PairedPhysicalBatchSampler(
            dataset, physical_batch_size=3, num_batches=2, seed=7
        )
        for batch in sampler:
            self.assertEqual(len(batch), 6)
            for start in range(0, len(batch), 2):
                physical = dataset.pairs[batch[start : start + 2], 0]
                np.testing.assert_array_equal(physical, [physical[0], physical[0]])


if __name__ == "__main__":
    unittest.main()
