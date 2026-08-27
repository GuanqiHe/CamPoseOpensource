from __future__ import annotations

import unittest

import numpy as np

from action_jacobian.simple_servo import (
    OOD_SIGNS,
    TRAIN_SIGNS,
    eef_pixel_jacobian,
    keypoints,
    make_sample,
    oracle_action,
    sample_state,
    world_to_pixel,
)


class SimpleServoTest(unittest.TestCase):
    def test_action_convention_preserves_physical_step(self):
        q = np.asarray([0.2, -0.4, 0.3], dtype=np.float32)
        target = np.asarray([1.5, 0.4], dtype=np.float32)
        physical_steps = []
        for signs in np.concatenate([TRAIN_SIGNS, OOD_SIGNS]):
            action, _ = oracle_action(q, target, signs)
            physical_steps.append(signs * action)
        for step in physical_steps[1:]:
            np.testing.assert_allclose(step, physical_steps[0], atol=1e-6)

    def test_eef_jacobian_matches_finite_difference(self):
        q = np.asarray([0.3, -0.5, 0.4], dtype=np.float32)
        epsilon = 1e-4
        for signs in np.concatenate([TRAIN_SIGNS, OOD_SIGNS]):
            analytic = eef_pixel_jacobian(q, signs)
            finite = np.zeros_like(analytic)
            for joint in range(3):
                raw = np.zeros(3, dtype=np.float32)
                raw[joint] = epsilon
                plus = world_to_pixel(keypoints(q + signs * raw)[-1])
                minus = world_to_pixel(keypoints(q - signs * raw)[-1])
                finite[:, joint] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(analytic, finite, atol=0.02, rtol=0.002)

    def test_pixel_field_changes_with_action_sign(self):
        q = np.asarray([0.1, 0.2, -0.3], dtype=np.float32)
        target = np.asarray([1.4, 0.2], dtype=np.float32)
        positive = make_sample(q, target, TRAIN_SIGNS[0]).pixel_jacobian
        flipped = make_sample(q, target, OOD_SIGNS[-1]).pixel_jacobian
        np.testing.assert_allclose(positive[:6], -flipped[:6], atol=1e-6)
        np.testing.assert_array_equal(positive[6], flipped[6])

    def test_sampled_targets_are_oracle_reachable(self):
        from action_jacobian.simple_servo import rollout_oracle

        rng = np.random.default_rng(7)
        for _ in range(50):
            q, target = sample_state(rng)
            success, _, _ = rollout_oracle(q, target, OOD_SIGNS[-1])
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
