from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class HydraConfigTest(unittest.TestCase):
    def compose(self, name: str, overrides: list[str] | None = None):
        with initialize_config_dir(
            version_base="1.3", config_dir=str(CONFIG_DIR)
        ):
            return compose(config_name=name, overrides=overrides or [])

    def test_all_methods_compose(self):
        for method in ("none", "sign_array", "global_token", "pixel_jacobian"):
            with self.subTest(method=method):
                cfg = self.compose("train", [f"method={method}"])
                self.assertEqual(cfg.method.name, method)
                self.assertEqual(cfg.data.chunk_size, 30)
                self.assertEqual(cfg.data.configs_per_frame, 2)
                self.assertEqual(cfg.evaluation.rollout_seeds_per_config, 3)

    def test_all_dataset_stages_compose(self):
        expected = {
            "collect": "gen_robosuite_format_demo.py",
            "build_cache": "build_pixel_jacobian_cache.py",
            "build_manifest": "build_sign_dr_manifest.py",
            "validate": "run_sign_dr_unit_test.py",
        }
        for stage, script in expected.items():
            with self.subTest(stage=stage):
                cfg = self.compose("generate", [f"stage={stage}"])
                self.assertEqual(cfg.stage.script, script)

    def test_eval_defaults_do_not_run_rollout_conditionally(self):
        cfg = self.compose("eval")
        self.assertFalse(cfg.evaluation.skip_rollout)
        self.assertEqual(cfg.evaluation.rollout_seeds, 50)
        self.assertEqual(cfg.evaluation.rollout_horizon, 400)


if __name__ == "__main__":
    unittest.main()
