"""Unit tests for configuration loading, dataclass parsing, and validation."""
import tempfile
import unittest
from pathlib import Path

from tabml_guard.config import Config, DatasetConfig, SplitConfig, load_config


class TestConfig(unittest.TestCase):
    def test_config_load_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "valid_config.yaml"
            cfg_file.write_text(
                "dataset:\n"
                "  source: csv\n"
                "  path: dummy.csv\n"
                "  target: target_col\n"
                "  time_col: time_col\n"
                "task: auto\n"
                "split:\n"
                "  train_frac: 0.7\n"
                "  test_frac: 0.3\n"
                "output_dir: custom_results\n",
                encoding="utf-8",
            )

            cfg = load_config(cfg_file)
            self.assertEqual(cfg.dataset.source, "csv")
            self.assertEqual(cfg.dataset.path, "dummy.csv")
            self.assertEqual(cfg.dataset.target, "target_col")
            self.assertEqual(cfg.dataset.time_col, "time_col")
            self.assertEqual(cfg.task, "auto")
            self.assertEqual(cfg.split.train_frac, 0.7)
            self.assertEqual(cfg.split.test_frac, 0.3)
            self.assertEqual(cfg.output_dir, "custom_results")

    def test_config_missing_target_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "missing_target.yaml"
            cfg_file.write_text(
                "dataset:\n"
                "  source: csv\n"
                "  path: data.csv\n"
                "  target: ''\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(cfg_file)

    def test_config_invalid_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "bad_source.yaml"
            cfg_file.write_text(
                "dataset:\n"
                "  source: postgresql\n"
                "  target: y\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(cfg_file)

    def test_config_invalid_split_sum_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "bad_split.yaml"
            cfg_file.write_text(
                "dataset:\n"
                "  source: csv\n"
                "  path: data.csv\n"
                "  target: label\n"
                "split:\n"
                "  train_frac: 0.8\n"
                "  test_frac: 0.5\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(cfg_file)


if __name__ == "__main__":
    unittest.main()
