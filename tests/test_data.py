"""Unit tests for data ingestion, schema profiling, task detection, and temporal splits."""
import unittest
import numpy as np
import pandas as pd

from tabml_guard.config import Config, DatasetConfig, SplitConfig
from tabml_guard.data import prepare_bundle, resolve_task


class TestData(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 60
        self.toy_clf_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "numeric_a": rng.normal(size=n),
            "numeric_b": rng.uniform(10, 50, size=n),
            "category_col": rng.choice(["Category_A", "Category_B", "Category_C"], size=n),
            "target": rng.choice([0, 1], size=n),
        })

        self.toy_reg_df = pd.DataFrame({
            "date_col": pd.date_range("2024-06-01", periods=n, freq="D"),
            "feature_1": rng.normal(loc=5, scale=2, size=n),
            "feature_2": rng.exponential(scale=1.5, size=n),
            "target_val": rng.normal(loc=100, scale=15, size=n),
        })

    def test_task_detection_classification(self):
        task = resolve_task(self.toy_clf_df["target"], task_cfg="auto")
        self.assertEqual(task, "classification")

    def test_task_detection_regression(self):
        task = resolve_task(self.toy_reg_df["target_val"], task_cfg="auto")
        self.assertEqual(task, "regression")

    def test_prepare_bundle_temporal_split(self):
        cfg = Config(
            dataset=DatasetConfig(
                source="csv",
                target="target",
                time_col="timestamp",
            ),
            split=SplitConfig(train_frac=0.6, test_frac=0.4),
        )

        bundle = prepare_bundle(self.toy_clf_df, cfg)
        self.assertTrue(bundle.has_time)
        self.assertEqual(bundle.task, "classification")
        self.assertEqual(len(bundle.X_train), 36)
        self.assertEqual(len(bundle.X_test), 24)
        self.assertLessEqual(bundle.time_train.max(), bundle.time_test.min())
        self.assertNotIn("target", bundle.X_train.columns)
        self.assertNotIn("timestamp", bundle.X_train.columns)

    def test_prepare_bundle_random_split_without_time(self):
        df_no_time = self.toy_clf_df.drop(columns=["timestamp"])
        cfg = Config(
            dataset=DatasetConfig(
                source="csv",
                target="target",
                time_col=None,
            ),
            split=SplitConfig(train_frac=0.6, test_frac=0.4),
        )

        bundle = prepare_bundle(df_no_time, cfg)
        self.assertFalse(bundle.has_time)
        self.assertEqual(len(bundle.X_train) + len(bundle.X_test), len(df_no_time))
        self.assertIsNone(bundle.time_train)
        self.assertIsNone(bundle.time_test)


if __name__ == "__main__":
    unittest.main()
