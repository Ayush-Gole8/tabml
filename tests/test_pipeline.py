"""End-to-end smoke tests for the TabMLGuard pipeline and reporting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from tabml_guard import TabMLGuard
from tabml_guard.config import (
    Config,
    DatasetConfig,
    DriftConfig,
    ModelsConfig,
    SplitConfig,
    TuningConfig,
)


class TestPipeline(unittest.TestCase):
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

    def test_pipeline_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "test_run"
            cfg = Config(
                dataset=DatasetConfig(source="csv", target="target", time_col="timestamp"),
                split=SplitConfig(train_frac=0.6, test_frac=0.4),
                models=ModelsConfig(boosters=["random_forest"], stacking=False),
                tuning=TuningConfig(budget_trials=2, cv_splits=2),
                drift=DriftConfig(n_chunks=2),
                output_dir=str(out_dir),
            )

            guard = TabMLGuard(config=cfg)
            results = guard.fit(self.toy_clf_df)

            # Validate output structure
            self.assertEqual(results.get("task"), "classification")
            self.assertIn("baselines", results)
            self.assertIn("boosting", results)
            self.assertIn("tuning", results)
            self.assertIn("explain", results)
            self.assertIn("drift", results)

            # Validate model retrieval & inference
            model = guard.get_best_model()
            self.assertIsNotNone(model)
            preds = guard.predict(self.toy_clf_df.drop(columns=["target", "timestamp"]))
            self.assertEqual(len(preds), len(self.toy_clf_df))

            # Validate report exports
            html_report = guard.save_report(fmt="html")
            self.assertTrue(Path(html_report).exists())
            self.assertGreater(Path(html_report).stat().st_size, 0)

            pdf_report = guard.save_report(fmt="pdf")
            self.assertTrue(Path(pdf_report).exists())
            self.assertGreater(Path(pdf_report).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
