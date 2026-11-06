"""Pytest fixtures for TabML-Guard test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_clf_df() -> pd.DataFrame:
    """Synthetic binary classification dataset with chronological order."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "numeric_a": rng.normal(size=n),
        "numeric_b": rng.uniform(10, 50, size=n),
        "category_col": rng.choice(["Category_A", "Category_B", "Category_C"], size=n),
        "target": rng.choice([0, 1], size=n),
    })


@pytest.fixture
def toy_reg_df() -> pd.DataFrame:
    """Synthetic continuous regression dataset with chronological order."""
    rng = np.random.default_rng(99)
    n = 60
    return pd.DataFrame({
        "date_col": pd.date_range("2024-06-01", periods=n, freq="D"),
        "feature_1": rng.normal(loc=5, scale=2, size=n),
        "feature_2": rng.exponential(scale=1.5, size=n),
        "target_val": rng.normal(loc=100, scale=15, size=n),
    })

