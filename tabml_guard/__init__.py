"""TabML-Guard: Explainable, Drift-Aware AutoML for Time-Ordered Tabular Data."""

from .__version__ import __version__
from .config import Config, load_config
from .pipeline import TabMLGuard

__all__ = [
    "__version__",
    "TabMLGuard",
    "Config",
    "load_config",
]
