"""Regression tests for the legacy top-level scripts kept at the repo root.

These files predate the package layout and aren't imported by anything else,
but they should still load cleanly so future contributors don't trip over
import-time side effects (see #3).
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_legacy(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_based_collaborative_filter_imports_without_side_effects():
    # Previously this module read a CSV and ran SVD at import time, which
    # crashed whenever the file wasn't present in the cwd.
    module = _import_legacy("model_based_collaborative_filter")
    assert callable(module.get_recommended_items)
