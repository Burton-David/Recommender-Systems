"""Exercise BPR's pure-Python fallback when the compiled kernel is unavailable.

`BPR.fit` imports `recommender_systems._kernels` inside the function body and
falls through to `_fit_python` on `ImportError`. CI always builds the kernel
(maturin is the build backend), so the fallback path would otherwise be
untested in normal CI runs. This module forces the fallback by poisoning
`sys.modules` before `BPR.fit` reaches its import.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from recommender_systems.bpr import BPR


@pytest.fixture
def block_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `from recommender_systems import _kernels` raise ImportError.

    Setting the entry to ``None`` in ``sys.modules`` is the documented way
    to signal "this module is not importable" — Python's import machinery
    treats the ``None`` sentinel as a deliberate block.
    """
    monkeypatch.setitem(sys.modules, "recommender_systems._kernels", None)


def _toy_ratings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 20, size=n),
            "item_id": rng.integers(0, 30, size=n),
            "rating": np.ones(n, dtype=np.float64),
        }
    ).drop_duplicates(subset=["user_id", "item_id"])


def test_python_fallback_trains_and_recommends(block_kernel: None) -> None:
    model = BPR(n_factors=8, epochs=3, random_state=0).fit(_toy_ratings())
    recs = model.recommend(0, n=5)
    assert isinstance(recs, list)
    assert len(recs) <= 5
    # Factors should have moved off the initialization.
    assert model._user_factors.shape == (20, 8)
    assert np.any(np.abs(model._user_factors) > 0)


def test_python_fallback_is_reproducible(block_kernel: None) -> None:
    ratings = _toy_ratings()
    a = BPR(n_factors=8, epochs=3, random_state=42).fit(ratings)
    b = BPR(n_factors=8, epochs=3, random_state=42).fit(ratings)
    np.testing.assert_allclose(a._user_factors, b._user_factors)
    np.testing.assert_allclose(a._item_factors, b._item_factors)
