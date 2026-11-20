import numpy as np
import pytest

from src.hurst import hurst_rs, fractal_dimension_from_hurst
from src.metrics import mae, mse, mape
from src.synthetic import generate


def test_metrics():
    assert mae([1, 2], [2, 4]) == 1.5
    assert mse([1, 2], [2, 4]) == 2.5
    assert mape([1, 2], [2, 4]) == 100
    with pytest.raises(ValueError):
        mape([0], [1])
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_rs_affine_invariance():
    x = np.random.default_rng(42).normal(size=2048)
    h, details = hurst_rs(x)
    assert hurst_rs(3 * x + 8)[0] == pytest.approx(h)
    assert len(details["block_counts"]) >= 2
    assert fractal_dimension_from_hurst(.7259) == pytest.approx(1.2741)


@pytest.mark.parametrize("x", [np.ones(1024), [1, np.nan], [1, 2]])
def test_invalid_rs(x):
    with pytest.raises(ValueError):
        hurst_rs(x)


def test_synthetic():
    a = generate(100, 42)
    assert a.equals(generate(100, 42))
    assert len(a) == 100
    assert a.date.is_unique
    assert not a.equals(generate(100, 43))
