import numpy as np
import pytest
from src.research import hurst_rs, permutation, forecast
from src.sarimax_forecast import fit_forecast

def test_affine():
    x=np.random.default_rng(5).normal(size=300)
    assert hurst_rs(x)==pytest.approx(hurst_rs(3*x+9))

def test_constant():
    with pytest.raises(ValueError): hurst_rs(np.ones(300))

def test_permutation():
    x=np.random.default_rng(5).normal(size=300)
    assert permutation(x,19)==permutation(x,19)

def test_forecast_causality():
    x=np.random.default_rng(7).normal(0,.03,90)
    _,a=forecast(x,holdout=2)
    x[-1]=9
    _,b=forecast(x,holdout=2)
    assert a==b

def test_explicit_future_exog():
    with pytest.raises(ValueError):
        fit_forecast(np.arange(40.),exog=np.ones((40,1)),steps=2)

def test_forecast_shape():
    x=np.random.default_rng(8).normal(size=100)
    _,pred,ci=fit_forecast(x,order=(1,0,0),steps=3)
    assert pred.shape==(3,) and ci.shape==(3,2)
