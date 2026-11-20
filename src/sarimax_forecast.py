"""Chronological holdout and explicit future covariates."""
import argparse
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_forecast(y, exog=None, order=(1,0,1), seasonal=(0,0,0,0), steps=30, future_exog=None):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or steps<1:
        raise ValueError('invalid target or horizon')
    if exog is not None:
        exog=np.asarray(exog,dtype=float); future_exog=np.asarray(future_exog,dtype=float)
        if exog.ndim!=2 or exog.shape[0]!=len(y) or future_exog.shape!=(steps,exog.shape[1]) or not np.isfinite(exog).all() or not np.isfinite(future_exog).all():
            raise ValueError('supply aligned historical and explicit future covariates')
    elif future_exog is not None: raise ValueError('unexpected future covariates')
    res=SARIMAX(y,exog=exog,order=order,seasonal_order=seasonal).fit(disp=False)
    if not res.mle_retvals.get('converged',False): raise RuntimeError('model did not converge')
    fc=res.get_forecast(steps=steps,exog=future_exog)
    return res,fc.predicted_mean,fc.conf_int(alpha=.05)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--y',required=True); ap.add_argument('--h',type=int,default=30); ap.add_argument('--order',default='1,0,1'); ap.add_argument('--seasonal',default='0,0,0,0'); a=ap.parse_args()
    y=pd.read_csv(a.csv)[a.y].to_numpy(float)
    if not 0<a.h<len(y): raise ValueError('holdout must leave training data')
    _,pred,_=fit_forecast(y[:-a.h],order=tuple(map(int,a.order.split(','))),seasonal=tuple(map(int,a.seasonal.split(','))),steps=a.h)
    err=y[-a.h:]-pred; print(json.dumps({'MAE':float(np.mean(abs(err))),'MSE':float(np.mean(err**2)),'holdout':a.h}))

if __name__=='__main__': main()
