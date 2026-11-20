"""Exploratory analysis of monthly S&P 500 index averages."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

def hurst_rs(x, scales=(8,12,16,24,32,48,64)):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('finite vector required')
    sizes=[]; values=[]
    for m in scales:
        if m<2 or len(x)//m<4: continue
        a=x[:len(x)//m*m].reshape(-1,m); a=a-a.mean(axis=1,keepdims=True)
        z=np.column_stack((np.zeros(len(a)),np.cumsum(a,axis=1)))
        s=a.std(axis=1,ddof=1); r=np.ptp(z,axis=1); ok=(s>0)&(r>0)
        if ok.any(): sizes.append(m); values.append(np.mean(r[ok]/s[ok]))
    if len(sizes)<3: raise ValueError('insufficient valid scales')
    return float(np.polyfit(np.log(sizes),np.log(values),1)[0])

def permutation(x, repeats=499, seed=42):
    rng=np.random.default_rng(seed); h=hurst_rs(x)
    null=np.array([hurst_rs(rng.permutation(x)) for _ in range(repeats)])
    return {'H':h,'null_median':float(np.median(null)),'null_95_interval':np.quantile(null,[.025,.975]).tolist(),'upper_tail_p':float((1+np.sum(null>=h))/(repeats+1)),'permutations':repeats}

def load(path):
    d=pd.read_csv(path,parse_dates=['Date']); d=d.set_index('Date')['SP500'].astype(float)
    if not d.index.is_unique or not d.index.is_monotonic_increasing or not np.isfinite(d).all() or (d<=0).any(): raise ValueError('invalid index series')
    if not d.index.equals(pd.date_range(d.index[0],periods=len(d),freq='MS')): raise ValueError('missing or nonmonthly timestamps')
    return d

def forecast(r, holdout=60):
    """Fixed specifications; expanding one-month-ahead forecasts; no tuning on test."""
    r=np.asarray(r,float); start=len(r)-holdout
    if start<60: raise ValueError('insufficient training data')
    predictions={k:[] for k in ('zero','historical_mean','seasonal_naive','AR1','XGBoost')}
    for t in range(start,len(r)):
        train=r[:t]; predictions['zero'].append(0.); predictions['historical_mean'].append(float(train.mean())); predictions['seasonal_naive'].append(float(train[-12]))
        model=SARIMAX(train,order=(1,0,0),trend='c').fit(disp=False)
        if not model.mle_retvals.get('converged',False): raise RuntimeError('AR1 failed to converge')
        predictions['AR1'].append(float(model.forecast(1)[0]))
        X=np.lib.stride_tricks.sliding_window_view(train,12)[:-1]; y=train[12:]
        reg=XGBRegressor(n_estimators=40,max_depth=2,learning_rate=.05,random_state=42,n_jobs=1)
        reg.fit(X,y); predictions['XGBoost'].append(float(reg.predict(train[-12:].reshape(1,-1))[0]))
    actual=r[start:]; out={}
    for k,v in predictions.items():
        err=actual-np.asarray(v); out[k]={'MAE':float(np.mean(abs(err))),'RMSE':float(np.sqrt(np.mean(err**2)))}
    return out,predictions

def run(path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=load(path); p=np.log(d.to_numpy()); r=np.diff(p); out.mkdir(parents=True,exist_ok=True)
    x={'log_level':p,'log_change':r,'absolute_log_change':abs(r)}
    report={'source':'https://github.com/datasets/s-and-p-500/blob/main/data/data.csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'start':str(d.index[0].date()),'end':str(d.index[-1].date()),'n':len(d),'frequency':'monthly average price index; not month-end or total return','series':{}}
    fig,axs=plt.subplots(3,2,figsize=(11,9))
    for i,(name,a) in enumerate(x.items()):
        stat,pval,*_=adfuller(a,autolag='AIC'); f,power=periodogram(a-np.mean(a),fs=1,window='hann',scaling='spectrum'); peak=int(np.argmax(power[1:])+1)
        report['series'][name]={'permutation':permutation(a),'ADF_stat':float(stat),'ADF_p':float(pval),'dominant_period_months':float(1/f[peak])}
        axs[i,0].plot(d.index[-len(a):],a); axs[i,0].set_title(name); axs[i,1].plot(f[1:],power[1:]); axs[i,1].set_title('Hann periodogram'); axs[i,1].set_xlabel('cycles/month')
    fig.tight_layout(); fig.savefig(out/'diagnostics.png',dpi=150); plt.close(fig)
    t=np.arange(len(p)); detrended=p-np.polyval(np.polyfit(t,p,1),t)
    months=d.index[1:].month.to_numpy(); deseason=r.copy()
    for m in range(1,13): deseason[months==m]-=r[months==m].mean()
    design=np.column_stack((np.ones(len(r)-1),r[:-1])); beta=np.linalg.lstsq(design,r[1:],rcond=None)[0]
    residual=r[1:]-design@beta
    report['sensitivity']={'detrended_log_level':permutation(detrended),'deseasonalized_log_change':permutation(deseason),'AR1_residual':permutation(residual),'AR1_coefficient':float(beta[1])}
    report['forecast'],pred=forecast(r); report['forecast_holdout']=60
    pd.DataFrame({'date':d.index[-60:],'actual':r[-60:],**pred}).to_csv(out/'predictions.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('csv',type=Path); ap.add_argument('--out',type=Path,default=Path('results')); a=ap.parse_args(); print(json.dumps(run(a.csv,a.out),indent=2))
