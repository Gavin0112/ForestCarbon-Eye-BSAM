"""CPU-only audit utility for frozen predictions; not the historical Test runner."""
import numpy as np
from scipy.stats import spearmanr

Q = {'q50':24.261154174804688, 'q90':214.8477752685547,
     'q95':292.0518798828124, 'q99':427.4302880859376}

def score(y, pred, mask):
    yy=np.asarray(y,dtype=np.float64)[mask]
    pp=np.asarray(pred,dtype=np.float64)[mask]
    if len(yy)==0: raise ValueError('empty stratum')
    e=pp-yy
    return {'n':int(len(yy)), 'rmse':float(np.sqrt(np.mean(e*e))),
            'mae':float(np.mean(np.abs(e))), 'bias':float(np.mean(e))}

def calculate(y, pred):
    y=np.asarray(y); pred=np.asarray(pred)
    if y.shape!=pred.shape or y.ndim!=1 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('invalid paired predictions')
    masks={'overall':np.ones(len(y),dtype=bool),
           'q50_q90':(y>Q['q50'])&(y<=Q['q90']),
           'q90_plus':y>Q['q90'], 'q95_plus':y>Q['q95'], 'q99_plus':y>Q['q99']}
    result={name:score(y,pred,mask) for name,mask in masks.items()}
    high=masks['q90_plus']
    result['q90_plus']['spearman']=float(spearmanr(y[high],pred[high]).statistic)
    return result
