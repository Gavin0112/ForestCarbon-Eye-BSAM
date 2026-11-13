"""One locked MSE-FiLM main run: official Train then full Validation; no Test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'n6_official_matched'))
sys.path.insert(0, str(EXP / 'p114_early_fusion'))
from n6_data import OfficialTrainData, OfficialValData, N_TRAIN, N_VAL
from run_p114_ef import alos_cache, numeric_alos
from mse_film_model import MSEFilm

SEED, ORDER_SEED, BATCH, EPOCHS, LR = 43, 4343, 128, 6, 1e-4
STEPS = math.ceil(N_TRAIN / BATCH)
Q = {'q50': 24.261154174804688, 'q90': 214.8477752685547,
     'q95': 292.0518798828124, 'q99': 427.4302880859376}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    p = path.with_suffix(path.suffix + '.partial')
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    p.replace(path)


def save_pt(path, obj):
    p = path.with_suffix(path.suffix + '.partial')
    torch.save(obj, p)
    p.replace(path)


def write_csv(path, rows):
    p = path.with_suffix(path.suffix + '.partial')
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    p.replace(path)


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


def config(data):
    return {'identity': 'MSE-FiLM main single model', 'seed': SEED,
            'order_seed': ORDER_SEED, 'train_rows': N_TRAIN, 'val_rows': N_VAL,
            'batch': BATCH, 'epochs': EPOCHS, 'steps_per_epoch': STEPS,
            'total_steps': STEPS * EPOCHS, 'optimizer': 'Adam', 'lr': LR,
            'scheduler': 'StepLR(30,0.1) once per epoch',
            'loss': 'centre-pixel MSE', 'precision': 'BF16 autocast, FP32 loss',
            'init': 'seed43 from scratch; no N5 checkpoint',
            'checkpoint_rule': 'fixed epoch6, no Validation selection',
            'normalization': 'P95 Train-fitted N5 S2/DEM/LC/8D plus P114 Train-only ALOS p1/p99',
            'normalization_sha256': sha(ROOT / 'experiments/agbd_lite/p95_geographic_multimodal_fusion/p95_1/P95_1_NORMALIZATION.json'),
            'alos_train_cache_sha256': sha(EXP / 'p113_alos_numeric_pilot/train_alos_prepare.json'),
            'train_q': {k: float(v) for k, v in zip(Q, data.q_official)},
            'dem_mean_m': float(data.stats['dem_pixel_mean']),
            'dem_std_m': float(data.stats['dem_pixel_std']),
            's2_band_order': ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'],
            'dem_grid_spacing_m': 10,
            'dn1': 'physical status unresolved; keep absolute numeric value and flag; exclude from radar texture estimates',
            'test_access': False}


def new_model(cfg):
    seed_all()
    return MSEFilm(cfg['dem_mean_m'], cfg['dem_std_m']).cuda()


def radar(cache, rows):
    dn = np.ascontiguousarray(np.asarray(cache[rows]).transpose(0, 3, 1, 2))
    x = numeric_alos(np.asarray(cache[rows]))
    return (torch.from_numpy(x).cuda(non_blocking=True),
            torch.from_numpy(dn.astype(np.int32)).cuda(non_blocking=True))


def preflight():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert not (HERE / 'MSE_FILM_EPOCH6.pt').exists()
    data = OfficialTrainData(); c = config(data)
    path = HERE / 'MSE_FILM_FROZEN_CONFIG.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8')) == c
    else: write_json(path, c)
    with h5py.File(ROOT / 'data/agbd_lite/AGBD-Lite-train.h5', 'r') as h:
        assert list(h['S2_bands'].attrs['order']) == c['s2_band_order']
        assert list(h['ALOS_bands'].attrs['order']) == ['HH', 'HV']
        assert h['ALOS_bands'].shape == (N_TRAIN, 25, 25, 2)
        raw = alos_cache('train')
        for i in (0, 117, N_TRAIN - 1): assert np.array_equal(h['ALOS_bands'][i], raw[i])
    m = new_model(c).train(); x = data.batch(np.arange(16)); a, dn = radar(raw, np.arange(16))
    assert [tuple(z.shape) for z in x[:4]] == [(16,12,25,25),(16,1,25,25),(16,26,25,25),(16,8)]
    assert a.shape == dn.shape == (16,2,25,25)
    assert torch.isfinite(a).all() and torch.isfinite(x[0]).all()
    with torch.autocast('cuda', dtype=torch.bfloat16): p = m(*x[:4], a, dn)
    loss = F.mse_loss(p.float(), x[4]); assert torch.isfinite(loss) and p.shape == (16,)
    loss.backward()
    grads = {name: float(next(mod.parameters()).grad.norm()) for name, mod in
             [('S2',m.s2),('DEM',m.dem.stem),('LC',m.lc.stem),('ALOS',m.alos)]}
    assert all(math.isfinite(v) and v > 0 for v in grads.values()), grads
    peak = torch.cuda.max_memory_allocated() / 2**20
    assert peak < 8192
    report = {'status':'PASS','grads':grads,'parameters':sum(z.numel() for z in m.parameters()),
              'peak_mib':peak,'config_sha256':sha(path),'test_accessed':False}
    write_json(HERE/'MSE_FILM_PREFLIGHT.json',report)
    print(json.dumps(report),flush=True)


def train():
    assert json.loads((HERE/'MSE_FILM_PREFLIGHT.json').read_text())['status'] == 'PASS'
    data = OfficialTrainData(); cfg = config(data)
    assert json.loads((HERE/'MSE_FILM_FROZEN_CONFIG.json').read_text()) == cfg
    m = new_model(cfg); opt = torch.optim.Adam(m.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=.1)
    resume = HERE/'MSE_FILM_RESUME.pt'
    logs=[]; first=1; start_at=0; sse_at=0.0
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config_sha256']==sha(HERE/'MSE_FILM_FROZEN_CONFIG.json')
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);set_rng(state['rng'])
        logs=state['logs'];first=state['epoch'];start_at=state['next_start'];sse_at=state['sse']
        print(json.dumps({'resume_epoch':first,'row':start_at}),flush=True)
    raw=alos_cache('train')
    for epoch in range(first,EPOCHS+1):
        m.train();torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
        order=np.random.default_rng(ORDER_SEED+epoch).permutation(data.all_rows)
        start0=start_at if epoch==first else 0
        sse=sse_at if epoch==first else 0.
        for start in range(start0,N_TRAIN,BATCH):
            ix=order[start:start+BATCH];batch=data.batch(ix);a,dn=radar(raw,ix)
            with torch.autocast('cuda',dtype=torch.bfloat16): out=m(*batch[:4],a,dn)
            loss=F.mse_loss(out.float(),batch[4])
            if not bool(torch.isfinite(loss)):raise FloatingPointError((epoch,start,'loss'))
            opt.zero_grad(set_to_none=True)
            (loss*len(ix)/BATCH).backward()
            if start//BATCH in (0,1000,2000,2905):
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
            opt.step();sse+=float(loss.detach())*len(ix)
            next_start=min(start+BATCH,N_TRAIN)
            if next_start%(BATCH*500)==0:
                save_pt(resume,{'epoch':epoch,'next_start':next_start,'sse':sse,
                    'model':m.state_dict(),'optimizer':opt.state_dict(),
                    'scheduler':scheduler.state_dict(),'rng':rng_state(),'logs':logs,
                    'config_sha256':sha(HERE/'MSE_FILM_FROZEN_CONFIG.json')})
                print(json.dumps({'epoch':epoch,'steps':(epoch-1)*STEPS+next_start//BATCH}),flush=True)
        scheduler.step();torch.cuda.synchronize()
        rec={'epoch':epoch,'steps':epoch*STEPS,'train_n':N_TRAIN,'train_mse':sse/N_TRAIN,
             'train_seconds':time.perf_counter()-tic,'peak_mib':torch.cuda.max_memory_allocated()/2**20,
             'lr':opt.param_groups[0]['lr']}
        logs.append(rec);write_csv(HERE/'MSE_FILM_TRAINING.csv',logs)
        save_pt(resume,{'epoch':epoch+1,'next_start':0,'sse':0.,'model':m.state_dict(),
                'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
                'logs':logs,'config_sha256':sha(HERE/'MSE_FILM_FROZEN_CONFIG.json')})
        print(json.dumps(rec),flush=True)
    assert len(logs)==6
    save_pt(HERE/'MSE_FILM_EPOCH6.pt',{'epoch':6,'steps':STEPS*6,'model':m.state_dict(),
           'config_sha256':sha(HERE/'MSE_FILM_FROZEN_CONFIG.json')})
    write_json(HERE/'MSE_FILM_TRAINING_LOCK.json',{'status':'COMPLETE','checkpoint_sha256':sha(HERE/'MSE_FILM_EPOCH6.pt'),
               'steps':STEPS*6,'test_accessed':False})


def metric(y,p):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);e=p-y
    masks={'overall':np.ones(len(y),bool),'q50_q90':(y>Q['q50'])&(y<=Q['q90']),
           'q90_plus':y>Q['q90'],'q95_plus':y>Q['q95'],'q99_plus':y>Q['q99']}
    out={}
    for name,sel in masks.items():
        ee=e[sel];out[name]={'n':int(sel.sum()),'rmse':float(np.sqrt(np.mean(ee**2))),
                            'mae':float(np.mean(abs(ee))),'bias':float(np.mean(ee))}
    s=masks['q90_plus'];out['q90_plus']['spearman']=float(spearmanr(y[s],p[s]).statistic)
    return out


def paired(y,base,pred,grid):
    rng=np.random.default_rng(20261022);out={}
    for name,sel in {'overall':np.ones(len(y),bool),'q50_q90':(y>Q['q50'])&(y<=Q['q90']),
                     'q90_plus':y>Q['q90']}.items():
        frame=pd.DataFrame({'grid':grid[sel],'y':y[sel],'base':base[sel],'pred':pred[sel]})
        cells=[]
        for _,g in frame.groupby('grid'):
            yy=g.y.to_numpy();cells.append((len(g),np.square(g.base.to_numpy()-yy).sum(),
                                             np.square(g.pred.to_numpy()-yy).sum()))
        cells=np.asarray(cells,np.float64)
        draws=cells[rng.integers(len(cells),size=(2000,len(cells)))].sum(1)
        gain=np.sqrt(draws[:,1]/draws[:,0])-np.sqrt(draws[:,2]/draws[:,0])
        out[name]={'grids':len(cells),'gain_base_minus_model':float(np.sqrt(cells[:,1].sum()/cells[:,0].sum())-
               np.sqrt(cells[:,2].sum()/cells[:,0].sum())),'ci95':np.quantile(gain,[.025,.975]).tolist()}
    return out


@torch.inference_mode()
def evaluate():
    lock=json.loads((HERE/'MSE_FILM_TRAINING_LOCK.json').read_text())
    assert lock['status']=='COMPLETE' and lock['checkpoint_sha256']==sha(HERE/'MSE_FILM_EPOCH6.pt')
    data=OfficialTrainData();cfg=config(data);val=OfficialValData(data);raw=alos_cache('val')
    m=new_model(cfg)
    ck=torch.load(HERE/'MSE_FILM_EPOCH6.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==6 and ck['config_sha256']==sha(HERE/'MSE_FILM_FROZEN_CONFIG.json')
    m.load_state_dict(ck['model'],strict=True);m.eval()
    pred=np.empty(N_VAL,np.float32);t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for start in range(0,N_VAL,64):
        stop=min(start+64,N_VAL);ix=np.arange(start,stop,dtype=np.int64)
        batch=val.batch(start,stop);a,dn=radar(raw,ix)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=m(*batch[:4],a,dn).float()
        assert bool(torch.isfinite(p).all());pred[start:stop]=p.cpu().numpy()
    seconds=time.perf_counter()-t
    ref=pd.read_csv(EXP/'n6_official_matched/M2_OFFICIAL_VAL_PREDICTIONS.csv')
    assert len(ref)==N_VAL and np.array_equal(ref.original_val_row.to_numpy(),np.arange(N_VAL))
    y=np.asarray(val.y,np.float32)
    assert np.array_equal(ref.agb_true.to_numpy(np.float32),y)
    pd.DataFrame({'original_val_row':np.arange(N_VAL),'grid_1deg':ref.grid_1deg,
                  'agb_true':y,'n5_pred':ref.agb_pred,'mse_film_pred':pred}).to_csv(HERE/'MSE_FILM_VAL_PREDICTIONS.csv',index=False)
    base=ref.agb_pred.to_numpy(np.float64)
    result={'model':metric(y,pred),'n5':metric(y,base),'paired':paired(y,base,pred,ref.grid_1deg.to_numpy()),
            'params':sum(z.numel() for z in m.parameters()),'train_seconds':float(pd.read_csv(HERE/'MSE_FILM_TRAINING.csv').train_seconds.sum()),
            'validation_inference_seconds':seconds,'peak_inference_mib':torch.cuda.max_memory_allocated()/2**20,
            'checkpoint_sha256':sha(HERE/'MSE_FILM_EPOCH6.pt'),'test_accessed':False}
    write_json(HERE/'MSE_FILM_VALIDATION.json',result);val.close();print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('preflight','train','evaluate'))
    phase=ap.parse_args().phase
    if phase=='preflight':preflight()
    elif phase=='train':train()
    else:evaluate()
