"""Fixed six-epoch N4 pilot; N1 data, order, optimizer, and model-selection protocol."""
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

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N1_DIR = HERE.parent / 'n1_niconet_mrf'
P95 = ROOT / 'experiments/agbd_lite/p95_geographic_multimodal_fusion'
sys.path.insert(0, str(N1_DIR))
import run_n1 as N1
from n4_model import NicoNet_SSCF

SEED, ORDER_SEED, BATCH, EPOCHS, LR = 43, 4343, 128, 6, 1e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k, t in state.items():
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path, rows):
    tmp = Path(path).with_suffix('.partial')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def atomic_save(path, obj):
    temp = Path(path).with_suffix('.partial')
    torch.save(obj, temp)
    temp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().square().sum())
    return math.sqrt(total)


def preflight(data):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    prior = json.loads((N1_DIR/'N1_FROZEN_CONFIG.json').read_text(encoding='utf-8'))
    assert (prior['seed'], prior['order_seed'], prior['batch'], prior['accum'],
            prior['epochs'], prior['lr']) == (SEED, ORDER_SEED, BATCH, 1, EPOCHS, LR)
    assert prior['model_sha256'] == sha(N1_DIR/'model.py')
    assert prior['split_sha256'] == sha(P95/'P95_0_SPATIAL_SPLIT.csv')
    assert prior['normalization_sha256'] == sha(P95/'p95_1'/'P95_1_NORMALIZATION.json')
    assert tuple(map(lambda r: len(data.roles[r]), ('inner_fit','inner_validation','spatial_dev'))) == (255972,51027,64850)
    ck = torch.load(N1_DIR/'M0'/'best.pt', map_location='cpu', weights_only=False)
    assert ck['epoch'] == 2 and ck['config'] == prior
    assert sha(N1_DIR/'M0'/'best.pt') == json.loads((N1_DIR/'N1_PRE_DEV_LOCK.json').read_text())['M0']
    N1.seed_all(); control = N1.N1NicoNet(False)
    N1.seed_all(); model = NicoNet_SSCF()
    shared_equal = all(torch.equal(t, model.base.state_dict()[k]) for k,t in control.state_dict().items())
    assert shared_equal
    shared_sha = state_sha(control.state_dict())
    assert sum(p.numel() for p in control.parameters()) == 1861762
    assert not torch.count_nonzero(model.delta.weight) and not torch.count_nonzero(model.delta.bias)
    control.cuda(); model.cuda()
    batch = data.batch(data.roles['inner_fit'][:BATCH]); model.capture = True
    control.eval(); model.eval(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        old = control(*batch[:4]); new = model(*batch[:4])
    maxdiff = float((old.float()-new.float()).abs().max())
    assert maxdiff <= 1e-5 and new.shape == (BATCH,) and torch.isfinite(new).all()
    initial = {k: float(v.cpu()) for k,v in model.aux.items()}
    assert initial['spectral_rms'] > 0 and initial['spatial_rms'] > 0
    assert initial['delta_rms'] == 0
    model.train(); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(*batch[:4]); loss=F.mse_loss(prediction.float(),batch[4])
    assert torch.isfinite(loss)
    loss.backward()
    first_delta_grad = norm(model.delta.parameters())
    assert first_delta_grad > 0 and math.isfinite(first_delta_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(*batch[:4]); loss2=F.mse_loss(prediction.float(),batch[4])
    assert torch.isfinite(loss2)
    loss2.backward(); torch.cuda.synchronize()
    spectral_grad=norm(model.spectral.parameters())
    spatial_grad=norm(model.spatial.parameters())
    interaction_grad=norm(model.interaction.parameters())
    assert all(math.isfinite(x) and x>0 for x in (spectral_grad,spatial_grad,interaction_grad))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    report=dict(shared_initial_state_sha256=shared_sha,
        shared_initial_tensors_equal=True, initial_prediction_max_abs_difference=maxdiff,
        initial_feature_rms=initial, first_step_delta_gradient_l2=first_delta_grad,
        second_step_spectral_gradient_l2=spectral_grad,
        second_step_spatial_gradient_l2=spatial_grad,
        second_step_interaction_gradient_l2=interaction_grad,
        m0_params=sum(p.numel() for p in control.parameters()),
        n4_params=sum(p.numel() for p in model.parameters()),
        train_batch=BATCH, peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        two_smoke_steps_seconds=time.perf_counter()-start, status='PASS')
    (HERE/'N4_GATE0.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps({'gate0':report}),flush=True)


def config(gate):
    obj=dict(seed=SEED,order_seed=ORDER_SEED,batch=BATCH,accum=1,epochs=EPOCHS,
      lr=LR,scheduler='StepLR(30,0.1)',loss='center-pixel MSE',
      precision='BF16 AMP, FP32 loss',augmentation='none',
      model='NicoNet_SSCF',transformer_dropout=0.0,
      selection_epochs=[2,4,6],selection_metric='inner_val_rmse',
      shared_initial_state_sha256=gate['shared_initial_state_sha256'],
      n1_config_sha256=sha(N1_DIR/'N1_FROZEN_CONFIG.json'),
      n1_m0_best_sha256=sha(N1_DIR/'M0'/'best.pt'),
      split_sha256=sha(P95/'P95_0_SPATIAL_SPLIT.csv'),
      normalization_sha256=sha(P95/'p95_1'/'P95_1_NORMALIZATION.json'),
      model_sha256=sha(HERE/'n4_model.py'),run_sha256=sha(__file__),
      torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    path=HERE/'N4_FROZEN_CONFIG.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==obj
    else:
        path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return obj


def predict(model,data,rows):
    model.eval(); ys=[]; preds=[]
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH):
            x=data.batch(rows[start:start+BATCH])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model(*x[:4])
            ys.append(x[4].cpu().numpy()); preds.append(p.float().cpu().numpy())
    return np.concatenate(ys),np.concatenate(preds)


def train(data,cfg):
    N1.seed_all(); model=NicoNet_SSCF().cuda()
    assert state_sha(model.base.state_dict())==cfg['shared_initial_state_sha256']
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=.1)
    resume=HERE/'resume.pt'; rows=[];best=float('inf');best_epoch=None;first_epoch=1;step=0
    N1.seed_all()
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['config']==cfg
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        set_rng(ck['rng']);rows=ck['rows'];best=ck['best'];best_epoch=ck['best_epoch']
        first_epoch=ck['epoch']+1;step=ck['step']
        print(json.dumps({'resumed_from_epoch':ck['epoch']}),flush=True)
    for epoch in range(first_epoch,EPOCHS+1):
        model.train(); model.capture=False;torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
        order=np.random.default_rng(ORDER_SEED+epoch).permutation(data.roles['inner_fit'])
        optimizer.zero_grad(set_to_none=True);total_sse=0.0;seen=0
        for start in range(0,len(order),BATCH):
            indices=order[start:start+BATCH];x=data.batch(indices)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model(*x[:4]);loss=F.mse_loss(p.float(),x[4])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'N4 nonfinite loss epoch={epoch},batch={start}')
            (loss*len(indices)/BATCH).backward()
            # The output projection is exactly zero at initialization, so the
            # interior paths acquire gradients only after the first update.
            if start//BATCH in (1,1000,1999):
                values=(norm(model.spectral.parameters()),norm(model.spatial.parameters()),
                        norm(model.interaction.parameters()))
                if not all(math.isfinite(v) and v>0 for v in values):
                    raise FloatingPointError(f'N4 path gradient missing epoch={epoch},batch={start}')
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            total_sse+=float(loss.detach())*len(indices);seen+=len(indices)
        assert seen==255972 and step==epoch*math.ceil(len(order)/BATCH)
        scheduler.step();torch.cuda.synchronize();train_sec=time.perf_counter()-t0
        row=dict(epoch=epoch,optimizer_steps=step,train_samples=seen,train_mse=total_sse/seen,
          train_seconds=train_sec,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
          lr=optimizer.param_groups[0]['lr'],val_rmse='',val_mae='',val_seconds='')
        if epoch in (2,4,6):
            t0=time.perf_counter();y,p=predict(model,data,data.roles['inner_validation'])
            m=N1.metrics(y,p);row.update(val_rmse=m['rmse'],val_mae=m['mae'],
                                       val_seconds=time.perf_counter()-t0)
            if m['rmse']<best:
                best=m['rmse'];best_epoch=epoch
                atomic_save(HERE/'best.pt',dict(model=model.state_dict(),epoch=epoch,
                           step=step,config=cfg,inner_val=m))
        rows.append(row)
        atomic_save(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
          scheduler=scheduler.state_dict(),rng=rng_state(),epoch=epoch,step=step,
          rows=rows,best=best,best_epoch=best_epoch,config=cfg))
        write_csv(HERE/'N4_TRAINING_LOG.csv',rows)
        print(json.dumps(row),flush=True)
    assert len(rows)==6 and best_epoch in (2,4,6)
    lock=dict(config_sha256=sha(HERE/'N4_FROZEN_CONFIG.json'),best_sha256=sha(HERE/'best.pt'),
      best_epoch=best_epoch,best_inner_rmse=best,epochs_completed=6,
      spatial_dev_evaluated=False)
    (HERE/'N4_PRE_DEV_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print(json.dumps({'training_complete':True,'best_epoch':best_epoch,'best_inner_rmse':best}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('gate0','train'))
    args=parser.parse_args()
    torch.backends.cudnn.benchmark=True
    data=N1.Data()
    if args.phase=='gate0':
        preflight(data)
    else:
        gate=json.loads((HERE/'N4_GATE0.json').read_text(encoding='utf-8'))
        assert gate['status']=='PASS'
        cfg=config(gate);train(data,cfg)


if __name__=='__main__':main()
