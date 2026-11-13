"""P114-EF: matched from-scratch 41-channel N5 early fusion, Train/Val only."""
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
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
N6=HERE.parent/'n6_official_matched'
PILOT=HERE.parent/'p113_alos_numeric_pilot'
sys.path.insert(0,str(N6))
import run_n6 as N6RUN
from n6_data import OfficialTrainData,OfficialValData,N_TRAIN,N_VAL

SEED,ORDER_SEED,BATCH,EPOCHS,LR=43,4343,128,6,1e-4
STEPS=math.ceil(N_TRAIN/BATCH)
N5_SHA='8dfa52323a20237d1a18e864dbba882f75d190e55827cc68b5de8115f64129fe'
INITIAL_N1_SHA='111cad3d743058bb68a6ce14897e24f62adb4e7d84e65b2103be6eba79c47a9b'
P1=np.array([-83.,-83.],np.float32)
P99=np.array([-1.99859619140625,-7.4848785400390625],np.float32)
TRAIN_ALOS=PILOT/'train_alos.u16';VAL_ALOS=PILOT/'val_alos.u16'
Q50=24.261154174804688;Q90=214.8477752685547;Q99=427.4302880859376
ARMS=('M0_zero','M1_real')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def state_sha(state):
    h=hashlib.sha256()
    for k,t in state.items():
        h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)


def atomic_torch(path,obj):
    tmp=path.with_suffix(path.suffix+'.partial');torch.save(obj,tmp);tmp.replace(path)


def write_csv(path,rows):
    if not rows:return
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)


class N5EF(nn.Module):
    """Original N5, with only the two first-layer 39-channel convolutions widened."""
    def __init__(self,base):
        super().__init__()
        self.n5=base
        entry=self.n5.base.backbone.body.entry_block
        assert entry.in_channels==39
        assert entry.conv1.in_channels==39 and entry.conv_shortcut.in_channels==39
        old_main,old_short=entry.conv1,entry.conv_shortcut
        def widen(old):
            new=nn.Conv2d(41,old.out_channels,1,bias=old.bias is not None)
            with torch.no_grad():
                new.weight.zero_();new.weight[:,:39].copy_(old.weight)
                if old.bias is not None:new.bias.copy_(old.bias)
            return new
        entry.conv1=widen(old_main)
        entry.conv_shortcut=widen(old_short)
        entry.in_channels=41
        self.n5.base.backbone.in_features=41
        assert sum(p.numel() for p in self.parameters())==2077991+768

    def forward(self,s2,dsm,lc,meta,radar):
        x=torch.cat((s2,dsm,lc,radar),dim=1)
        assert x.shape[1:]==(41,25,25) and meta.shape[1]==8
        return self.n5.base.backbone((x,meta))[:,0,12,12]


def fresh_pair():
    """Recreate original seed43 untrained N5 before adding zero SAR slices."""
    base=N6RUN.fresh_model('M2')
    original={k:t.detach().cpu().clone() for k,t in base.state_dict().items()}
    ef=N5EF(base)
    assert ef.n5.base.backbone.body.entry_block.conv1.weight.shape[1]==41
    assert ef.n5.base.backbone.body.entry_block.conv_shortcut.weight.shape[1]==41
    return ef,original


def alos_cache(split):
    n,path=(N_TRAIN,TRAIN_ALOS) if split=='train' else (N_VAL,VAL_ALOS)
    assert path.exists() and path.stat().st_size==n*25*25*2*2
    return np.memmap(path,mode='r',dtype=np.uint16,shape=(n,25,25,2))


def numeric_alos(dn):
    raw=dn.astype(np.float32)
    db=20.0*np.log10(np.maximum(raw,1.0))-83.0
    out=np.clip((db-P1)/(P99-P1),0,1)
    out[dn==0]=0
    return np.ascontiguousarray(out.transpose(0,3,1,2))


def radar_batch(arm,cache,rows):
    assert arm in ARMS
    a=numeric_alos(np.asarray(cache[rows]))
    assert np.isfinite(a).all()
    if arm=='M0_zero':a.fill(0)
    return torch.from_numpy(a).cuda(non_blocking=True)


def config():
    return dict(task='P114_EF',protocol='ALOS_NUMERICAL_PROTOCOL_ONLY',
        source_n5_checkpoint=str(N6/'M2_epoch6.pt'),source_n5_sha256=N5_SHA,
        initialization='seed43 original untrained N5; optical weights copied; two new radar input slices zero',
        original_n1_initial_sha256=INITIAL_N1_SHA,
        architecture='N5 39->41 only entry conv1 and entry conv_shortcut',
        channels={'S2':[0,11],'DEM':[12,12],'LC':[13,38],'ALOS_HH':[39,39],'ALOS_HV':[40,40]},
        original_input='same N6 OfficialTrainData/OfficialValData',
        radar_source='existing H5 ALOS_bands HH/HV copied row-aligned to P113 read-only cache',
        alos_cache_train=str(TRAIN_ALOS),alos_cache_val=str(VAL_ALOS),
        alos_numeric='DN>0:20*log10(DN)-83 dB; DN0->0; Train p1/p99; clip [0,1]',
        alos_p1=P1.tolist(),alos_p99=P99.tolist(),dn1_physical_qa='UNRESOLVED',
        seed=SEED,order_seed=ORDER_SEED,train_rows=N_TRAIN,val_rows=N_VAL,
        epochs=EPOCHS,batch=BATCH,steps_per_epoch=STEPS,total_steps=STEPS*EPOCHS,
        optimizer='Adam',lr=LR,scheduler='StepLR(30,0.1) once per epoch',
        precision='BF16 autocast/FP32 loss',loss='centre pixel MSE',augmentation='none',
        last_short_batch_gradient_factor=(N_TRAIN%BATCH)/BATCH,
        checkpoint_epochs=[2,4,6],evaluation_epochs=[2,4,6],official_result_epoch=6,
        m0_max_abs_rmse_gap_from_n5=.30,
        m1_gain_min=.5,m1_spatial_ci_lower_gt=0,m1_q90_non_degrade=True,
        paired_bootstrap_repeats=3000,paired_bootstrap_seed=20261020,
        q50=Q50,q90_strict_gt=Q90,q99_strict_gt=Q99,test_access=False)


def lock_config():
    HERE.mkdir(parents=True,exist_ok=True)
    p=HERE/'P114_EF_CONFIG.json';cfg=config()
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==cfg
    else:atomic_json(p,cfg)
    assert sha(N6/'M2_epoch6.pt')==N5_SHA
    return cfg


def preflight():
    cfg=lock_config()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train=OfficialTrainData();assert len(train.all_rows)==N_TRAIN
    with h5py.File(ROOT/'data/agbd_lite/AGBD-Lite-train.h5','r') as h:
        d=h['ALOS_bands'];assert d.shape==(N_TRAIN,25,25,2) and d.dtype==np.uint16
        assert list(d.attrs['order'])==['HH','HV']
        cache=alos_cache('train')
        for i in (0,117,N_TRAIN-1):assert np.array_equal(d[i],cache[i])
    with h5py.File(ROOT/'data/agbd_lite/AGBD-Lite-val.h5','r') as h:
        d=h['ALOS_bands'];assert d.shape==(N_VAL,25,25,2) and d.dtype==np.uint16
        assert list(d.attrs['order'])==['HH','HV']
        cachev=alos_cache('val')
        for i in (0,117,N_VAL-1):assert np.array_equal(d[i],cachev[i])
    m0,original=fresh_pair();m1,original1=fresh_pair()
    assert all(torch.equal(original[k],original1[k]) for k in original)
    assert state_sha(m0.state_dict())==state_sha(m1.state_dict())
    entry=m0.n5.base.backbone.body.entry_block
    optical_ok=True;new_zero=True
    for oldname,newname in (('base.backbone.body.entry_block.conv1','n5.base.backbone.body.entry_block.conv1'),
                            ('base.backbone.body.entry_block.conv_shortcut','n5.base.backbone.body.entry_block.conv_shortcut')):
        w=original[oldname+'.weight'];nw=m0.state_dict()[newname+'.weight']
        optical_ok &= torch.equal(w,nw[:,:39])
        new_zero &= bool(torch.count_nonzero(nw[:,39:])==0)
        optical_ok &= torch.equal(original[oldname+'.bias'],m0.state_dict()[newname+'.bias'])
    for k in original:
        if k.endswith('conv1.weight') and 'entry_block' in k:continue
        if k.endswith('conv_shortcut.weight') and 'entry_block' in k:continue
        optical_ok &= torch.equal(original[k],m0.state_dict()['n5.'+k])
    assert optical_ok and new_zero
    m0=m0.cuda().eval();m1=m1.cuda().eval()
    orig=N6RUN.fresh_model('M2').cuda().eval()
    indices=np.arange(BATCH,dtype=np.int64);batch=train.batch(indices)
    r_real=radar_batch('M1_real',alos_cache('train'),indices)
    r_zero=torch.zeros_like(r_real)
    assert tuple(t.shape for t in batch[:4])==((128,12,25,25),(128,1,25,25),(128,26,25,25),(128,8))
    assert all(bool(torch.isfinite(x).all()) for x in (*batch,r_real))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        p_orig=orig(*batch[:4]);p0=m0(*batch[:4],r_zero);p1=m1(*batch[:4],r_real)
    delta0=float((p0-p_orig).abs().max());delta1=float((p1-p0).abs().max())
    assert delta0<=1e-5 and delta1<=1e-5,(delta0,delta1)
    m1.train();m1.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=F.mse_loss(m1(*batch[:4],r_real).float(),batch[4])
    assert bool(torch.isfinite(loss));loss.backward()
    grad_main=m1.n5.base.backbone.body.entry_block.conv1.weight.grad[:,39:]
    grad_short=m1.n5.base.backbone.body.entry_block.conv_shortcut.weight.grad[:,39:]
    assert all(bool(torch.isfinite(g).all()) and float(g.norm())>0 for g in (grad_main,grad_short))
    peak=torch.cuda.max_memory_allocated()/2**20;assert peak<8192
    norm_stats=json.loads((ROOT/'experiments/agbd_lite/p95_geographic_multimodal_fusion/p95_1/P95_1_NORMALIZATION.json').read_text(encoding='utf-8'))
    val=OfficialValData(train)
    archive=N6/'M2_OFFICIAL_VAL_PREDICTIONS.csv';bad=0;count=0
    with archive.open(newline='',encoding='utf-8') as f:
        for count,row in enumerate(csv.DictReader(f),1):
            i=count-1
            if int(row['original_val_row'])!=i or float(row['agb_true'])!=float(val.y[i]):bad+=1
    assert count==N_VAL and bad==0
    val.close()
    qa=dict(status='PASS',config_sha256=sha(HERE/'P114_EF_CONFIG.json'),
        formal_n5_sha256=sha(N6/'M2_epoch6.pt'),initial_41_state_sha256=state_sha(m0.state_dict()),
        initial_original_state_sha256=state_sha(original),initial_m0_m1_exact=True,
        original_39_weights_exact=bool(optical_ok),new_radar_slices_zero=bool(new_zero),
        step0_original_vs_m0_max_abs=delta0,step0_m0_vs_m1_max_abs=delta1,
        radar_conv1_gradient_l2=float(grad_main.norm()),radar_shortcut_gradient_l2=float(grad_short.norm()),
        forward_loss=float(loss),peak_cuda_mib=peak,train_rows=N_TRAIN,val_rows=N_VAL,
        validation_archive_bad_rows=bad,original_train_norm_sha256=sha(ROOT/'experiments/agbd_lite/p95_geographic_multimodal_fusion/p95_1/P95_1_NORMALIZATION.json'),
        original_s2_p1=norm_stats['s2_p1'],original_s2_p99=norm_stats['s2_p99'],
        alos_p1=P1.tolist(),alos_p99=P99.tolist(),alos_numerical_protocol_only=True,test_accessed=False)
    atomic_json(HERE/'INITIALIZATION_EQUIVALENCE.json',qa)
    print(json.dumps(dict(phase='A',status='PASS',delta0=delta0,delta1=delta1,peak_cuda_mib=peak,
        grad_main=qa['radar_conv1_gradient_l2'],grad_short=qa['radar_shortcut_gradient_l2'])),flush=True)


def assert_preflight():
    cfg=lock_config()
    qa=json.loads((HERE/'INITIALIZATION_EQUIVALENCE.json').read_text(encoding='utf-8'))
    assert qa['status']=='PASS' and qa['config_sha256']==sha(HERE/'P114_EF_CONFIG.json')
    return cfg,qa


def train(arm):
    cfg,qa=assert_preflight();assert arm in ARMS
    if arm=='M1_real':
        decision=json.loads((HERE/'M0_GATE.json').read_text(encoding='utf-8'))
        assert decision['status']=='M0_REPRODUCTION_PASS'
    done=HERE/f'{arm}_TRAINING_LOCK.json'
    if done.exists():
        lock=json.loads(done.read_text(encoding='utf-8'))
        assert lock['epochs_completed']==6 and lock['steps']==17436
        print(json.dumps(dict(arm=arm,already_complete=True)),flush=True);return
    data=OfficialTrainData();raw=alos_cache('train')
    model,original=fresh_pair();assert state_sha(model.state_dict())==qa['initial_41_state_sha256']
    model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=.1)
    resume=HERE/f'{arm}_resume.pt';logs=[];epoch_start=1;row_start=0;sse_start=0.0
    seed_all()
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['arm']==arm and ck['config_sha256']==sha(HERE/'P114_EF_CONFIG.json')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);logs=ck['logs']
        epoch_start=ck['epoch'];row_start=ck['next_start'];sse_start=ck['sse']
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        print(json.dumps(dict(arm=arm,resume_epoch=epoch_start,resume_row=row_start)),flush=True)
    for epoch in range(epoch_start,EPOCHS+1):
        model.train();torch.cuda.reset_peak_memory_stats();tic=time.perf_counter()
        assert all(m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d))
        order=np.random.default_rng(ORDER_SEED+epoch).permutation(data.all_rows)
        start0=row_start if epoch==epoch_start else 0
        sse=sse_start if epoch==epoch_start else 0.0
        for start in range(start0,N_TRAIN,BATCH):
            ix=order[start:start+BATCH]
            x=data.batch(ix);radar=radar_batch(arm,raw,ix)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(*x[:4],radar);loss=F.mse_loss(pred.float(),x[4])
            if not bool(torch.isfinite(loss)):raise FloatingPointError((arm,epoch,start,'loss'))
            (loss*len(ix)/BATCH).backward()
            if start//BATCH in (0,1000,2000,2905):
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError((arm,epoch,start,'gradient'))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            sse+=float(loss.detach())*len(ix)
            next_start=min(start+BATCH,N_TRAIN)
            if next_start%(BATCH*500)==0:
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
                atomic_torch(resume,dict(arm=arm,epoch=epoch,next_start=next_start,sse=sse,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=rng,logs=logs,config_sha256=sha(HERE/'P114_EF_CONFIG.json')))
                print(json.dumps(dict(arm=arm,epoch=epoch,steps=(epoch-1)*STEPS+next_start//BATCH)),flush=True)
        scheduler.step();torch.cuda.synchronize()
        rec=dict(arm=arm,epoch=epoch,optimizer_steps=epoch*STEPS,train_samples=N_TRAIN,
            train_mse=sse/N_TRAIN,train_seconds=time.perf_counter()-tic,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            lr=optimizer.param_groups[0]['lr'],validation_used=False)
        logs.append(rec)
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        atomic_torch(resume,dict(arm=arm,epoch=epoch+1,next_start=0,sse=0.0,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng,logs=logs,config_sha256=sha(HERE/'P114_EF_CONFIG.json')))
        if epoch in (2,4,6):
            atomic_torch(HERE/f'{arm}_epoch{epoch}.pt',dict(arm=arm,epoch=epoch,step=epoch*STEPS,
                model=model.state_dict(),config_sha256=sha(HERE/'P114_EF_CONFIG.json'),
                initial_state_sha256=qa['initial_41_state_sha256']))
        write_csv(HERE/f'{arm}_TRAINING_LOG.csv',logs)
        print(json.dumps(rec),flush=True)
    final=HERE/f'{arm}_epoch6.pt'
    assert final.exists() and len(logs)==6
    atomic_json(done,dict(arm=arm,epochs_completed=6,steps=17436,
        final_sha256=sha(final),config_sha256=sha(HERE/'P114_EF_CONFIG.json'),
        initial_state_sha256=qa['initial_41_state_sha256'],test_accessed=False))
    print(json.dumps(dict(arm=arm,training_complete=True,sha256=sha(final))),flush=True)


def metric(y,p):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);e=p-y
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),
        r2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else None,
        bias=float(np.mean(e)),prediction_mean=float(np.mean(p)),prediction_std=float(np.std(p)))


def masks(y,category):
    return dict(Overall=np.ones(len(y),bool),Q0_Q50=y<=Q50,Q50_Q90=(y>Q50)&(y<=Q90),
        Q90_plus=y>Q90,Q99_plus=y>Q99,
        AGB_0_50=y<50,AGB_50_100=(y>=50)&(y<100),AGB_100_200=(y>=100)&(y<200),
        AGB_200_plus=y>=200,DN_normal=category==0,DN_partial_1=category==1,DN_whole_1=category==2)


def evaluate(arm,epoch):
    cfg,qa=assert_preflight();assert arm in ARMS and epoch in (2,4,6)
    if arm=='M1_real':assert json.loads((HERE/'M0_GATE.json').read_text(encoding='utf-8'))['status']=='M0_REPRODUCTION_PASS'
    ckpath=HERE/f'{arm}_epoch{epoch}.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['arm']==arm and ck['epoch']==epoch and ck['step']==epoch*STEPS
    assert ck['config_sha256']==sha(HERE/'P114_EF_CONFIG.json')
    model,_=fresh_pair();model.load_state_dict(ck['model'],strict=True);model=model.cuda().eval()
    data=OfficialTrainData();val=OfficialValData(data);raw=alos_cache('val')
    y=np.asarray(val.y,np.float32);meta=np.asarray(val.meta,np.float32)
    category=np.memmap(PILOT/'val_dn1_category.u8',mode='r',dtype=np.uint8,shape=(N_VAL,))
    pred=np.empty(N_VAL,np.float32)
    tic=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,N_VAL,BATCH):
            stop=min(start+BATCH,N_VAL);ix=np.arange(start,stop,dtype=np.int64)
            x=val.batch(start,stop);radar=radar_batch(arm,raw,ix)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(*x[:4],radar)
            pred[start:stop]=out.float().cpu().numpy()
    assert np.isfinite(pred).all()
    out=HERE/f'{arm}_epoch{epoch}_VAL_PREDICTIONS.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(('row','y','latitude','longitude','dn1_category','pred'))
        for i in range(N_VAL):w.writerow((i,float(y[i]),float(meta[i,2]),float(meta[i,3]),int(category[i]),float(pred[i])))
    rows=[dict(arm=arm,epoch=epoch,subset=key,**metric(y[mask],pred[mask])) for key,mask in masks(y,category).items()]
    write_csv(HERE/f'{arm}_epoch{epoch}_VAL_METRICS.csv',rows)
    atomic_json(HERE/f'{arm}_epoch{epoch}_VAL_LOCK.json',dict(arm=arm,epoch=epoch,rows=N_VAL,
        checkpoint_sha256=sha(ckpath),prediction_sha256=sha(out),evaluation_seconds=time.perf_counter()-tic,test_accessed=False))
    val.close()
    print(json.dumps(dict(arm=arm,epoch=epoch,overall_rmse=rows[0]['rmse'],q90_rmse=next(x['rmse'] for x in rows if x['subset']=='Q90_plus'))),flush=True)


def assess_m0():
    cfg,qa=assert_preflight()
    for epoch in (2,4,6):
        assert (HERE/f'M0_zero_epoch{epoch}_VAL_LOCK.json').exists()
    rows=[]
    for epoch in (2,4,6):
        rows.extend(csv.DictReader((HERE/f'M0_zero_epoch{epoch}_VAL_METRICS.csv').open(newline='',encoding='utf-8')))
    final=next(r for r in rows if r['epoch']=='6' and r['subset']=='Overall')
    gap=float(final['rmse'])-57.00143377449156
    passed=abs(gap)<=.30
    decision=dict(status='M0_REPRODUCTION_PASS' if passed else 'M0_REPRODUCTION_FAIL',
        m0_epoch6_rmse=float(final['rmse']),formal_n5_rmse=57.00143377449156,
        gap_m0_minus_n5=gap,max_abs_gap=.30,m0_final_checkpoint_sha256=sha(HERE/'M0_zero_epoch6.pt'),
        test_accessed=False)
    atomic_json(HERE/'M0_GATE.json',decision)
    lines=['# M0 零雷达早期融合复现','',
        'M0 与 M1 从同一未训练 seed43 初始化状态构建；此报告仅核验 M0，不依据其结果更改训练设置。','',
        '| 固定 epoch | Overall RMSE | Q90+ RMSE | Train MSE |','|---:|---:|---:|---:|']
    logs={int(r['epoch']):r for r in csv.DictReader((HERE/'M0_zero_TRAINING_LOG.csv').open(newline='',encoding='utf-8'))}
    for ep in (2,4,6):
        overall=next(r for r in rows if int(r['epoch'])==ep and r['subset']=='Overall')
        high=next(r for r in rows if int(r['epoch'])==ep and r['subset']=='Q90_plus')
        lines.append(f"| {ep} | {float(overall['rmse']):.4f} | {float(high['rmse']):.4f} | {float(logs[ep]['train_mse']):.2f} |")
    lines.extend(['',f"正式 N5 历史 epoch6 Validation RMSE 57.0014；M0 epoch6 差值（M0−N5）为 {gap:+.4f} Mg/ha，允许绝对差 ≤0.30。",
        '',f"**判定：{decision['status']}。**"+(" 可进入 M1 固定训练。" if passed else " 停止 M1，先排查实现差异；不得调参追分。"),
        '',f"M0 epoch6 SHA-256 `{decision['m0_final_checkpoint_sha256']}`。Test 未访问。"])
    (HERE/'M0_REPRODUCTION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('preflight','train','evaluate','assess_m0'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--epoch',type=int,choices=(2,4,6))
    args=parser.parse_args()
    if args.phase=='preflight':preflight()
    elif args.phase=='train':
        assert args.arm is not None;train(args.arm)
    elif args.phase=='evaluate':
        assert args.arm is not None and args.epoch is not None;evaluate(args.arm,args.epoch)
    else:assess_m0()
