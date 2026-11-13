"""Official Train six-epoch matched N1/N4/N5 run, with no Validation selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
N1_DIR=HERE.parent/'n1_niconet_mrf'
N4_DIR=HERE.parent/'n4_niconet_sscf'
N5_DIR=HERE.parent/'n5_capacity_control'
P95=ROOT/'experiments/agbd_lite/p95_geographic_multimodal_fusion/p95_1'
sys.path.insert(0,str(N4_DIR));sys.path.insert(0,str(N5_DIR))
import run_n4 as N4
from n4_model import NicoNet_SSCF
from n5_model import NicoNet_CapacityControl
from n6_data import OfficialTrainData,N_TRAIN

SEED,ORDER_SEED,BATCH,EPOCHS,LR=43,4343,128,6,1e-4
EXPECTED_STEPS=math.ceil(N_TRAIN/BATCH)
MODELS={'M0':lambda:N4.N1.N1NicoNet(False),
        'M1':NicoNet_SSCF,'M2':NicoNet_CapacityControl}
PARAMS={'M0':1861762,'M1':2078147,'M2':2077991}


def fresh_model(arm):
    N4.N1.seed_all()
    model=MODELS[arm]()
    backbone=model.state_dict() if arm=='M0' else model.base.state_dict()
    assert N4.state_sha(backbone)=='111cad3d743058bb68a6ce14897e24f62adb4e7d84e65b2103be6eba79c47a9b'
    assert sum(p.numel() for p in model.parameters())==PARAMS[arm]
    return model


def fixed_config(data):
    assert EXPECTED_STEPS==2906 and EXPECTED_STEPS*EPOCHS==17436
    obj=dict(seed=SEED,order_seed=ORDER_SEED,epochs=EPOCHS,
        train_n=N_TRAIN,batch=BATCH,drop_last=False,steps_per_epoch=EXPECTED_STEPS,
        optimizer_steps=EXPECTED_STEPS*EPOCHS,last_batch_n=N_TRAIN%BATCH,
        last_batch_loss_factor=(N_TRAIN%BATCH)/BATCH,
        optimizer='Adam',lr=LR,scheduler='StepLR(30,0.1) by epoch',
        loss='centre-pixel MSE',precision='BF16 AMP, FP32 loss',augmentation='none',
        checkpoint_rule='fixed epoch 6; official Validation never selects epoch',
        epoch_checkpoints=[2,4,6],
        shared_initial_state_sha256='111cad3d743058bb68a6ce14897e24f62adb4e7d84e65b2103be6eba79c47a9b',
        input_normalization='P95 inner-fit-fitted Train-only statistics; reused unchanged for all official Train rows and applied to official Validation',
        normalization_sha256=N4.sha(P95/'P95_1_NORMALIZATION.json'),
        train_s2_cache_bytes=(P95/'train_s2_spatial_norm.f32').stat().st_size,
        n1_model_sha256=N4.sha(N1_DIR/'model.py'),
        n4_model_sha256=N4.sha(N4_DIR/'n4_model.py'),
        n5_model_sha256=N4.sha(N5_DIR/'n5_model.py'),
        data_sha256=N4.sha(HERE/'n6_data.py'),
        run_sha256=N4.sha(__file__),
        train_quantiles=dict(zip(('q50','q90','q95','q99'),map(float,data.q_official))),
        arm_params=PARAMS,gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    path=HERE/'N6_FROZEN_CONFIG.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==obj
    else:
        path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return obj


def gate0(data):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert len(data.all_rows)==N_TRAIN
    assert np.isfinite(data.q_official).all()
    models={arm:fresh_model(arm).cuda() for arm in MODELS}
    batch=data.batch(np.arange(BATCH,dtype=np.int64))
    for model in models.values():model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictions={arm:model(*batch[:4]).float() for arm,model in models.items()}
    differences={arm:float((predictions[arm]-predictions['M0']).abs().max()) for arm in ('M1','M2')}
    assert all(v<=1e-5 for v in differences.values())
    gradients={}
    for arm in ('M1','M2'):
        model=models[arm];model.train();model.zero_grad(set_to_none=True)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=F.mse_loss(model(*batch[:4]).float(),batch[4])
        assert torch.isfinite(loss);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=F.mse_loss(model(*batch[:4]).float(),batch[4])
        assert torch.isfinite(loss);loss.backward()
        paths=([model.spectral,model.spatial] if arm=='M1' else [model.adapter.fc1])
        values=[N4.norm(path.parameters()) for path in paths]
        assert all(math.isfinite(v) and v>0 for v in values)
        gradients[arm]=values
    report=dict(status='PASS',train_n=N_TRAIN,steps_per_epoch=EXPECTED_STEPS,
        initial_max_abs_difference=differences,second_step_path_gradient_l2=gradients,
        train_label_quantiles=dict(zip(('q50','q90','q95','q99'),map(float,data.q_official))),
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        validation_opened=False,test_opened=False)
    (HERE/'N6_GATE0.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'gate0':report}),flush=True)


def unified_log():
    records=[]
    for arm in MODELS:
        path=HERE/f'{arm}_training.csv'
        if path.exists():
            records.extend(csv.DictReader(path.open(newline='',encoding='utf-8')))
    if records:N4.write_csv(HERE/'N6_TRAINING_LOG.csv',records)


def train_arm(arm,data,cfg):
    assert arm in MODELS
    if arm=='M2':
        decision=json.loads((HERE/'N6_M2_GATE.json').read_text(encoding='utf-8'))
        assert decision['train_m2'] is True
    model=fresh_model(arm).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=.1)
    resume=HERE/f'{arm}_resume.pt';logs=[];first_epoch=1;step=0
    N4.N1.seed_all()
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config']==cfg and state['arm']==arm
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);N4.set_rng(state['rng'])
        logs=state['logs'];first_epoch=state['epoch']+1;step=state['step']
        print(json.dumps({'arm':arm,'resumed_from_epoch':state['epoch']}),flush=True)
    for epoch in range(first_epoch,EPOCHS+1):
        model.train()
        if arm=='M1':model.capture=False
        torch.cuda.reset_peak_memory_stats();start_time=time.perf_counter()
        order=np.random.default_rng(ORDER_SEED+epoch).permutation(data.all_rows)
        optimizer.zero_grad(set_to_none=True);seen=0;sse=0.0
        for start in range(0,N_TRAIN,BATCH):
            indices=order[start:start+BATCH]
            batch=data.batch(indices)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(*batch[:4]);loss=F.mse_loss(output.float(),batch[4])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'{arm} nonfinite loss epoch={epoch},batch={start}')
            (loss*len(indices)/BATCH).backward()
            if start//BATCH in (0,1000,2000,2905):
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError(f'{arm} nonfinite gradient epoch={epoch},batch={start}')
            optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
            seen+=len(indices);sse+=float(loss.detach())*len(indices)
        assert seen==N_TRAIN and step==epoch*EXPECTED_STEPS
        scheduler.step();torch.cuda.synchronize()
        record=dict(arm=arm,epoch=epoch,optimizer_steps=step,train_samples=seen,
            train_mse=sse/seen,train_seconds=time.perf_counter()-start_time,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            lr=optimizer.param_groups[0]['lr'],validation_used=False)
        if epoch in (2,4,6):
            N4.atomic_save(HERE/f'{arm}_epoch{epoch}.pt',dict(arm=arm,epoch=epoch,
                step=step,model=model.state_dict(),config=cfg))
        logs.append(record)
        N4.atomic_save(resume,dict(arm=arm,epoch=epoch,step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=N4.rng_state(),logs=logs,config=cfg))
        N4.write_csv(HERE/f'{arm}_training.csv',logs)
        unified_log();print(json.dumps(record),flush=True)
    assert len(logs)==EPOCHS
    final=HERE/f'{arm}_epoch6.pt'
    (HERE/f'{arm}_TRAINING_LOCK.json').write_text(json.dumps(dict(arm=arm,
        epochs_completed=6,optimizer_steps=17436,final_sha256=N4.sha(final),
        config_sha256=N4.sha(HERE/'N6_FROZEN_CONFIG.json'),
        official_validation_evaluated=False),indent=2),encoding='utf-8')
    print(json.dumps({'arm':arm,'training_complete':True,'final_sha256':N4.sha(final)}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('gate0','train'))
    parser.add_argument('--arm',choices=list(MODELS))
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark=True
    data=OfficialTrainData()
    if args.phase=='gate0':gate0(data)
    else:
        assert args.arm is not None
        assert json.loads((HERE/'N6_GATE0.json').read_text(encoding='utf-8'))['status']=='PASS'
        train_arm(args.arm,data,fixed_config(data))


if __name__=='__main__':main()
