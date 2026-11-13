"""Reproducible matched six-epoch N1 pilot on frozen P95 inner_fit."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import N1NicoNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P95 = ROOT / "experiments/agbd_lite/p95_geographic_multimodal_fusion"
P95DATA = P95 / "p95_1"
N = 371849
SEED = 43
ORDER_SEED = 4343
BATCH = 128
ACCUM = 1
EPOCHS = 6
LR = 1e-4
DEVICE = "cuda"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(path, state):
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(state, temp)
    temp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class Data:
    def __init__(self):
        stats = json.loads((P95DATA / "P95_1_NORMALIZATION.json").read_text())
        self.stats = stats
        self.s2 = np.memmap(P95DATA / "train_s2_spatial_norm.f32", mode="r", dtype=np.float32, shape=(N, 12, 25, 25))
        self.dem = np.memmap(ROOT / "experiments/agbd_lite/p19_2_dem_pilot/run_aw3d30/train_dem.i16", mode="r", dtype=np.int16, shape=(N, 25, 25))
        self.lc = np.memmap(ROOT / "experiments/agbd_lite/p48_lc_multimodal_pilot/train_lc.u8", mode="r", dtype=np.uint8, shape=(N, 25, 25, 2))
        self.meta = np.memmap(ROOT / "experiments/agbd_lite/p54_final_feature_alignment/train_features_raw.f32", mode="r", dtype=np.float32, shape=(N, 8))
        self.y = np.memmap(ROOT / "data/agbd_lite/cache_s2only/train_y.f32", mode="r", dtype=np.float32, shape=(N,))
        self.lut = np.full(256, len(stats["lc_classes"]), np.int64)
        for i, code in enumerate(stats["lc_classes"]):
            self.lut[code] = i
        self.eye = np.eye(23, dtype=np.float32)
        self.eye[-1] = 0  # unknown class is not a valid one-hot class
        roles = {"inner_fit": [], "inner_validation": [], "spatial_dev": []}
        components = []
        with (P95 / "P95_0_SPATIAL_SPLIT.csv").open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                assert int(row["original_train_row"]) == i
                role = row["inner_role"] or row["role"]
                if role in roles:
                    roles[role].append(i)
                if role == "spatial_dev":
                    components.append(int(row["component_id"]))
        assert [len(roles[k]) for k in roles] == [255972, 51027, 64850]
        self.roles = {k: np.asarray(v, np.int64) for k, v in roles.items()}
        self.components = np.asarray(components, np.int16)
        self.q = np.quantile(self.y[self.roles["inner_fit"]], [.5, .9, .95])

    def batch(self, rows):
        rows = np.asarray(rows, np.int64)
        s2 = np.asarray(self.s2[rows], np.float32)
        dsm = ((np.asarray(self.dem[rows], np.float32) - self.stats["dem_pixel_mean"])
               / self.stats["dem_pixel_std"])[:, None]
        raw = np.asarray(self.lc[rows])
        cls, prob = raw[..., 0], raw[..., 1]
        valid_cls, valid_prob = cls != 255, prob != 255
        onehot = self.eye[self.lut[cls]].transpose(0, 3, 1, 2)
        lc = np.concatenate((onehot,
            valid_cls[:, None].astype(np.float32),
            (np.where(valid_prob, prob, 0).astype(np.float32) / 100)[:, None],
            valid_prob[:, None].astype(np.float32)), 1)
        # P95 uses 22 classes + unknown channel + 3 ancillary channels.
        meta = np.asarray(self.meta[rows], np.float32).copy()
        meta[:, 2:] = ((meta[:, 2:] - self.stats["metadata_mean_columns_2_7"])
                       / self.stats["metadata_std_columns_2_7"])
        y = np.asarray(self.y[rows], np.float32)
        return tuple(torch.from_numpy(np.ascontiguousarray(z)).to(DEVICE) for z in (s2, dsm, lc, meta, y))


def new_models():
    seed_all()
    m0 = N1NicoNet(False)
    m1 = N1NicoNet(True)
    m1.backbone.load_state_dict(copy.deepcopy(m0.backbone.state_dict()))
    return m0, m1


def gate0(data):
    m0, m1 = new_models()
    assert sum(p.numel() for p in m0.parameters()) == 1861762
    assert sum(p.numel() for p in m1.parameters()) < 2000000
    m0.to(DEVICE); m1.to(DEVICE)
    x = data.batch(data.roles["inner_fit"][:BATCH])
    m0.eval(); m1.eval()
    with torch.no_grad():
        p0 = m0(*x[:4]); p1, gd, gl = m1(*x[:4], return_gates=True)
    diff = float((p0-p1).abs().max())
    assert p0.shape == (BATCH,) and gd.shape == gl.shape == (BATCH, 1)
    assert diff <= 1e-5 and bool((gd == 1).all()) and bool((gl == 1).all())
    m1.train()
    pred = m1(*x[:4]); loss = F.mse_loss(pred, x[4]); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m1.gate.parameters())
    assert m1.gate.dsm_gate[-1].weight.grad.abs().sum() > 0
    assert m1.gate.lc_gate[-1].weight.grad.abs().sum() > 0
    opt = torch.optim.Adam(m1.parameters(), lr=LR)
    opt.step(); opt.zero_grad(set_to_none=True)
    F.mse_loss(m1(*x[:4]), x[4]).backward()
    assert m1.gate.s2_projection.weight.grad.abs().sum() > 0
    assert m1.gate.dsm_projection.weight.grad.abs().sum() > 0
    assert m1.gate.lc_projection.weight.grad.abs().sum() > 0
    report = dict(m0_params=sum(p.numel() for p in m0.parameters()),
                  m1_params=sum(p.numel() for p in m1.parameters()), initial_max_abs_diff=diff,
                  gate_initial=1, forward_backward="PASS", projection_gradient_after_step="PASS",
                  peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
    (HERE / "N1_GATE0.json").write_text(json.dumps(report, indent=2))
    del m0, m1
    torch.cuda.empty_cache()
    return report


def metrics(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    e = p.astype(np.float64)-y.astype(np.float64)
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(e*e))),
                mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)) if len(y)>1 else float("nan"))


@torch.no_grad()
def predict(model, data, rows, gates=False):
    model.eval()
    pp, gg, yy = [], [], []
    for start in range(0, len(rows), BATCH):
        x = data.batch(rows[start:start+BATCH])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(*x[:4], return_gates=gates)
        if gates:
            p, gd, gl = result
            gg.append(np.stack((gd.float().cpu().numpy()[:,0], gl.float().cpu().numpy()[:,0]), 1))
        else:
            p = result
        pp.append(p.float().cpu().numpy()); yy.append(x[4].cpu().numpy())
    return np.concatenate(yy), np.concatenate(pp), np.concatenate(gg) if gates else None


def train_arm(arm, data, initial, config):
    model = N1NicoNet(arm == "M1")
    model.backbone.load_state_dict(initial)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=.1)
    folder = HERE / arm; folder.mkdir(exist_ok=True)
    resume = folder / "resume.pt"
    rows_log = []; best = math.inf; best_epoch = None; start_epoch = 1; step = 0
    seed_all()
    if resume.exists():
        s = torch.load(resume, map_location="cpu", weights_only=False)
        assert s["config"] == config and s["arm"] == arm
        model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"]); set_rng(s["rng"])
        rows_log=s["rows"]; best=s["best"]; best_epoch=s["best_epoch"]
        start_epoch=s["epoch"]+1; step=s["step"]
    for epoch in range(start_epoch, EPOCHS+1):
        model.train(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
        order=np.random.default_rng(ORDER_SEED+epoch).permutation(data.roles["inner_fit"])
        opt.zero_grad(set_to_none=True); total_sse=0.; seen=0
        for start in range(0, len(order), BATCH):
            indices=order[start:start+BATCH]
            if len(indices)==0: continue
            x=data.batch(indices)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p=model(*x[:4]); loss=F.mse_loss(p.float(), x[4])
            if not torch.isfinite(loss): raise FloatingPointError(f"{arm} epoch {epoch} batch {start}")
            (loss*len(indices)/(BATCH*ACCUM)).backward()
            total_sse += float(loss.detach())*len(indices); seen+=len(indices)
            if (start//BATCH+1)%ACCUM==0 or start+BATCH>=len(order):
                opt.step(); opt.zero_grad(set_to_none=True); step+=1
        assert seen==255972
        scheduler.step(); torch.cuda.synchronize(); train_sec=time.perf_counter()-t0
        row=dict(arm=arm, epoch=epoch, steps=step, train_mse=total_sse/seen,
                 train_seconds=train_sec, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                 lr=opt.param_groups[0]["lr"])
        if epoch in (2,4,6):
            t0=time.perf_counter(); y,p,_=predict(model,data,data.roles["inner_validation"])
            row.update(val_rmse=metrics(y,p)["rmse"], val_seconds=time.perf_counter()-t0)
            if row["val_rmse"] < best:
                best=row["val_rmse"]; best_epoch=epoch
                atomic_save(folder/"best.pt",dict(model=model.state_dict(),epoch=epoch,config=config))
        else:
            row.update(val_rmse="",val_seconds="")
        rows_log.append(row)
        atomic_save(resume,dict(arm=arm,config=config,model=model.state_dict(),
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),
            epoch=epoch,step=step,rows=rows_log,best=best,best_epoch=best_epoch))
        save_csv(folder/"epoch_metrics.csv",rows_log)
        save_csv(HERE/"N1_MATCHED_TRAINING_LOG.csv",
                 [*read_csv(HERE/"M0/epoch_metrics.csv"),*read_csv(HERE/"M1/epoch_metrics.csv")])
        print(json.dumps(row),flush=True)


def read_csv(path):
    if not path.exists(): return []
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))


def evaluate(data, config):
    checkpoints={arm: HERE/arm/"best.pt" for arm in ("M0","M1")}
    lock={arm:sha(path) for arm,path in checkpoints.items()}
    lock_path=HERE/"N1_PRE_DEV_LOCK.json"
    if lock_path.exists(): assert json.loads(lock_path.read_text())==lock
    else: lock_path.write_text(json.dumps(lock,indent=2))
    predictions={}; gatevals={}; internal=[]; spatial=[]; gate_rows=[]; efficiency=[]
    for arm in ("M0","M1"):
        state=torch.load(checkpoints[arm],map_location="cpu",weights_only=False)
        assert state["config"]==config
        model=N1NicoNet(arm=="M1").to(DEVICE)
        model.load_state_dict(state["model"])
        for role in ("inner_validation","spatial_dev"):
            y,p,g=predict(model,data,data.roles[role],arm=="M1")
            predictions[(arm,role)]=(y,p)
            if g is not None: gatevals[role]=g
            target=internal if role=="inner_validation" else spatial
            masks={"overall":np.ones(len(y),bool),"q0_q50":y<data.q[0],
                   "q50_q90":(y>=data.q[0])&(y<data.q[1]),
                   "q90_q95":(y>=data.q[1])&(y<data.q[2]),"q90_plus":y>=data.q[1]}
            for name,mask in masks.items(): target.append(dict(arm=arm,group=name,**metrics(y,p,mask)))
            if role=="spatial_dev":
                for comp in np.unique(data.components):
                    target.append(dict(arm=arm,group=f"cluster_{comp}",**metrics(y,p,data.components==comp)))
        sample=data.batch(data.roles["inner_fit"][:8]); model.eval()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            for _ in range(10): model(*sample[:4])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
            for _ in range(50): model(*sample[:4])
            torch.cuda.synchronize()
        efficiency.append(dict(arm=arm,params=sum(p.numel() for p in model.parameters()),
                               batch8_ms=(time.perf_counter()-t0)*1000/50,
                               peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                               precision="BF16",preprocessing_included=False))
        del model; torch.cuda.empty_cache()
    for role in ("inner_validation","spatial_dev"):
        y=predictions[("M1",role)][0]; assert np.array_equal(y,predictions[("M0",role)][0])
        g=gatevals[role]
        for name,mask in [("overall",np.ones(len(y),bool)),
                          ("q0_q50",y<data.q[0]),("q50_q90",(y>=data.q[0])&(y<data.q[1])),
                          ("q90_plus",y>=data.q[1])]:
            gate_rows.append(dict(split=role,group=name,n=int(mask.sum()),
                dsm_mean=float(g[mask,0].mean()),dsm_std=float(g[mask,0].std()),
                lc_mean=float(g[mask,1].mean()),lc_std=float(g[mask,1].std())))
        if role=="spatial_dev":
            for comp in np.unique(data.components):
                mask=data.components==comp
                gate_rows.append(dict(split=role,group=f"cluster_{comp}",n=int(mask.sum()),
                    dsm_mean=float(g[mask,0].mean()),dsm_std=float(g[mask,0].std()),
                    lc_mean=float(g[mask,1].mean()),lc_std=float(g[mask,1].std())))
    save_csv(HERE/"N1_INTERNAL_VALIDATION.csv",internal)
    save_csv(HERE/"N1_SPATIAL_DEV_RESULTS.csv",spatial)
    save_csv(HERE/"N1_MODALITY_GATE_ANALYSIS.csv",gate_rows)
    save_csv(HERE/"N1_EFFICIENCY.csv",efficiency)
    (HERE/"N1_POST_DEV_LOCK.json").write_text(json.dumps(lock,indent=2))
    return internal,spatial,gate_rows,efficiency


def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("gate0","train","evaluate","all"))
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark=True
    data=Data()
    config=dict(seed=SEED,order_seed=ORDER_SEED,batch=BATCH,accum=ACCUM,epochs=EPOCHS,
        lr=LR,scheduler="StepLR(30,0.1)",loss="MSE",precision="BF16 AMP, FP32 loss",
        split_sha256=sha(P95/"P95_0_SPATIAL_SPLIT.csv"),
        normalization_sha256=sha(P95DATA/"P95_1_NORMALIZATION.json"),
        model_sha256=sha(HERE/"model.py"))
    config_path=HERE/"N1_FROZEN_CONFIG.json"
    if config_path.exists():
        assert json.loads(config_path.read_text())==config
    else:
        config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    if args.phase in ("gate0","all","train"):
        if not (HERE/"N1_GATE0.json").exists(): print(gate0(data),flush=True)
    if args.phase in ("train","all"):
        initial=new_models()[0].backbone.state_dict()
        for arm in ("M0","M1"): train_arm(arm,data,initial,config)
    if args.phase in ("evaluate","all"): evaluate(data,config)


if __name__=="__main__": main()
