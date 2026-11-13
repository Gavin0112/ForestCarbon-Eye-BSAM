"""Time-budgeted, paired BSAM/plain-MSE experiment on official Train/Validation.

Test is intentionally absent. One GPU arm runs at a time. Checkpoints are isolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "mse_film_main_20261022"),
                str(EXP / "mse_film_pbsam_20261023"),
                str(EXP / "mse_film_bsam_20261023")]
import run_mse_film as R
from mse_film_model import MSEFilm
from bsam_step import BalancedSharpnessStep, BatchNormSecondPass

SOURCE = EXP / "mse_film_bsam_20261023/MSE_FILM_BSAM_CONFIG.json"
REF_VAL = EXP / "mse_film_convergence_sfr_20261022/MSE_FILM_EPOCH7_VAL_PREDICTIONS.csv"
PROTOCOL = HERE / "LOCKED_PROTOCOL.json"
BATCH = 128
Q75 = 89.13347625732422


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_pt(path, state):
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def tensor_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def new_model(seed, source):
    R.SEED = seed
    R.seed_all()
    c = source["original_input_config"]
    return MSEFilm(c["dem_mean_m"], c["dem_std_m"]).cuda()


def ready():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert source["official_train_rows"] == R.N_TRAIN == 371849
    assert source["official_validation_rows"] == R.N_VAL == 103550
    assert source["batch"] == BATCH and source["rho"] == 0.2
    assert source["lr"] == 1e-4
    return source


def update(model, opt, batch, radar, dn, arm, weights, stepper, bn):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(*batch[:4], radar, dn)
    residual = (out.float() - batch[4].float()).square()
    raw_mse = residual.mean()
    assert bool(torch.isfinite(raw_mse))
    scale = len(batch[4]) / BATCH
    if arm == "bsam":
        ids = torch.clamp((batch[4].float() / 10.).floor().long(), 0, 49)
        weighted = (residual * weights[ids]).mean()
        (weighted * scale).backward()
        stepper.perturb()
        bn.freeze_running_update()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out2 = model(*batch[:4], radar, dn)
            second = F.mse_loss(out2.float(), batch[4].float())
            assert bool(torch.isfinite(second))
            (second * scale).backward()
        finally:
            bn.restore_running_update()
            stepper.restore()
    else:
        (raw_mse * scale).backward()
    opt.step()
    return float(raw_mse.detach())


@torch.inference_mode()
def evaluate(model, val, val_radar, source):
    model.eval()
    out = np.empty(R.N_VAL, np.float32)
    tic = time.perf_counter()
    for start in range(0, R.N_VAL, 64):
        stop = min(start + 64, R.N_VAL)
        x = val.batch(start, stop)
        a, dn = R.radar(val_radar, np.arange(start, stop, dtype=np.int64))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(*x[:4], a, dn).float()
        assert bool(torch.isfinite(p).all())
        out[start:stop] = p.cpu().numpy()
    y = np.asarray(val.y, np.float32)
    assert np.array_equal(source.agb_true.to_numpy(np.float32), y)
    metric = R.metric(y, out)
    e = out.astype(np.float64) - y.astype(np.float64)
    metric["overall"]["r2"] = float(1 - np.square(e).sum() / np.square(y-y.mean()).sum())
    high = y > R.Q["q90"]
    metric["q90_plus"]["residual_variance"] = float(np.var(e[high], ddof=0))
    mid = (y > R.Q["q50"]) & (y <= Q75)
    metric["q50_q75"] = {"n": int(mid.sum()), "rmse": float(np.sqrt(np.mean(e[mid]**2)))}
    metric["validation_seconds"] = time.perf_counter() - tic
    return metric, out


def get_state(model, opt, sched, epoch, start, steps, train_s, sse, seen, logs, best):
    return dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                rng=R.rng_state(), epoch=epoch, start=start, steps=steps,
                train_seconds=train_s, sse=sse, seen=seen, logs=logs, best=best,
                protocol_sha256=sha(PROTOCOL))


def run_arm(seed, arm):
    source = ready()
    lock = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert lock["status"] == "FROZEN" and lock["source_config_sha256"] == sha(SOURCE)
    assert seed in lock["seeds"] and arm in ("plain", "bsam")
    folder = HERE / f"seed{seed}" / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COMPLETE.json").exists():
        raise RuntimeError(f"Completed arm must not be rerun: {folder}")
    model = new_model(seed, source)
    initial_sha = tensor_sha(model)
    assert sum(p.numel() for p in model.parameters()) == 2123367
    init_file = HERE / f"seed{seed}" / "INITIAL_SHA.json"
    if init_file.exists():
        assert json.loads(init_file.read_text())["sha256"] == initial_sha
    else:
        write_json(init_file, {"seed": seed, "sha256": initial_sha})
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=.1)
    stepper = BalancedSharpnessStep(model, opt, 0.2) if arm == "bsam" else None
    bn = BatchNormSecondPass(model) if arm == "bsam" else None
    weights = torch.tensor(source["bin_weights"], device="cuda", dtype=torch.float32)
    data = R.OfficialTrainData()
    R.SEED = seed
    assert R.config(data)["normalization_sha256"] == source["original_input_config"]["normalization_sha256"]
    assert len(data.y) == R.N_TRAIN
    val = R.OfficialValData(data)
    source_val = pd.read_csv(REF_VAL)
    assert len(source_val) == R.N_VAL and source_val.original_val_row.is_unique
    assert np.array_equal(source_val.original_val_row.to_numpy(np.int64), np.arange(R.N_VAL))
    train_radar, val_radar = R.alos_cache("train"), R.alos_cache("val")
    assert np.array_equal(source_val.agb_true.to_numpy(np.float32), np.asarray(val.y, np.float32))
    logs = []; best = {"rmse": float("inf"), "node": None}
    epoch, start, steps, train_s, sse, seen = 1, 0, 0, 0., 0., 0
    last = folder / "last.pt"
    if last.exists():
        z = torch.load(last, map_location="cpu", weights_only=False)
        assert z["protocol_sha256"] == sha(PROTOCOL)
        model.load_state_dict(z["model"], strict=True)
        opt.load_state_dict(z["optimizer"]); sched.load_state_dict(z["scheduler"])
        R.set_rng(z["rng"])
        epoch, start, steps = z["epoch"], z["start"], z["steps"]
        train_s, sse, seen = z["train_seconds"], z["sse"], z["seen"]
        logs, best = z["logs"], z["best"]
        print(f"Resume {seed}/{arm} epoch {epoch}, row {start}, steps {steps}", flush=True)
    budget = float(lock["train_seconds_budget"])
    step_time_estimate = float(lock["step_time_estimate"][arm])
    train_radar = R.alos_cache("train")
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(node):
        nonlocal best, sse, seen
        metrics, pred = evaluate(model, val, val_radar, source_val)
        record = dict(seed=seed, arm=arm, node=node, epoch=epoch, updates=steps,
                      samples=seen, train_seconds=train_s, lr=float(opt.param_groups[0]["lr"]),
                      train_mse=(sse / seen if seen else None),
                      peak_mib=float(torch.cuda.max_memory_allocated()/2**20),
                      validation=metrics)
        logs.append(record)
        write_json(folder / "VALIDATION_NODES.json", logs)
        if metrics["overall"]["rmse"] < best["rmse"]:
            best = {"rmse": metrics["overall"]["rmse"], "node": node, "steps": steps}
            save_pt(folder / "best.pt", get_state(model,opt,sched,epoch,start,steps,train_s,sse,seen,logs,best))
            pd.DataFrame(dict(original_val_row=source_val.original_val_row.to_numpy(np.int64),
                grid_1deg=source_val.grid_1deg.to_numpy(),
                agb_true=np.asarray(val.y, np.float32), prediction=pred)).to_csv(
                    folder / "best_validation_predictions.csv", index=False)
        save_pt(folder / f"node_{steps:05d}.pt",
                get_state(model,opt,sched,epoch,start,steps,train_s,sse,seen,logs,best))
        print(json.dumps(record, ensure_ascii=False), flush=True)

    while train_s + step_time_estimate < budget:
        model.train()
        order = np.random.default_rng(seed * 101 + epoch).permutation(data.all_rows)
        for cursor in range(start, R.N_TRAIN, BATCH):
            if train_s + step_time_estimate >= budget:
                break
            tick = time.perf_counter()
            ix = order[cursor:cursor+BATCH]
            x = data.batch(ix); a, dn = R.radar(train_radar, ix)
            loss = update(model,opt,x,a,dn,arm,weights,stepper,bn)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            train_s += elapsed
            step_time_estimate = 0.95*step_time_estimate + 0.05*elapsed
            steps += 1; seen += len(ix); sse += loss*len(ix)
            start = min(cursor+BATCH, R.N_TRAIN)
            if steps % 500 == 0:
                save_pt(last,get_state(model,opt,sched,epoch,start,steps,train_s,sse,seen,logs,best))
                print(json.dumps({"seed":seed,"arm":arm,"steps":steps,"train_seconds":train_s}),flush=True)
        if start == R.N_TRAIN:
            checkpoint(f"epoch_{epoch}")
            sched.step()
            epoch += 1; start = 0
            save_pt(last,get_state(model,opt,sched,epoch,start,steps,train_s,sse,seen,logs,best))
        else:
            break
    if not logs or logs[-1]["updates"] != steps:
        checkpoint("budget_final")
    save_pt(last,get_state(model,opt,sched,epoch,start,steps,train_s,sse,seen,logs,best))
    write_json(folder/"COMPLETE.json", dict(status="COMPLETE",seed=seed,arm=arm,
        steps=steps,samples=seen,train_seconds=train_s,budget=budget,
        best=best,initial_sha256=initial_sha,best_sha256=sha(folder/"best.pt"),
        protocol_sha256=sha(PROTOCOL),test_access=False))
    val.close()


def calibrate():
    source = ready()
    data = R.OfficialTrainData()
    train_radar = R.alos_cache("train")
    times = {}
    for arm in ("plain", "bsam"):
        model = new_model(43,source).train()
        opt = torch.optim.Adam(model.parameters(),lr=1e-4)
        stepper = BalancedSharpnessStep(model,opt,0.2) if arm=="bsam" else None
        bn = BatchNormSecondPass(model) if arm=="bsam" else None
        weights = torch.tensor(source["bin_weights"],device="cuda",dtype=torch.float32)
        order = np.random.default_rng(4344).permutation(data.all_rows)
        elapsed=[]
        for k in range(24):
            ix=order[k*BATCH:(k+1)*BATCH]
            tick=time.perf_counter()
            x=data.batch(ix);a,dn=R.radar(train_radar,ix)
            update(model,opt,x,a,dn,arm,weights,stepper,bn)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-tick)
        times[arm]=dict(steps=24,mean=float(np.mean(elapsed[4:])),
                        median=float(np.median(elapsed[4:])),
                        min=float(np.min(elapsed[4:])),max=float(np.max(elapsed[4:])))
        del model,opt,stepper,bn
        torch.cuda.empty_cache()
    write_json(HERE/"CURRENT_HARDWARE_CALIBRATION.json",
               dict(gpu=torch.cuda.get_device_name(0),measurements=times,
                    note="24 real Train steps per arm; first four excluded from summary; models discarded"))
    print(json.dumps(times,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("calibrate","train"))
    p.add_argument("--seed",type=int,choices=(43,44,45))
    p.add_argument("--arm",choices=("plain","bsam"))
    a=p.parse_args()
    if a.command=="calibrate": calibrate()
    else:
        assert a.seed is not None and a.arm is not None
        run_arm(a.seed,a.arm)
