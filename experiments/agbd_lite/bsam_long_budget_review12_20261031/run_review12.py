"""Frozen Review-12: 14-epoch paired Plain MSE / BSAM, official Train and Val only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "bsam_fair_compute_review02_20261030"))
import run_fair as F

R = F.R
PROTOCOL = HERE / "FROZEN_PROTOCOL.json"
PREFLIGHT = HERE / "PREFLIGHT_REPORT.json"
EPOCHS = 14
STEPS = math.ceil(R.N_TRAIN / F.BATCH)


def atomic_json(path: Path, obj):
    F.write_json(path, obj)


def state(model, optimizer, scheduler, seed, arm, epoch, cursor, updates,
          train_seconds, epoch_sse, logs, best):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=R.rng_state(),
                seed=seed, arm=arm, epoch=epoch, cursor=cursor, updates=updates,
                train_seconds=train_seconds, epoch_sse=epoch_sse,
                logs=logs, best=best, protocol_sha256=F.sha(PROTOCOL))


def check_protocol():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert p["status"] == "FROZEN" and p["epochs"] == EPOCHS
    assert p["steps_per_epoch"] == STEPS and p["total_updates_per_arm"] == STEPS * EPOCHS
    assert p["batch"] == F.BATCH == 128 and p["seeds"] == [43, 44, 45]
    assert p["arms"] == ["plain", "bsam"]
    for path, expected in p["source_sha256"].items():
        assert F.sha(EXP / path) == expected, path
    return p


def preflight():
    p = check_protocol()
    source = F.ready()
    assert source["rho"] == 0.2 and source["lr"] == 1e-4
    assert source["bin_weights"] == json.loads(F.SOURCE.read_text(encoding="utf-8"))["bin_weights"]
    checks = []
    for seed in p["seeds"]:
        a = F.new_model(seed, source)
        h1 = F.tensor_sha(a)
        del a
        b = F.new_model(seed, source)
        h2 = F.tensor_sha(b)
        n_params = sum(x.numel() for x in b.parameters())
        assert h1 == h2 and n_params == 2123367
        del b
        order = np.random.default_rng(seed * 101 + 1).permutation(R.N_TRAIN)
        assert len(order) == R.N_TRAIN and np.unique(order).size == R.N_TRAIN
        checks.append(dict(seed=seed, paired_initial_tensor_sha256=h1,
                           parameters=n_params, first_epoch_first_rows=order[:8].tolist(),
                           first_epoch_last_rows=order[-8:].tolist()))
    data = R.OfficialTrainData()
    R.SEED = 43
    actual = R.config(data)
    assert actual["normalization_sha256"] == source["original_input_config"]["normalization_sha256"]
    assert len(data.y) == R.N_TRAIN == 371849
    val = R.OfficialValData(data)
    assert len(val.y) == R.N_VAL == 103550
    val.close()
    # Both arms import the same update function. Its branches differ only in the
    # previously frozen BSAM two-pass rule; optimizer and scheduler are shared.
    assert F.sha(EXP / "bsam_fair_compute_review02_20261030/run_fair.py") == p["source_sha256"]["bsam_fair_compute_review02_20261030/run_fair.py"]
    report = dict(status="PASS", official_train=R.N_TRAIN, official_validation=R.N_VAL,
                  steps_per_epoch=STEPS, epochs=EPOCHS, total_updates_per_arm=STEPS*EPOCHS,
                  model_pairs=checks, order_rule="default_rng(seed*101+epoch).permutation(all_rows)",
                  same_input_and_preprocessing=True, same_batch_and_order=True,
                  same_adam_lr_scheduler=True, bsam_frozen_two_pass_source=True,
                  source_config_sha256=F.sha(F.SOURCE), protocol_sha256=F.sha(PROTOCOL),
                  test_access=False)
    atomic_json(PREFLIGHT, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def run_arm(seed: int, arm: str):
    p = check_protocol()
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS" and report["protocol_sha256"] == F.sha(PROTOCOL)
    source = F.ready()
    assert seed in p["seeds"] and arm in p["arms"]
    folder = HERE / f"seed{seed}" / arm
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / "COMPLETE.json"
    if complete.exists():
        print(f"Already complete: {seed}/{arm}", flush=True)
        return
    model = F.new_model(seed, source)
    initial_sha = F.tensor_sha(model)
    expected = next(x["paired_initial_tensor_sha256"] for x in report["model_pairs"] if x["seed"] == seed)
    assert initial_sha == expected
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.1)
    stepper = F.BalancedSharpnessStep(model, optimizer, source["rho"]) if arm == "bsam" else None
    bn = F.BatchNormSecondPass(model) if arm == "bsam" else None
    weights = torch.tensor(source["bin_weights"], device="cuda", dtype=torch.float32)
    data = R.OfficialTrainData()
    R.SEED = seed
    assert R.config(data)["normalization_sha256"] == source["original_input_config"]["normalization_sha256"]
    val = R.OfficialValData(data)
    source_val = pd.read_csv(F.REF_VAL)
    assert len(source_val) == R.N_VAL and source_val.original_val_row.is_unique
    assert np.array_equal(source_val.original_val_row.to_numpy(np.int64), np.arange(R.N_VAL))
    assert np.array_equal(source_val.agb_true.to_numpy(np.float32), np.asarray(val.y, np.float32))
    train_radar, val_radar = R.alos_cache("train"), R.alos_cache("val")
    epoch, cursor, updates, train_seconds, epoch_sse = 1, 0, 0, 0., 0.
    logs, best = [], dict(rmse=float("inf"), epoch=None)
    last = folder / "last.pt"
    if last.exists():
        z = torch.load(last, map_location="cpu", weights_only=False)
        assert z["protocol_sha256"] == F.sha(PROTOCOL) and z["seed"] == seed and z["arm"] == arm
        model.load_state_dict(z["model"], strict=True)
        optimizer.load_state_dict(z["optimizer"])
        scheduler.load_state_dict(z["scheduler"])
        R.set_rng(z["rng"])
        epoch, cursor, updates = z["epoch"], z["cursor"], z["updates"]
        train_seconds, epoch_sse, logs, best = z["train_seconds"], z["epoch_sse"], z["logs"], z["best"]
        assert updates == (epoch-1)*STEPS + math.ceil(cursor/F.BATCH)
        print(json.dumps(dict(event="resume", seed=seed, arm=arm, epoch=epoch, cursor=cursor, updates=updates)), flush=True)
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    for e in range(epoch, EPOCHS+1):
        model.train()
        order = np.random.default_rng(seed * 101 + e).permutation(data.all_rows)
        begin = cursor if e == epoch else 0
        sse = epoch_sse if e == epoch else 0.
        epoch_start = time.perf_counter()
        lr_before = float(optimizer.param_groups[0]["lr"])
        for row in range(begin, R.N_TRAIN, F.BATCH):
            ix = order[row:row+F.BATCH]
            tic = time.perf_counter()
            x = data.batch(ix)
            radar, dn = R.radar(train_radar, ix)
            loss = F.update(model, optimizer, x, radar, dn, arm, weights, stepper, bn)
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tic
            updates += 1
            sse += loss * len(ix)
            next_cursor = min(row+F.BATCH, R.N_TRAIN)
            if updates % 500 == 0:
                F.save_pt(last, state(model, optimizer, scheduler, seed, arm, e,
                                      next_cursor, updates, train_seconds, sse, logs, best))
                print(json.dumps(dict(event="progress", seed=seed, arm=arm, epoch=e, updates=updates)), flush=True)
        assert updates == e*STEPS
        metric, pred = F.evaluate(model, val, val_radar, source_val)
        rec = dict(seed=seed, arm=arm, epoch=e, updates=updates, train_mse=sse/R.N_TRAIN,
                   lr=lr_before, train_step_seconds=train_seconds,
                   epoch_wall_seconds=time.perf_counter()-epoch_start,
                   peak_gpu_mib=float(torch.cuda.max_memory_allocated()/2**20),
                   validation=metric)
        logs.append(rec)
        atomic_json(folder/"EPOCH_METRICS.json", logs)
        if metric["overall"]["rmse"] < best["rmse"]:
            best = dict(rmse=metric["overall"]["rmse"], epoch=e, updates=updates)
            F.save_pt(folder/"best.pt", state(model, optimizer, scheduler, seed, arm, e,
                                             R.N_TRAIN, updates, train_seconds, sse, logs, best))
            pd.DataFrame(dict(original_val_row=source_val.original_val_row.to_numpy(np.int64),
                              grid_1deg=source_val.grid_1deg.to_numpy(),
                              agb_true=np.asarray(val.y, np.float32), prediction=pred)).to_csv(
                                  folder/"best_validation_predictions.csv", index=False)
        scheduler.step()
        cursor, epoch_sse = 0, 0.
        F.save_pt(last, state(model, optimizer, scheduler, seed, arm, e+1,
                              cursor, updates, train_seconds, epoch_sse, logs, best))
        print(json.dumps(dict(event="epoch_complete", seed=seed, arm=arm, epoch=e,
                              overall_rmse=metric["overall"]["rmse"],
                              q90_rmse=metric["q90_plus"]["rmse"], best=best), ensure_ascii=False), flush=True)
    assert len(logs) == EPOCHS and updates == EPOCHS*STEPS
    result = dict(status="COMPLETE", seed=seed, arm=arm, epochs=EPOCHS,
                  updates=updates, samples=EPOCHS*R.N_TRAIN, best=best,
                  train_step_seconds=train_seconds, total_wall_seconds=time.perf_counter()-wall_start,
                  peak_gpu_mib=float(torch.cuda.max_memory_allocated()/2**20),
                  parameters=sum(x.numel() for x in model.parameters()),
                  initial_tensor_sha256=initial_sha, best_checkpoint_sha256=F.sha(folder/"best.pt"),
                  protocol_sha256=F.sha(PROTOCOL), test_access=False)
    atomic_json(complete, result)
    val.close()
    print(json.dumps(dict(event="arm_complete", **result), ensure_ascii=False), flush=True)


def queue():
    check_protocol()
    assert json.loads(PREFLIGHT.read_text(encoding="utf-8"))["status"] == "PASS"
    for seed in (43, 44, 45):
        for arm in ("plain", "bsam"):
            run_arm(seed, arm)
            atomic_json(HERE/"QUEUE_STATUS.json", dict(last_complete=f"seed{seed}/{arm}",
                                                        completed=sum((HERE/f"seed{s}"/a/"COMPLETE.json").exists()
                                                                      for s in (43,44,45) for a in ("plain","bsam")),
                                                        total=6, test_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "arm", "queue"))
    parser.add_argument("--seed", type=int, choices=(43,44,45))
    parser.add_argument("--arm", choices=("plain", "bsam"))
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "arm":
        assert args.seed is not None and args.arm is not None
        run_arm(args.seed, args.arm)
    else:
        queue()
