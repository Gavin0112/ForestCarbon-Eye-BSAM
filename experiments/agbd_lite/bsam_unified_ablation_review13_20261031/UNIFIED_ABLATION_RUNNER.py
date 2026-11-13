"""Six-arm Review-13 design aligned with the authoritative delivery archive.

This runner was reconstructed from a planning version; the archive does not
contain the exact server training runner. It is not proof of a reproduced run.
No Test path is present. Runtime imports are deferred until --train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CONFIGS = HERE / "ABLATION_CONFIGS"
SOURCE_CONFIG = EXP / "mse_film_bsam_20261023" / "MSE_FILM_BSAM_CONFIG.json"
PROTOCOL = CONFIGS / "PROTOCOL_TEMPLATE.json"
ARMS = tuple(f"M{i}" for i in range(7))
RESULT_FIELDS = ("arm", "seed", "optimizer_mode", "selected_epoch", "selected_node", "optimizer_updates",
                 "training_seconds", "peak_gpu_mib", "model_parameter_count",
                 "requires_grad_parameter_count", "active_gradient_parameter_count",
                 "overall_rmse", "overall_mae", "overall_bias",
                 "q50_q90_rmse", "q90_plus_rmse", "q90_plus_bias",
                 "q90_plus_spearman", "validation_rows", "checkpoint_sha256",
                 "protocol_sha256", "arm_config_sha256")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def static_check() -> None:
    p = read(PROTOCOL)
    assert p["status"] in ("PENDING_REVIEW12", "FROZEN")
    if p["status"] == "PENDING_REVIEW12":
        assert p["execution_enabled"] is False
        assert p["optimizer_mode"] is None and p["budget_mode"] is None
    assert p["seeds"] == [43, 44, 45]
    assert p["official_train_rows"] == 371849 and p["official_validation_rows"] == 103550
    assert p["batch"] == 128 and p["lr"] == 1e-4
    expected = {
        "M0": [], "M1": ["s2", "s2_date_meta"],
        "M2": ["dem_patch", "terrain_meta"], "M3": ["lc"],
        "M4": ["alos_norm", "alos_dn"], "M5": ["film_identity"],
        "M6": ["all_meta"],
    }
    for arm, masks in expected.items():
        cfg = read(CONFIGS / f"{arm}.json")
        assert cfg["arm"] == arm and cfg["masks"] == masks
        assert cfg["enabled"] is True
    assert len(set(RESULT_FIELDS)) == len(RESULT_FIELDS)
    # Check source *text*, not checkpoints, caches, labels or tensors.
    source = (EXP / "mse_film_main_20261022" / "mse_film_model.py").read_text(encoding="utf-8")
    assert "class MSEFilm" in source and "self.backbone((f, meta))" in source
    assert "self.lc(lc)" in source and "self.alos(radar, dn)" in source
    bsam = (EXP / "mse_film_bsam_20261023" / "bsam_step.py").read_text(encoding="utf-8")
    assert "class BalancedSharpnessStep" in bsam and "class BatchNormSecondPass" in bsam
    film = (HERE.parents[2] / "third_party" / "AGBD-GFMs" / "model" / "nico_net_film.py").read_text(encoding="utf-8")
    assert "if b is not None: out = self.film1(out, b)" in film
    assert "if b is not None: out = self.film2(out, b)" in film
    assert "if b is not None: out = self.film3(out, b)" in film
    assert "if b is not None: shortcut = self.film_shortcut(shortcut, b)" in film
    print("STATIC_CHECK_PASS: M0-M6 configs, pending protocol, source interfaces; no model or data loaded")


def mask_inputs(arm, s2, dem, lc, meta, radar, dn):
    """Mask after frozen normalization; never mutate loader/cache tensors."""
    if arm == "M1":
        s2 = s2.new_zeros(s2.shape)
        meta = meta.clone(); meta[:, 0:2] = 0  # S2 acquisition date is S2-sourced.
    elif arm == "M2":
        dem = dem.new_zeros(dem.shape)
        meta = meta.clone(); meta[:, 4:8] = 0
    elif arm == "M3":
        lc = lc.new_zeros(lc.shape)  # 22 indicators + unused slot + 3 validity/confidence.
    elif arm == "M4":
        radar = radar.new_zeros(radar.shape)
        dn = dn.new_zeros(dn.shape)  # RadarEncoder derives flags and texture from DN.
    elif arm == "M6":
        meta = meta.new_zeros(meta.shape)
    return s2, dem, lc, meta, radar, dn


def runtime_model(base, arm):
    """Keep the original state_dict/parameterization; disable only selected computation."""
    if arm == "M5":
        # NicoNet_FiLM first computes its MLP; its body accepts b=None and skips
        # all four entry FiLM calls. MLP/FiLM parameters remain registered but idle.
        def remove_condition(_module, inputs):
            feature, _condition = inputs[0]
            return ((feature, None),)
        base.backbone.body.register_forward_pre_hook(remove_condition)
    return base


def make_masked_model(base, arm, torch):
    class MaskedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = runtime_model(base, arm)

        def forward(self, s2, dem, lc, meta, radar, dn):
            return self.base(*mask_inputs(arm, s2, dem, lc, meta, radar, dn))
    return MaskedModel()


def frozen_lock(arm, seed):
    p, cfg = read(PROTOCOL), read(CONFIGS / f"{arm}.json")
    assert cfg["enabled"], "Optional/disabled arm cannot run"
    assert p["status"] == "FROZEN" and p["execution_enabled"] is True
    assert p["optimizer_mode"] in ("plain_mse", "bsam")
    assert p["budget_mode"] in ("fixed_updates", "equal_train_seconds")
    assert p["checkpoint_rule"] == "minimum_full_validation_overall_rmse"
    assert p["validation_frequency"] == "each_completed_epoch_and_final_budget_node"
    assert p["seeds"] == [43, 44, 45] and seed in p["seeds"]
    assert p["batch"] == 128 and p["lr"] == 1e-4
    assert p["official_train_rows"] == 371849 and p["official_validation_rows"] == 103550
    assert p["source_config_sha256"] == sha(SOURCE_CONFIG)
    assert p["model_source_sha256"] == sha(EXP / "mse_film_main_20261022" / "mse_film_model.py")
    assert p["update_source_sha256"] == sha(EXP / "bsam_fair_compute_review02_20261030" / "run_fair.py")
    if p["budget_mode"] == "fixed_updates":
        assert isinstance(p["fixed_updates"], int) and p["fixed_updates"] > 0
    else:
        assert isinstance(p["train_seconds_budget"], (int, float)) and p["train_seconds_budget"] > 0
    return p, cfg


def execute(arm: str, seed: int) -> None:
    # This is never called by static-check. All GPU/model/data imports live here.
    import sys
    import numpy as np
    import pandas as pd
    import torch
    sys.path[:0] = [str(EXP / "bsam_fair_compute_review02_20261030"),
                    str(EXP / "mse_film_main_20261022"),
                    str(EXP / "mse_film_bsam_20261023")]
    import run_fair as F
    import run_mse_film as R
    from mse_film_model import MSEFilm
    from bsam_step import BalancedSharpnessStep, BatchNormSecondPass

    p, _cfg = frozen_lock(arm, seed)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    source = read(SOURCE_CONFIG)
    assert len(source["bin_weights"]) == 50 and source["rho"] == 0.2
    out = HERE / "runs" / f"seed{seed}" / arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise RuntimeError("Completed arm cannot be rerun")
    R.SEED = seed
    R.seed_all()
    c = source["original_input_config"]
    model = make_masked_model(MSEFilm(c["dem_mean_m"], c["dem_std_m"]), arm, torch).cuda()
    initial_sha = F.tensor_sha(model)
    init_path = HERE / "runs" / f"seed{seed}" / "INITIAL_SHA.json"
    if init_path.exists():
        assert read(init_path)["sha256"] == initial_sha
    else:
        atomic_json(init_path, {"seed": seed, "sha256": initial_sha})
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    is_bsam = p["optimizer_mode"] == "bsam"
    stepper = BalancedSharpnessStep(model, opt, source["rho"]) if is_bsam else None
    bn = BatchNormSecondPass(model) if is_bsam else None
    weights = torch.as_tensor(source["bin_weights"], device="cuda", dtype=torch.float32)
    data = R.OfficialTrainData()
    assert len(data.y) == 371849
    assert R.config(data)["normalization_sha256"] == c["normalization_sha256"]
    val = R.OfficialValData(data)
    val_reference = pd.read_csv(EXP / "mse_film_convergence_sfr_20261022" /
                                "MSE_FILM_EPOCH7_VAL_PREDICTIONS.csv")
    assert len(val_reference) == 103550 and val_reference.original_val_row.is_unique
    assert np.array_equal(val_reference.original_val_row.to_numpy(np.int64), np.arange(103550))
    assert np.array_equal(val_reference.agb_true.to_numpy(np.float32), np.asarray(val.y, np.float32))
    train_radar, val_radar = R.alos_cache("train"), R.alos_cache("val")
    logs, best = [], {"overall_rmse": math.inf, "epoch": None}
    epoch, start, updates, train_seconds = 1, 0, 0, 0.0
    last_path = out / "last.pt"

    def state():
        return dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                    rng=R.rng_state(), epoch=epoch, start=start, updates=updates,
                    train_seconds=train_seconds, logs=logs, best=best,
                    protocol_sha256=sha(PROTOCOL), arm_config_sha256=sha(CONFIGS / f"{arm}.json"),
                    initial_sha256=initial_sha)

    def save(path):
        tmp = path.with_suffix(path.suffix + ".partial")
        torch.save(state(), tmp); os.replace(tmp, path)

    if last_path.exists():
        z = torch.load(last_path, map_location="cpu", weights_only=False)
        assert z["protocol_sha256"] == sha(PROTOCOL)
        assert z["arm_config_sha256"] == sha(CONFIGS / f"{arm}.json")
        assert z["initial_sha256"] == initial_sha
        model.load_state_dict(z["model"], strict=True)
        opt.load_state_dict(z["optimizer"]); sched.load_state_dict(z["scheduler"])
        R.set_rng(z["rng"])
        epoch, start, updates, train_seconds = z["epoch"], z["start"], z["updates"], z["train_seconds"]
        logs, best = z["logs"], z["best"]

    def checkpoint(node):
        nonlocal best
        metrics, predictions = F.evaluate(model, val, val_radar, val_reference)
        record = {"arm": arm, "seed": seed, "node": node, "epoch": epoch,
                  "optimizer_updates": updates, "training_seconds": train_seconds,
                  "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "lr": opt.param_groups[0]["lr"], "validation": metrics}
        logs.append(record)
        atomic_json(out / "VALIDATION_NODES.json", logs)
        current = metrics["overall"]["rmse"]
        if current < best["overall_rmse"]:
            best = {"overall_rmse": current, "epoch": epoch, "node": node, "updates": updates}
            save(out / "best.pt")
            pd.DataFrame({"original_val_row": val_reference.original_val_row,
                          "grid_1deg": val_reference.grid_1deg,
                          "agb_true": np.asarray(val.y, np.float32),
                          "prediction": predictions}).to_csv(out / "best_validation_predictions.csv", index=False)
        save(out / f"node_{updates:06d}.pt")

    torch.cuda.reset_peak_memory_stats()
    try:
        while True:
            model.train()
            order = np.random.default_rng(seed * 101 + epoch).permutation(data.all_rows)
            for cursor in range(start, 371849, 128):
                if p["budget_mode"] == "fixed_updates" and updates >= p["fixed_updates"]:
                    break
                if p["budget_mode"] == "equal_train_seconds" and train_seconds >= p["train_seconds_budget"]:
                    break
                tick = time.perf_counter()
                ix = order[cursor:cursor + 128]
                batch = data.batch(ix)
                radar, dn = R.radar(train_radar, ix)
                loss = F.update(model, opt, batch, radar, dn,
                                "bsam" if is_bsam else "plain", weights, stepper, bn)
                assert math.isfinite(loss)
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - tick
                updates += 1; start = min(cursor + 128, 371849)
                if updates % 500 == 0:
                    save(last_path)
            full_epoch = start == 371849
            if full_epoch:
                checkpoint(f"epoch_{epoch}")
                sched.step(); epoch += 1; start = 0
                save(last_path)
            done = (updates >= p["fixed_updates"] if p["budget_mode"] == "fixed_updates"
                    else train_seconds >= p["train_seconds_budget"])
            if done:
                if not full_epoch and (not logs or logs[-1]["optimizer_updates"] != updates):
                    checkpoint("budget_final")
                break
        save(last_path)
        assert best["epoch"] is not None
        chosen = next(x for x in logs if x["node"] == best["node"])
        m = chosen["validation"]
        mid = m.get("q50_q90", {})
        high = m.get("q90_plus", {})
        row = dict(arm=arm, seed=seed, optimizer_mode=p["optimizer_mode"],
                   selected_epoch=best["epoch"], selected_node=best["node"],
                   optimizer_updates=best["updates"],
                   training_seconds=train_seconds, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
                   model_parameter_count=sum(z.numel() for z in model.parameters()),
                   requires_grad_parameter_count=sum(z.numel() for z in model.parameters() if z.requires_grad),
                   active_gradient_parameter_count=sum(z.numel() for z in model.parameters() if z.grad is not None),
                   overall_rmse=m["overall"]["rmse"], overall_mae=m["overall"]["mae"],
                   overall_bias=m["overall"]["bias"], q50_q90_rmse=mid.get("rmse"),
                   q90_plus_rmse=high.get("rmse"), q90_plus_bias=high.get("bias"),
                   q90_plus_spearman=high.get("spearman"), validation_rows=103550,
                   checkpoint_sha256=sha(out / "best.pt"), protocol_sha256=sha(PROTOCOL),
                   arm_config_sha256=sha(CONFIGS / f"{arm}.json"))
        with (out / "RESULT.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS); writer.writeheader(); writer.writerow(row)
        atomic_json(out / "COMPLETE.json", {"status": "COMPLETE", "arm": arm, "seed": seed,
                                             "best": best, "all_updates": updates, "test_access": False})
    finally:
        val.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("static-check", "train"), default="static-check", nargs="?")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=(43, 44, 45))
    parser.add_argument("--execute-frozen", action="store_true")
    a = parser.parse_args()
    if a.command == "static-check":
        static_check(); return
    if not a.execute_frozen or a.arm is None or a.seed is None:
        parser.error("training requires --arm, --seed, --execute-frozen and a frozen protocol")
    execute(a.arm, a.seed)


if __name__ == "__main__":
    main()
