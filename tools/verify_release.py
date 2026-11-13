"""Read-only checks for the repository source, checkpoints, and aggregate records.

This does not read raw AGBD-Lite, train models, or replay historical inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "agbd_lite"
METRICS = ROOT / "results" / "metrics"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(EXP / "bsam_long_budget_review12_20261031" / "FROZEN_PROTOCOL.json")
    for rel, expected in protocol["source_sha256"].items():
        path = EXP / rel
        assert path.is_file(), path
        assert sha256(path) == expected, path

    sets = (
        ("REVIEW12_DELIVERY_20261103", "REVIEW12_SIX_WEIGHTS_TEST_METRICS.json",
         {f"seed{s}_{mode}" for s in (43, 44, 45) for mode in ("plain", "bsam")}),
        ("REVIEW13_DELIVERY_20261103", "R13_TEST_METRICS.json",
         {f"M{i}" for i in range(1, 7)}),
        ("S2ONLY_DELIVERY_20261110", "S2ONLY_SIX_WEIGHTS_TEST_METRICS.json",
         {f"seed{s}_{mode}" for s in (43, 44, 45) for mode in ("plain", "bsam")}),
    )
    for folder, name, expected in sets:
        data = read_json(METRICS / folder / name)
        assert set(data["arms"]) == expected, folder
        for arm, record in data["arms"].items():
            assert record["test_metrics"]["overall"]["n"] == 119991, (folder, arm)

    m6 = read_json(EXP / "bsam_unified_ablation_review13_20261031" / "ABLATION_CONFIGS" / "M6.json")
    assert m6["arm"] == "M6" and m6["masks"] == ["all_meta"] and m6["enabled"]

    expected_weights = {}
    for folder, name, _ in sets:
        data = read_json(METRICS / folder / name)
        if folder.startswith("REVIEW12"):
            parent, prefix = "main_model", "review12_"
        elif folder.startswith("REVIEW13"):
            parent, prefix = "ablations", "review13_"
        else:
            parent, prefix = "ablations", "s2only_"
        for arm, record in data["arms"].items():
            path = ROOT / parent / "weights" / f"{prefix}{arm}.pt"
            expected_weights[path] = record["checkpoint_sha256"]

    e7 = read_json(METRICS / "MSE_FILM_E7_OFFICIAL_TEST_METRICS.json")
    expected_weights[ROOT / "main_model" / "weights" / "mse_film_E7.pt"] = e7["checkpoint_sha256"]
    actual_weights = {p for p in ROOT.rglob("*.pt") if p.is_file()}
    assert actual_weights == set(expected_weights), (actual_weights - set(expected_weights), set(expected_weights) - actual_weights)
    for path, expected in expected_weights.items():
        assert sha256(path) == expected, path

    for folder in ("main_model", "ablations"):
        with (ROOT / folder / "weights_index.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert {ROOT / folder / row["checkpoint"] for row in rows} == {
            path for path in expected_weights if path.parent == ROOT / folder / "weights"
        }
        for row in rows:
            assert expected_weights[ROOT / folder / row["checkpoint"]] == row["sha256"]

    forbidden = {".h5", ".hdf5", ".ckpt", ".f32", ".i16", ".u8", ".u16"}
    assert not [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    print("PASS: five frozen source hashes, 18 aggregate arms, M6 mask, 19 checkpoint binaries and indices, no dataset payload")


if __name__ == "__main__":
    main()
