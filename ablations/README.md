# Ablation experiments and checkpoint records

This folder collects the **paper-reported** ablation records and checkpoint lookup table. The binary checkpoints are in `weights/` under the [checkpoint terms](../MODEL_WEIGHTS_TERMS.md).

## Input and FiLM interventions

The six Review-13 intervention checkpoints, named `review13_M1.pt` through `review13_M6.pt`, are included in `weights/`. The [protocol](../docs/experiment_protocol.md) defines each mask; `input_film/configs/`, `input_film/validation/` and the [Test metric record](input_film/R13_TEST_METRICS.json) provide the configuration and aggregate evidence. M0 reuses the Review-12 seed-43 BSAM checkpoint; there is no separately trained or duplicated M0 checkpoint. The M6 checkpoint is the all-metadata-masked arm with FiLM retained. The delivered Review-13 archive does not include a byte-identical training entry. The [reconstructed alignment runner](../experiments/agbd_lite/bsam_unified_ablation_review13_20261031/UNIFIED_ABLATION_RUNNER.py) documents the intended configuration but is not proof of the server execution.

The source tree also contains an older planning `ABLATION_CONFIGS/M0.json` with `init=fresh_seeded_MSEFilm`. **That file is not the final M0 protocol and was not used to create an additional M0 weight.** The final M0 identity is the documented Review-12 seed-43 BSAM reuse in `input_film/M0_REUSE_SUMMARY.json`; no M0 config is copied into this folder.

## S2-only information ablation

The six Validation-selected S2-only checkpoints, named `s2only_seed43_plain.pt` through `s2only_seed45_bsam.pt`, are included in `weights/`. They use the full model architecture with non-S2 inputs fixed to zero. See the [training entry](../experiments/agbd_lite/s2_only_paired_20261110/run_s2only.py), [frozen protocol](s2_only/FROZEN_PROTOCOL.json), [Validation logs](s2_only/validation/ARM_SUMMARY.csv) and [Test metric record](s2_only/TEST_METRICS.json).

`weights_index.csv` lists the twelve files, selected epoch, official Test RMSE and original SHA-256. The included binaries match the indexed SHA-256 values. The AGBD-Lite dataset must be obtained from its creators.

## Earlier compute-control comparison

`compute_control/EQUAL_TIME_TEST_METRICS.json` retains the earlier three-seed equal-time comparison disclosed briefly in the paper. Its BSAM results were less favorable. This is a separate training-budget question, and its exploratory/intermediate checkpoint files are not among the paper-result weights indexed above.
