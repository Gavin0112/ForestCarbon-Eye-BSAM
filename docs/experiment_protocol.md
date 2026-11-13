# Experiment protocol and six-arm identity

The numbers below are frozen archive observations, not a plan to run Test again. Train/Validation/Test contain 371,849/103,550/119,991 official AGBD-Lite rows. Train-only quantiles define strata. Validation Overall RMSE chooses checkpoints; Test is used for the reported evaluation after selection.

## Review-12: full-source paired Plain MSE and BSAM

Seeds 43, 44 and 45 share initial parameters and sample order within each pair. Batch 128, Adam LR 1e-4, StepLR(30, 0.1), BF16 forward, FP32 losses and optimizer, and 2,906 updates per full epoch are frozen. All six arms completed 14 epochs. Two BSAM arms continued to their Validation minima at E19; the third selected E14. Plain minima were E5, E5, E4. The serial per-update BSAM/Plain runtime ratio was 1.849. Original source entry: `run_review12.py`; protocol: `FROZEN_PROTOCOL.json` beside it. The full server execution record and results are in the original Review-12 archive, not distributed here.

The included `results/validation/ARM_SUMMARY.csv` and `ALL_ARMS_EPOCH_METRICS.csv` are from a **later training continuation evaluated on Validation**: Plain arms ultimately reached E25 and BSAM arms E80. Those extra epochs were recorded after the frozen Test evaluation and did not generate the reported Test checkpoints or revise their selection. Read the Test checkpoint epochs from `results/summaries/test_summary.csv` and the archived Review-12 metric JSON, rather than treating `epochs_done` in that later Validation file as the Test training budget.

Review-12's E14→E19 convergence extension, Review-13's interrupted arms resumed from checkpoints, and the later Review-12 E25/E80 continuation are different continuation events within the **same server-side project**. The separate archive names identify handoffs, not different machines or independent retraining.

| Seed | Plain Test RMSE | BSAM Test RMSE | BSAM best epoch |
| ---: | ---: | ---: | ---: |
| 43 | 58.7567 | 57.7201 | 19 |
| 44 | 59.3787 | 57.7020 | 19 |
| 45 | 58.7060 | 57.7406 | 14 |

## Review-13: final six interventions

M0 is **Review-12 seed-43 BSAM E19**, reused by documented protocol decision from the same server-side project; it is the reference, not a separately trained intervention. Six archived arms M1–M6 each ran 24 epochs / 69,744 updates. The archive contains six weights and six full-Test prediction files, with matching hashes in `R13_TEST_METRICS.json`. The six intervention checkpoints are included in `ablations/weights/` under the [checkpoint terms](../MODEL_WEIGHTS_TERMS.md); their per-file index remains here. Row-level predictions are not bundled. The provided `UNIFIED_ABLATION_RUNNER.py` is a reconstructed alignment script; the delivered Review-13 archive does not contain a byte-identical copy of its training entry. M6 masks all eight metadata values while retaining FiLM.

| Final arm | Actual intervention | Test Overall RMSE | Code/config status |
| --- | --- | ---: | --- |
| M0 | Full model, reused Review-12 seed-43 BSAM | 57.7201 | Reuse evidence verified |
| M1 | Zero S2 and acquisition date | 62.8947 | Archived result verified; runner reconstructed |
| M2 | Zero DEM and terrain metadata | 60.3787 | Archived result verified; runner reconstructed |
| M3 | Zero land cover | 58.1394 | Archived result verified; runner reconstructed |
| M4 | Zero scaled ALOS and raw DN | 58.1887 | Archived result verified; runner reconstructed |
| M5 | Disable FiLM conditioning | 58.9326 | Archived result verified; runner reconstructed |
| M6 | Zero all eight metadata values, keep FiLM | 58.6989 | Archived result verified; runner reconstructed |

The six `ABLATION_CONFIGS/M1.json`–`M6.json` in this repository match the final names and mask meanings. Their presence does not prove that the reconstructed runner generated the archived weights. The archived `PROTOCOL_TEMPLATE.json` contains server-specific paths and has not been released here.

## S2-only and matched-time check

The S2-only runs retain the full network, zero all non-S2 model inputs after frozen preprocessing, and train from scratch with the same seeds. Patience is five full-Validation checks, with no 14-epoch minimum and a 28-epoch cap. Mean Test RMSE: Plain 67.3275, BSAM 64.4229. A separate earlier matched-time experiment gave each arm 5,400 s on an RTX 4060 Laptop GPU; its BSAM Test RMSE was higher than Plain by 0.4814, 0.9365, 1.0380 for seeds 43–45. These experiments answer different budget questions.
