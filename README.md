# Improving High-Biomass Estimates: Multisource Fusion with Balanced Sharpness-Aware Training

This repository contains the code, experiment records, and 19 project-trained model checkpoints for the study named above. The checkpoints may be used for noncommercial scientific research under the [model weight terms](MODEL_WEIGHTS_TERMS.md).

## Study overview

The study by Chang, Cai, Guo, Ma, and Lian predicts GEDI Level 4A above-ground biomass density (AGBD) on the **AGBD-Lite dataset created by Sialelli, Scheibenreif, Wegner, and Schindler**. The dataset and its creation are not contributions of this project. Cite both the [AGBD-Lite dataset](https://doi.org/10.5281/zenodo.18485030) and the [original AGBD paper](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025) when using those data.

## Data and code attribution

- **Dataset:** AGBD-Lite v1 is the work of Ghjulia Sialelli, Linus Scheibenreif, Jan Dirk Wegner and Konrad Schindler, *AGBD-Lite* (2026), [Zenodo DOI 10.5281/zenodo.18485030](https://doi.org/10.5281/zenodo.18485030). Its [dataset card](https://huggingface.co/datasets/prs-eth/AGBD-Lite) identifies CC BY-NC 4.0. The original parent dataset is documented by Ghjulia Sialelli, Torben Peters, Jan Dirk Wegner and Konrad Schindler, *AGBD: A Global-scale Biomass Dataset* (2025), [DOI 10.5194/isprs-annals-X-G-2025-829-2025](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025). These are two distinct works with different author lists; cite both as requested by the AGBD-Lite record.
- **Upstream model and code:** The FCN-FiLM/NicoNet backbone and original implementation belong to the authors of [AGBD-GFMs](https://github.com/ghjuliasialelli/AGBD-GFMs), whose repository [licenses its code under MIT](https://github.com/ghjuliasialelli/AGBD-GFMs/blob/main/LICENSE). We do not include a copy of their repository, pretrained baselines or dataset-generation code. Users obtain that dependency from its authors and retain its copyright and license notices when redistributing any copied portions.
- **Our contribution:** This project adapts that backbone for footprint regression with separate source encoders, 112-channel fusion, an eight-dimensional FiLM condition and a capacity adapter, and evaluates the BSAM training update. Our code and checkpoints are our experimental outputs; we do not claim to have created AGBD-Lite or NicoNet.

The root [MIT LICENSE](LICENSE) covers only the original project source code, configurations, documentation, aggregate result records and indices. The 19 project-trained checkpoints are governed separately by [MODEL_WEIGHTS_TERMS.md](MODEL_WEIGHTS_TERMS.md) for noncommercial scientific research; MIT does not apply to them. Third-party code and data retain their own terms.

No AGBD-Lite H5 files, GEDI labels, source satellite tiles or upstream dataset-generation files are in this repository. Obtain data from the official creators' links above. Locally trained checkpoints and their SHA-256 indices are in `main_model/` and `ablations/`.

## Method and architecture

Our work adapts the [AGBD-GFMs FCN-FiLM/NicoNet implementation](https://github.com/ghjuliasialelli/AGBD-GFMs) for a single footprint prediction. Separate Sentinel-2, DEM, land-cover, and ALOS encoders form a 112-channel entry. Eight date, location, and terrain descriptors condition that entry through FiLM. A residual capacity adapter and eight separable-convolution blocks precede the center-footprint regression head. The training method is **BSAM**: Train-label-frequency weights shape a temporary perturbation, followed by an unweighted center-pixel MSE gradient and an Adam update. See [architecture](docs/architecture.md) and [third-party notices](third_party/THIRD_PARTY_NOTICES.md).

## Repository structure

- [main_model/](main_model/README.md): main predictor, training-source map, checkpoint binaries and provenance indices.
- [ablations/](ablations/README.md): paper-reported input/FiLM and S2-only ablation records, checkpoint binaries and provenance indices.
- `experiments/agbd_lite/`: original relative source paths retained for frozen hash checks. Some historical folder names contain older method names; the final method is BSAM.
- `configs/`: frozen Review-12 and S2-only protocol records.
- `results/metrics/` and `results/validation/`: archived aggregate Test and Validation evidence, checkpoint and prediction hashes; no row-level labels or predictions.
- `tools/`: offline metric checker for user-supplied, legitimately obtained predictions.
- `data/`: access and required data layout only; no dataset payload.
- `docs/`: protocol, architecture, reproduction limits and provenance.

The 19 `.pt` checkpoints (about 395 MB) are included and indexed by SHA-256. They are covered by [noncommercial research checkpoint terms](MODEL_WEIGHTS_TERMS.md) and configured for Git LFS by `.gitattributes`. The later resume delivery's six best weights are byte-identical to the Review-12 copies and are not duplicated. M0 refers to the Review-12 seed-43 BSAM checkpoint.

## Dataset and access

Download AGBD-Lite v1 from the [creators' Zenodo record](https://doi.org/10.5281/zenodo.18485030); consult their [dataset card and schema](https://huggingface.co/datasets/prs-eth/AGBD-Lite). The official Train, Validation and Test splits contain 371,849, 103,550 and 119,991 records. This repository does not redistribute their H5 files, labels, statistics, original satellite data or generation code. See [data access](data/README.md) for required external assets and preprocessing limitations. AGBD-Lite is identified as CC BY-NC 4.0 on the creators' dataset card; users must follow its terms.

## Environment and installation

The archived full-source runs reported Python 3.10, PyTorch 2.5.1 and CUDA 12.5.1. A complete training environment lock was not delivered. `requirements.txt` records direct Python dependencies but is **not** an exact historical environment. Install a PyTorch build compatible with your GPU and verify the environment before training. The upstream NicoNet module is a separate dependency; obtain it from its authors into `third_party/AGBD-GFMs/` after checking its current license and a suitable version. This repository does not vendor or claim ownership of that upstream code.

## Training and evaluation

The main paired Plain MSE/BSAM experiment used seeds 43–45, batch 128, Adam at 1e-4, a common first 14 Train epochs, full Validation checkpoint selection and BSAM radius 0.2. The [main model guide](main_model/README.md) links the model and training source and identifies each checkpoint. Review-12, Review-13 and the later Review-12 continuation were performed within the same server-side project; their separate archives are delivery packages, not evidence of different servers. The provided Review-13 runner is a **reconstructed alignment script**, not a byte-identical copy of its archived training entry point. Large preprocessing caches and historical validation assets are absent. **Do not treat these scripts as verified clean-room training commands.** See [reproduction limits](docs/reproduction.md).

To inspect an already frozen prediction CSV that you have the right to use, run:

```bash
python tools/evaluate.py path/to/predictions.csv --prediction-column agb_pred --label-column agb_true --id-column original_test_row
```

This reads only the supplied CSV. It does not train or select a checkpoint. It must not be used to tune on Test.

## Ablation experiments

The [ablation folder](ablations/README.md) documents the six Review-13 input/FiLM interventions M1–M6 and six S2-only paired checkpoints. M0 reuses Review-12 seed-43 BSAM; all 19 project-trained binaries are included. Selected epochs, hashes and metrics are mapped in the two `weights_index.csv` files and the [protocol](docs/experiment_protocol.md). The reconstructed Review-13 runner is identified separately from archived execution evidence.

## Results

All aggregate numbers below come from archived full official Test evaluations; units are Mg ha⁻¹. No Test result is used to select a checkpoint in the reported protocols. The historical project contains earlier Test exploration, which is recorded in [reproduction limits](docs/reproduction.md).

| Experiment | Plain MSE | BSAM | Scope |
| --- | ---: | ---: | --- |
| Full source, mean of seeds 43–45 | 58.9471 | 57.7209 | Equal first 14 Train epochs; Validation-selected checkpoints can have different final epochs |
| S2-only, mean of seeds 43–45 | 67.3275 | 64.4229 | Full architecture with non-S2 inputs fixed to zero |

An earlier three-seed equal-time experiment (5,400 s per arm on an RTX 4060 Laptop GPU) had higher BSAM Test RMSE by 0.481, 0.936 and 1.038 in its three pairs; the [original aggregate record](results/metrics/EQUAL_TIME_TEST_METRICS.json) is included. The main equal-exposure gain is **not** an equal-time gain. Six-arm Test RMSE values are in [protocol](docs/experiment_protocol.md) and machine-readable records under `results/metrics/`. The study's MSE-FiLM E7 comparator is preserved in its [aggregate metric record](results/metrics/MSE_FILM_E7_OFFICIAL_TEST_METRICS.json); external FCN-FiLM numbers come from [Sialelli et al.'s preprint](https://arxiv.org/abs/2608.04792v2), Table A.4.

## Reproducibility, license and citation

This repository supports source, configuration, checkpoint, checkpoint-hash and aggregate-result inspection. An independently verified from-scratch rerun requires additional assets and verification. Large caches, the exact original Review-13 server runner, a complete environment lock and upstream-version binding remain unresolved. See [reproduction limits](docs/reproduction.md).

**Licensing:** original project source, configurations, documentation, aggregate result records and indices use the project [MIT LICENSE](LICENSE). The nineteen project-trained weights have separate [noncommercial scientific-research terms](MODEL_WEIGHTS_TERMS.md). Neither grant covers AGBD-Lite or upstream code. Cite this work as Chang, Cai, Guo, Ma, and Lian, *Improving High-Biomass Estimates: Multisource Fusion with Balanced Sharpness-Aware Training*. Cite upstream data and code separately using their official records.
