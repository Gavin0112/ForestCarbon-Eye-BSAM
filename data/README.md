# Data access and required assets

AGBD-Lite v1 is produced and distributed by Ghjulia Sialelli, Linus Scheibenreif, Jan Dirk Wegner and Konrad Schindler. Download it from [their Zenodo record](https://doi.org/10.5281/zenodo.18485030) and read the [dataset card](https://huggingface.co/datasets/prs-eth/AGBD-Lite), which identifies CC BY-NC 4.0. Cite the dataset **and** the distinct [original AGBD paper](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025), authored by Ghjulia Sialelli, Torben Peters, Jan Dirk Wegner and Konrad Schindler. This project did not create either dataset. Do not add downloaded H5 files, GEDI labels or upstream data to this repository.

The historical scripts expect this project-root layout after obtaining and preparing the data under its terms:

```text
data/agbd_lite/AGBD-Lite-train.h5
data/agbd_lite/AGBD-Lite-val.h5
data/agbd_lite/AGBD-Lite-test.h5
data/agbd_lite/cache_s2only/...
experiments/agbd_lite/p95_geographic_multimodal_fusion/p95_1/train_s2_spatial_norm.f32
experiments/agbd_lite/p19_2_dem_pilot/run_aw3d30/...
experiments/agbd_lite/p48_lc_multimodal_pilot/...
experiments/agbd_lite/p54_final_feature_alignment/...
experiments/agbd_lite/p113_alos_numeric_pilot/...
```

The H5 files alone are insufficient for the historical training scripts: those scripts also read Train-derived Sentinel-2 normalization, DEM, land-cover, metadata, ALOS and label caches, plus validation identity records. This repository does not contain a verified pipeline to regenerate every cache from public upstream files. File names in this layout describe historical read paths; creating empty files or substituting synthetic arrays will not reproduce the study.

The final model consumes Sentinel-2 `[12,25,25]`, standardized DEM `[1,25,25]`, land cover `[26,25,25]`, scaled ALOS HH/HV `[2,25,25]`, paired raw ALOS DN `[2,25,25]`, and eight metadata values. Targets are official `GEDI/agbd`. The study describes zero/NoData handling and frozen Train-side scaling; [reproduction limits](../docs/reproduction.md) distinguish what is verified from what remains uncertain about source products and raster alignment.
