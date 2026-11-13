# Architecture and source identity

The paper's active predictor is the `MSEFilm` class in `experiments/agbd_lite/mse_film_main_20261022/mse_film_model.py`. This file is a byte-identical copy of the original project source (SHA-256 `36e18eb993b2a7688fac5364fd9cda8947f882af54be934100a9f8e46e40e94a`). The second `mse_film_pbsam_20261023/mse_film_model.py` copy is identical; its older folder name does not name the final method.

| Paper mechanism | Source class/function | Verified description |
| --- | --- | --- |
| Sentinel-2 encoder | `GroupedS2` | 12 bands, grouped transforms, 64 output channels; B09 also appears once as a dedicated channel |
| DEM encoder | `TerrainEncoder` | standardized height plus four derived terrain maps, 16 output channels |
| Land-cover encoder | `LandCoverEncoder` | 26 supplied channels, neighborhood composition and embeddings, 16 output channels |
| ALOS encoder | `RadarEncoder` | scaled HH/HV and paired raw DN-derived flags/texture, 16 output channels |
| Fusion and FiLM | `MSEFilm` and upstream `PointwiseBlock` | 112 concatenated channels; 8D metadata conditions entry only |
| Capacity adapter | `CapacityAdapter` in `n5_model.py` | residual 256→421→256 pointwise path |
| Shared backbone and head | `NicoNet_CapacityControl` and upstream `NicoNet_FiLM` | eight residual SepConv blocks, long skip, one output map, center value retained |
| BSAM update | `BalancedSharpnessStep`, `BatchNormSecondPass` in `bsam_step.py`; update in `run_fair.py` | weighted first-pass perturbation, ordinary MSE second pass, restored-weight Adam update |

The instantiated model has 2,123,367 parameters, including an inherited 257-parameter second-moments head that is not used in the active forward path. The underlying NicoNet code is by the [AGBD-GFMs authors](https://github.com/ghjuliasialelli/AGBD-GFMs), is **not** bundled here, and must not be presented as independently invented by this project.

The paper's last-minibatch factor `|B|/128`, BF16 forward/FP32 loss behavior, and second-pass batch-normalization handling are documented in its Methods section. This source snapshot supports the architecture, but a full training replay is still blocked by missing caches and upstream-version binding.
