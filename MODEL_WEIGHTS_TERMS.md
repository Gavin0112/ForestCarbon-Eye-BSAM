# Model checkpoint use terms

Copyright © 2026 ForestCarbon-Eye-BSAM project authors (Chang, Cai, Guo, Ma, and Lian).

These terms apply **only** to the nineteen project-trained `.pt` checkpoint files listed in [main_model/weights_index.csv](main_model/weights_index.csv) and [ablations/weights_index.csv](ablations/weights_index.csv). They do not license the repository source code, any third-party code, or the AGBD-Lite data.

**Permission from the ForestCarbon-Eye-BSAM project authors:** You may download, use, modify, and redistribute these checkpoint files solely for **noncommercial scientific research**, provided that you retain this notice, identify modifications, and cite *Footprint-Level Above-Ground Biomass Density Estimation via Multisource Fusion and Balanced Sharpness-Aware Training* together with the [AGBD-Lite dataset](https://doi.org/10.5281/zenodo.18485030) and its [parent AGBD publication](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025). Commercial use or redistribution for commercial purposes requires separate permission from the project authors. No trademark, patent, dataset, or upstream source-code rights are granted by this notice.

The [AGBD-Lite creators' dataset card](https://huggingface.co/datasets/prs-eth/AGBD-Lite) identifies their data as CC BY-NC 4.0. Obtain that dataset from its creators and comply with its own terms. These checkpoints were trained by this project and are not the upstream authors' pretrained baseline weights. This project does not claim authorship of AGBD-Lite or NicoNet.

The checkpoint format uses PyTorch serialization. Load files only from a trusted source; older training metadata may require serialization globals outside PyTorch's strict `weights_only` loader.
