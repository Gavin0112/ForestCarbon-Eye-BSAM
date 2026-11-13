# Third-party notices

This repository contains no upstream repository, third-party pretrained model weights, raw dataset, satellite tile, or row-level prediction file. Its nineteen project-trained weights have [separate terms](../MODEL_WEIGHTS_TERMS.md). The root [MIT LICENSE](../LICENSE) covers original project source, configurations, documentation, aggregate result records and indices only; the project-trained checkpoints use [separate noncommercial research terms](../MODEL_WEIGHTS_TERMS.md). No upstream license is assigned to either by implication.

## AGBD-GFMs and NicoNet

The project model imports `nico_net_film.PointwiseBlock` from the external [AGBD-GFMs repository](https://github.com/ghjuliasialelli/AGBD-GFMs). The [official repository license](https://github.com/ghjuliasialelli/AGBD-GFMs/blob/main/LICENSE) states MIT, copyright © 2026 Ghjulia Sialelli, Linus Scheibenreif, Jan Dirk Wegner, Konrad Schindler. MIT permits modification and redistribution of its covered software if the copyright and permission notice remain with copies or substantial portions. The historical upstream commit is not identified by the retained experimental records. The upstream `nico_net_film.py`, SimpleFCN and UNet files are not packaged here. If code is later copied into this repository, review its actual file and retain the full MIT notice.

## AGBD-Lite and its parent publication

[AGBD-Lite](https://zenodo.org/records/18485030) is by Ghjulia Sialelli, Linus Scheibenreif, Jan Dirk Wegner and Konrad Schindler. Its [creators' dataset card](https://huggingface.co/datasets/prs-eth/AGBD-Lite) labels the dataset CC BY-NC 4.0. Cite both the dataset DOI 10.5281/zenodo.18485030 and the distinct parent [AGBD publication](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025), by Ghjulia Sialelli, Torben Peters, Jan Dirk Wegner and Konrad Schindler. These data and their creation are not this project's contribution. We do not redistribute them.

## Other named upstream projects

No PANGAEA code or configuration is present. The [PANGAEA repository](https://github.com/VMarsocci/pangaea-bench) is only research context; no file or historical commit has been tied to this repository. No CROMA code, configuration or `CROMA_base.pt` is present. The [CROMA code license](https://github.com/antofuller/CROMA/blob/main/LICENSE) is MIT, copyright © 2023 Anthony Fuller, but this does not by itself establish the weight's license. Review any future CROMA weight under its own repository terms before including it.

## Runtime packages

The requirements file names PyTorch, NumPy, Pandas, SciPy, h5py and PyTorch Lightning without pinned versions. They are external installs; no wheel, source distribution or vendored package is included. Licenses for a future locked environment must be checked against the **exact installed versions**. PyTorch Lightning is declared because the external AGBD-GFMs `nico_net_film.py` imports it; no direct import appears in this repository's source.
