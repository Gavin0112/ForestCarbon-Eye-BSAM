# Reproduction status and limits

## What can be checked from this repository

- Inspect the exact model and BSAM source snapshots and their SHA-256 links to Review-12's frozen protocol.
- Inspect the six-arm identity, protocol descriptions and aggregate metric JSON, including checkpoint and prediction hashes.
- Recalculate metrics offline from a legitimately obtained archived prediction CSV with `tools/evaluate.py`; the bundled CSVs do not include row-level predictions or labels.

## What cannot yet be independently replayed

The official AGBD-Lite H5 files are distributed by their creators, not by this project. The historical scripts also require several large Train/Validation preprocessing caches, validation identity records and project-specific path conventions. This repository includes neither those assets nor a verified end-to-end cache builder. The upstream NicoNet version used for this adaptation is not bound to an immutable commit in the available evidence. Review-12, Review-13 and the later Review-12 continuation ran within the same server-side project. The delivered Review-13 archive lacks a byte-identical training entry, so the provided runner is a reconstructed interface; this does not imply a different server or a missing experiment. The archived training environment is not fully locked. Consequently, successful syntax checks or an offline metric check do **not** establish a from-scratch replication of all experiments.

## Data processing boundary

The frozen numerical path uses Sentinel-2 BOA offset handling and Train-side scaling; DEM height standardization and neighborhood terrain features; land-cover class/validity/confidence channels; ALOS DN-to-dB scaling, zero handling and texture eligibility; and eight metadata features. The paper reports these operations. The exact upstream DEM product identity, ALOS DN=1 semantics, original source-tile NoData handling and pixel-level cross-source registration remain unverified. Do not substitute a differently processed tensor and call it the paper's input.

## Test history

The reported Review-12, Review-13 and S2-only protocols select checkpoints on Validation and evaluate frozen checkpoints on Test. The larger historical project also contains earlier explorations that inspected Test results and influenced research decisions. The reported protocol does not describe Test as a never-seen blind holdout throughout the project. Do not use archived or newly computed Test scores to choose new preprocessing, masks or checkpoints.

## Next reproducibility gate

Before claiming independent training reproducibility, bind the upstream NicoNet commit, verify the full cache-generation and data-preparation workflow from legally obtained AGBD-Lite, recover or document the exact Review-13 server runner, record a complete software environment, and rerun a small real-data Train/Validation smoke test. A full experiment replay would require separate compute and should not be inferred from the offline checks provided here.
