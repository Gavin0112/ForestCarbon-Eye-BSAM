# Review-13 Runner and Archived Results

Authoritative delivery: `archives/REVIEW13_DELIVERY_20261103.tar.gz`. The six intervention arms are M1–M6; M0 reuses the Review-12 seed-43 BSAM checkpoint from the same server-side project. Review-12, Review-13, and the later Review-12 resume belong to the same server-side project and were packaged separately for delivery. The delivered Review-13 archive does not include the full training entry used at the time.

| Arm | Definition | Archived Test RMSE (Mg/ha) |
| --- | --- | ---: |
| M1 | Zero S2 and acquisition date | 62.8947 |
| M2 | Zero DEM patch and terrain metadata | 60.3787 |
| M3 | Zero land cover | 58.1394 |
| M4 | Zero scaled ALOS values and raw DN | 58.1887 |
| M5 | Skip the FiLM conditioning mechanism | 58.9326 |
| M6 | Zero all eight metadata values while retaining FiLM | 58.6989 |

`ABLATION_CONFIGS/PROTOCOL_TEMPLATE.json` is a planning template with execution disabled.
