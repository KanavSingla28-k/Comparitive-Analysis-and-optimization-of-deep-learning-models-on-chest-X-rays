# Executed reproduction and implementation validation

Completed on 22 October 2026, using Python 3.13.7, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8, RTX 4050 Laptop GPU (6140.5 MiB), Windows, and NVIDIA driver 617.42.

## What was actually executed

| Check | Outcome |
|---|---|
| Dataset SHA-256 and independently recomputed cleanup membership | PASS; 4698 / 524 / 624 |
| Frozen Phase 3 files | All 18 unchanged |
| Pilot run integrity | All 12 checkpoints and five-epoch histories checked; all predictive metrics recomputed |
| Saved checkpoint replay | All 12 models executed on all 624 real test images in AMP and FP32; maximum difference from stored same-precision probabilities was zero |
| Analysis regeneration | All 2000 bootstrap draws, model predictive means/SD and paired intervals reproduced exactly |
| Full fresh training | Nine core configurations plus three paired controls completed from official pretrained weights |
| Fresh-versus-pilot checkpoints | All 12 PT files were byte-identical, checked by SHA-256 |
| Fresh-versus-pilot predictions | All saved prediction arrays were exactly equal, including FP32 probabilities, labels and indices |
| Full hardware stages | Inference and synthetic full-gradient training probes completed and saved raw samples |
| Explained JSON | Model, paired, hardware and per-run result sections contain descriptions beside numerical evidence |
| CSV/report/figures | Export hashes passed and all report figure links resolve |
| Output path containment | An outside-project run directory was rejected before writing |

Re-run the independent equality validator with:

```powershell
& 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' -B '04_implementation/src/validate_reproduction.py'
```

Machine-readable evidence: `06_results/reproduction_validation.json`. The validator itself is readable Python; its own source hash is recorded. It tests exact equality for these two named reproduction runs in this environment, not universal equality for a different machine.

## Where to review the outputs

| Evidence | Saved-evidence regeneration | Fresh training |
|---|---|---|
| Complete artifacts | `05_experiments/reproduced_v1/` | `05_experiments/retrained_v1/` |
| Report and CSV/JSON/figures | `06_results/reproduced_v1/` | `06_results/retrained_v1/` |
| Full execution log | Original pilot logs remain in the archive; replay verification is in the run artifacts | `05_experiments/retrained_v1/execution.log` |
| Corrected final report analysis log | `05_experiments/reproduced_v1/analysis_regeneration.log` | `05_experiments/retrained_v1/analysis_regeneration.log` |

The active code preserves the pilot recipe and makes its execution paths explicit. Launch-time source hashes are preserved in provenance. The final analysis regeneration separately records current analysis/export source hashes. One inherited narrative sentence was corrected after the fresh measurements: MobileNet's synthetic AMP step was slower in the pilot but faster in the fresh session. Numerical measurements were never replaced to match a narrative; the final report directs interpretation to its session-specific tables.

## Predictive reproduction versus timing variability

The predictive results reproduced exactly on this laptop:

| Configuration | Mean test AUROC | Seed SD |
|---|---|---|
| ResNet-18 partial fine-tuning | 0.988913 | 0.002105 |
| DenseNet-121 partial fine-tuning | 0.985516 | 0.001184 |
| MobileNetV3-Small partial fine-tuning | 0.982247 | 0.000756 |
| ResNet-18 frozen control | 0.969496 | 0.000877 |

The paired ResNet fine-tuning AUROC difference was +0.019417, exploratory 95% interval [0.011256, 0.029072]. Other declared architecture difference intervals include zero; these means do not establish a reliable universal ranking.

Hardware speed ratios differed between the original and fresh sessions:

| Architecture | Original synthetic batch-16 AMP speedup | Fresh synthetic batch-16 AMP speedup | Fresh allocation reduction |
|---|---|---|---|
| ResNet-18 | 2.037x | 1.299x | 25.3% |
| DenseNet-121 | 2.003x | 1.028x | 44.3% |
| MobileNetV3-Small | 0.859x | 2.959x | 34.8% |

A speedup below one means AMP was slower. These short probes have no independent run-to-run confidence interval and do not establish a stable architecture-specific training speedup. Do not pool the sessions' timing values into the predictive comparison. Both sessions found lower tensor allocation with AMP for these tested batch-16 workloads. All three full-gradient AMP batch-64 probes completed, including DenseNet at approximately 4.17 GiB of live tensor allocation; that is a tested fit, not a hardware maximum or guaranteed long-run memory budget.

Batch-1 FP32 forward latency was lower than AMP for all three architectures in both sessions. Preprocessing and transfer were outside those timings. Training time including validation changed from about 5.59 to 4.29 minutes after counting shared ResNet warm-up once. Setup, prediction replay, plots and hardware probing are additional wall time.

The existing test split had already been examined, and exact image hashes cannot establish patient independence. The result is a reproducible college benchmark, not a new clinical validation.