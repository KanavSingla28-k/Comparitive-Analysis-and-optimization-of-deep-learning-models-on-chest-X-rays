# Evidence index

| Item | Location | Purpose |
| --- | --- | --- |
| Final report | [07_report/FINAL_COMPARATIVE_REPORT.md](../07_report/FINAL_COMPARATIVE_REPORT.md) | Fresh-session methods, comparison, costs, uncertainty, limitations and references |
| Explained results | [07_report/final_results_explained.json](../07_report/final_results_explained.json) | Measurements and descriptions together, plus accepted review results |
| Training code | [04_implementation/src/experiment.py](../04_implementation/src/experiment.py) | Models, preprocessing, trainable stages, loss, training, prediction and timing |
| Python settings | [04_implementation/src/settings.py](../04_implementation/src/settings.py) | Models, seeds and hyperparameters in inspectable Python |
| Analysis | [04_implementation/src/analyze.py](../04_implementation/src/analyze.py) | Bootstrap, model scores, original plots and report |
| Independent review | [04_implementation/src/review_statistics.py](../04_implementation/src/review_statistics.py) | Weighted-pair bootstrap and raw measurement checks |
| Finalizer | [04_implementation/src/finalize_project.py](../04_implementation/src/finalize_project.py) | Acceptance and packaging code; no model retraining |
| Run commands | [run.ps1](../run.ps1) | Full and staged training or saved-evidence reproduction |
| Final review commands | [finalize.ps1](../finalize.ps1) | Independent statistics, report assembly and integrity verification |
| Run instructions | [04_implementation/README.md](../04_implementation/README.md) | Environment installation, data, weights and commands |
| Pipeline methods | [04_implementation/PIPELINE.md](../04_implementation/PIPELINE.md) | Readable experimental flow |
| Reproduction proof | [04_implementation/REPRODUCTION_VALIDATION.md](../04_implementation/REPRODUCTION_VALIDATION.md) | Exact weights and probabilities in the tested environment |
| Statistical review | [06_results/PHASE_5_STATISTICAL_REVIEW.md](../06_results/PHASE_5_STATISTICAL_REVIEW.md) | Uncertainty, verification and throughput limitations |
| Numeric checks | [06_results/statistics/phase_5_checks.csv](../06_results/statistics/phase_5_checks.csv) | One row per check with discrepancy and tolerance |
| Accepted comparison | [06_results/tables/accepted_comparison.csv](../06_results/tables/accepted_comparison.csv) | Three core architectures plus a frozen control |
| Precision comparison | [06_results/tables/precision_consistency.csv](../06_results/tables/precision_consistency.csv) | Identical-real-input probability and decision changes |
| Hardware sessions | [06_results/tables/hardware_sessions.csv](../06_results/tables/hardware_sessions.csv) | Two separate measured sessions |
| Optimization proof | [06_results/PHASE_7_OPTIMIZATION.md](../06_results/PHASE_7_OPTIMIZATION.md) | Measured learning, memory, timing and capacity evidence |
| Next experiment | [05_experiments/NEXT_OPTIMIZATION_EXPERIMENT.md](../05_experiments/NEXT_OPTIMIZATION_EXPERIMENT.md) | Validation-only batch-32 plan with fixed criteria; not executed |
| Canonical dataset | [archive/pilot_2026-10-22/data/pneumoniamnist_224.npz](../archive/pilot_2026-10-22/data/pneumoniamnist_224.npz) | Actual 224-pixel data, checksum verified |
| Clean membership | [03_data/manifests/train_indices.npy](../03_data/manifests/train_indices.npy) | Accepted training indices; validation/test indices adjacent |
| Fresh provenance | [05_experiments/retrained_v1/provenance.json](../05_experiments/retrained_v1/provenance.json) | Launch and final-analysis source hashes |
| Fresh execution log | [05_experiments/retrained_v1/execution.log](../05_experiments/retrained_v1/execution.log) | Actual full-run console record |
| Dependencies | [04_implementation/requirements-lock.txt](../04_implementation/requirements-lock.txt) | Exact measured versions; preserved offline dependencies also retained |
| Citation review | [02_literature/VERIFIED_SOURCES.md](../02_literature/VERIFIED_SOURCES.md) | Verified primary-source scope and relevance |
| Project details | [01_planning/PROJECT_DETAILS.md](../01_planning/PROJECT_DETAILS.md) | User-supplied team and supervisor details |

Complete checkpoints, NPZ predictions, bootstrap draws and raw timing samples are under `05_experiments/retrained_v1/artifacts/`. The pilot remains identifiable. The manifest includes the preserved dependencies and dataset; source-only Git excludes binary artifacts.
