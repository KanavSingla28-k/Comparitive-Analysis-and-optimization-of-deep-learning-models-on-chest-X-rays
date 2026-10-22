# Milestone tracker

Prepared: 22 October 2026. Update this file during subsequent execution.

**Current status:** Phases 0–8 are technically complete. The final report, independent statistics review, optimization note, runnable code and integrity-checked evidence package are ready for supervisor review. College formatting and submission remain handoff tasks.

**Existing evidence:** the dated pilot archive contains the earlier twelve-run study. Its measured outputs and previous verification record are available for review. Formal adoption of that evidence is tracked below; pending review does not mean the pilot never happened.

| Phase | Outcome | Current status | Available evidence / next action | Accepted evidence path | Completed date |
|---|---|---|---|---|---|
| 0 | Scope and evidence reuse decided | COMPLETE | Adopted pediatric binary benchmark; reuse pilot evidence subject to later acceptance checks | `01_planning/PHASE_0_DECISION_RECORD.md` | 22 October 2026 |
| 1 | Dataset/splits accepted | COMPLETE | Official metadata/checksum matched; indices verified; accepted 4,698 / 524 / 624 split | `03_data/PHASE_1_DATA_ACCEPTANCE.md` and `03_data/manifests/phase_1_verification.json` | 22 October 2026 |
| 2 | Environment/pipeline ready | COMPLETE | CUDA and versions verified; six smoke modes and three checkpoint/prediction checks passed | `04_implementation/PHASE_2_READINESS.md` | 22 October 2026 |
| 3 | Protocol frozen | COMPLETE | `cxr_compare_v1` settings/matrix fixed; eighteen files hash-pinned; metadata compatibility checked | `01_planning/PHASE_3_PROTOCOL.md` and `05_experiments/configs/protocol_freeze_v1.json` | 22 October 2026 |
| 4 | Nine model/seed comparison runs accepted | COMPLETE | Dataset/checkpoint/history/prediction checks passed; all twelve checkpoints replayed exactly in both precisions | `05_experiments/PHASE_4_RUN_ACCEPTANCE.md` | 22 October 2026 |
| 5 | Evaluation/statistics accepted | COMPLETE (technical) | 894 numerical checks and accepted outputs | `06_results/PHASE_5_STATISTICAL_REVIEW.md` | 22 October 2026 |
| 6 | Comparative report accepted | COMPLETE (technical) | Report, checked citations, team details and figures | `07_report/FINAL_COMPARATIVE_REPORT.md` | 22 October 2026 |
| 7 | Optimization proof of concept accepted | COMPLETE (technical) | Paired learning and disclosed hardware probes | `06_results/PHASE_7_OPTIMIZATION.md` | 22 October 2026 |
| 8 | Final package ready | COMPLETE (technical) | Evidence index, manifest, commands and next experiment | `08_handoff/START_HERE.md` | 22 October 2026 |

## Evidence inventory for review

All paths below are relative to `archive/pilot_2026-10-22/`.

| Evidence | Location |
|---|---|
| Earlier measured comparison report | `comparative_analysis_report.md` |
| Dataset checksum and duplicate findings | `artifacts/dataset_audit.json` |
| Environment | `artifacts/environment.json`, `requirements-lock.txt` |
| Forward/backward/freeze checks | `artifacts/smoke_test.json` |
| Protocol and source hash | `artifacts/protocol_lock.json` |
| Experiment defaults | `configs/experiment.json` |
| Statistical-analysis defaults | `configs/analysis_protocol.json` |
| Epoch histories and run manifests | `artifacts/history/` |
| Trained weights | `artifacts/checkpoints/` |
| Saved predictions | `artifacts/predictions/` |
| Per-run scores | `artifacts/metrics.csv` |
| Aggregate scores | `artifacts/comparison_summary.csv` |
| Bootstrap and comparison statistics | `artifacts/statistical_summary.json`, `artifacts/bootstrap_samples.npz` |
| Hardware measurements | `artifacts/benchmarks/` |
| Previous artifact verification | `artifacts/verification.json` |
| Implementation snapshot | `src/`, `run.ps1` |

## Comparison acceptance checklist

- [x] The title and objectives describe the pediatric pneumonia binary task.
- [x] Dataset provenance, label mapping, checksum, splits, and cleanup are accepted.
- [x] Patient-ID limitations and previously examined test scores are disclosed.
- [x] Three architectures have three usable seed runs under the accepted protocol.
- [x] Predictive metrics and hardware measurements use consistent definitions.
- [x] Means, seed variability, paired uncertainty, and individual seed scores are included.
- [x] Every table and figure traces to saved outputs.
- [x] Inconclusive architecture differences are described honestly.
- [x] The core comparative-analysis report is complete before optional expansion.

## Decision and deviation log

| Date/time IST | Phase | Decision or deviation | Reason | Evidence | Follow-up |
|---|---|---|---|---|---|
| 22 October 2026 | Preparation | Consolidate project files under Desktop\\irs and preserve the previous work as a dated pilot | User requested folder preparation and milestone phases without coding | Root README and archive inventory | Completed |
| 22 October 2026 | 0 | Adopt pediatric binary scope, the three-model comparison, and conditional reuse of pilot evidence | User requested Phase 0 completion; files and recorded source hashes inspected | `01_planning/PHASE_0_DECISION_RECORD.md` | Phase 1 dataset acceptance |
| 22 October 2026 | 1 | Accept canonical dataset and recorded retained split policy | Official checksum/labels verified and raw arrays/indices independently checked | `03_data/PHASE_1_DATA_ACCEPTANCE.md` | Phase 2 readiness |
| 22 October 2026 | 2 | Reuse existing CUDA runtime and reference pipeline | Six GPU smoke checks and representative checkpoint/serialization checks passed; archive preserved | `04_implementation/PHASE_2_READINESS.md` | Phase 3 protocol freeze |
| 22 October 2026 | 3 | Freeze `cxr_compare_v1` for existing benchmark evidence acceptance | Recipe/matrix aligned; eighteen files pinned; random latency-input wording clarified | `01_planning/PHASE_3_PROTOCOL.md` | Phase 4 run acceptance |

**Implementation extension:** at the user's request, created and executed the active pipeline, reproduced all twelve checkpoints and predictions exactly through full fresh training, regenerated analysis, and validated explained JSON/CSV/Markdown outputs. Details: `04_implementation/REPRODUCTION_VALIDATION.md`.

## Next action

Read `08_handoff/START_HERE.md`, review team details, demonstrate the code, apply the college template if required and present the completed comparison. The proposed batch-32 validation pilot is not executed.
