# Milestone tracker

Prepared: 22 October 2026. Update this file during subsequent execution.

**Current status:** Phases 0, 1, and 2 completed on 22 October 2026. Scope, dataset acceptance, and current GPU/pipeline readiness are recorded. Phase 2 used six single-minibatch smoke updates on disposable model copies; no new comparison training or test inference was performed.

**Existing evidence:** the dated pilot archive contains the earlier twelve-run study. Its measured outputs and previous verification record are available for review. Formal adoption of that evidence is tracked below; pending review does not mean the pilot never happened.

| Phase | Outcome | Current status | Available evidence / next action | Accepted evidence path | Completed date |
|---|---|---|---|---|---|
| 0 | Scope and evidence reuse decided | COMPLETE | Adopted pediatric binary benchmark; reuse pilot evidence subject to later acceptance checks | `01_planning/PHASE_0_DECISION_RECORD.md` | 22 October 2026 |
| 1 | Dataset/splits accepted | COMPLETE | Official metadata/checksum matched; indices verified; accepted 4,698 / 524 / 624 split | `03_data/PHASE_1_DATA_ACCEPTANCE.md` and `03_data/manifests/phase_1_verification.json` | 22 October 2026 |
| 2 | Environment/pipeline ready | COMPLETE | CUDA and versions verified; six smoke modes and three checkpoint/prediction checks passed | `04_implementation/PHASE_2_READINESS.md` | 22 October 2026 |
| 3 | Protocol frozen | Proposed defaults documented | Match proposed protocol to adopted evidence | — | — |
| 4 | Nine model/seed comparison runs accepted | Pilot runs available; formal acceptance pending | Inspect manifests, histories, predictions, and checkpoint records | — | — |
| 5 | Evaluation/statistics accepted | Pilot analysis available; formal acceptance pending | Review metrics, bootstrap method, timing scopes, and figures | — | — |
| 6 | Comparative report accepted | Pilot report available; final report pending | Apply report acceptance checklist | — | — |
| 7 | Optimization proof of concept accepted | Pilot evidence available; formal acceptance pending | Review paired control and precision/capacity measurements | — | — |
| 8 | Final package ready | Pending | Complete after comparison acceptance | — | — |

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
- [ ] Three architectures have three usable seed runs under the accepted protocol.
- [ ] Predictive metrics and hardware measurements use consistent definitions.
- [ ] Means, seed variability, paired uncertainty, and individual seed scores are included.
- [ ] Every table and figure traces to saved outputs.
- [ ] Inconclusive architecture differences are described honestly.
- [ ] The core comparative-analysis report is complete before optional expansion.

## Decision and deviation log

| Date/time IST | Phase | Decision or deviation | Reason | Evidence | Follow-up |
|---|---|---|---|---|---|
| 22 October 2026 | Preparation | Consolidate project files under Desktop\\irs and preserve the previous work as a dated pilot | User requested folder preparation and milestone phases without coding | Root README and archive inventory | Completed |
| 22 October 2026 | 0 | Adopt pediatric binary scope, the three-model comparison, and conditional reuse of pilot evidence | User requested Phase 0 completion; files and recorded source hashes inspected | `01_planning/PHASE_0_DECISION_RECORD.md` | Phase 1 dataset acceptance |
| 22 October 2026 | 1 | Accept canonical dataset and recorded retained split policy | Official checksum/labels verified and raw arrays/indices independently checked | `03_data/PHASE_1_DATA_ACCEPTANCE.md` | Phase 2 readiness |
| 22 October 2026 | 2 | Reuse existing CUDA runtime and reference pipeline | Six GPU smoke checks and representative checkpoint/serialization checks passed; archive preserved | `04_implementation/PHASE_2_READINESS.md` | Phase 3 protocol freeze |

## Next action

Execute **Phase 3 — freeze the comparison protocol**. Environment and pipeline readiness are accepted. Preserve the existing benchmark evidence and bind versioned active paths before any new formal run.
