# Phase 0 — completed scope and evidence decision

Completed: 22 October 2026 (IST).

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

## Outcome

Phase 0 is complete. The study scope and evidence-reuse route below are adopted under the user's instruction to complete this phase. College-specific administrative requirements remain to be supplied when available; no supervisor approval is claimed.

All subsequent project documents, implementation, configurations, data copies, caches, results, and reports belong under this root. The former Documents workspace is a recovery copy and will not be used for new project work. Existing shared system tools may be read or used; changes to project-controlled environments and dependencies should be made under the project root rather than altering an unrelated shared setup.

## 1. Adopted study scope

**Working title:** Comparative Analysis of Pretrained CNNs for Pediatric Pneumonia Classification with an Initial Efficiency Study.

**Research question:** Under a common, limited-budget training protocol, how do ResNet-18, DenseNet-121, and MobileNetV3-Small compare in pneumonia-versus-normal classification and resource use, and what measurable benefit does a small optimization experiment provide?

**Primary aim:** complete a reproducible college-level model comparison. **Secondary aim:** demonstrate one limited optimization direction with measurements and uncertainty.

| Decision | Adopted scope |
|---|---|
| Task | Binary pediatric pneumonia versus normal |
| Dataset | PneumoniaMNIST+, actual 224 × 224 release |
| Architectures | ResNet-18, DenseNet-121, MobileNetV3-Small |
| Hardware | Existing RTX 4050 Laptop GPU, approximately 6 GB VRAM |
| Study route | Review and adopt suitable existing benchmark evidence; reproduce only work that fails an acceptance criterion or is required for a documented change |
| Novelty | An educational empirical comparison; no new architecture is required |
| Priority | Finish comparison and report before expanding optimization |
| Working completion target | Tonight, 22 October 2026; official college submission time remains an administrative detail |

The pediatric binary scope is a deliberate narrowing of the original broader chest X-ray proposal. An official title fixed by the college can retain its wording with an accurate scope statement or subtitle; it does not justify describing these data as adult fourteen-finding results.

## 2. Reference protocol to carry into Phase 3

Use the archived protocol as the reference: ImageNet-1K V1 weights; two head-only epochs plus three partial fine-tuning epochs; three seeds (42, 43, 44 for exact reproduction); batch 16; the recorded shared preprocessing and weighted binary loss; fixed final checkpoints; threshold 0.5.

Primary endpoint: AUROC. Secondary endpoints: average precision, F1, precision, recall, specificity, accuracy, and balanced accuracy. Resource endpoints: parameter count, weights size, training duration, inference latency/throughput, and GPU allocation.

The core matrix is three architectures × three seeds = nine comparison runs. Three paired ResNet frozen controls support the optimization study. Scope is fixed here; Phase 3 still checks the detailed recipe, versioning, statistical definitions, and timing scopes before any new execution. Choosing this reference now is not a claim of retrospective preregistration for a new study.

## 3. Evidence review performed for Phase 0

The following checks inspected existing files only. No training, inference, dataset processing, or statistical recomputation was performed during Phase 0.

| Evidence | Observation in the archived snapshot | Decision |
|---|---|---|
| Run manifests | Twelve manifests: nine partial fine-tuning comparison runs and three frozen ResNet controls; each records five epochs | Eligible for reuse; detailed acceptance in Phase 4 |
| Checkpoints and predictions | Both files present for all twelve manifests | Preserve and review; no default retraining |
| Training source | SHA-256 matches the archived protocol lock | Reference implementation can be reused after Phase 2 readiness review |
| Analysis source | SHA-256 matches the archived analysis manifest | Reference analysis can be reviewed in Phase 5 |
| Prior verification record | Reports twelve checkpoint checks, independent AUROC/F1 checks, shared initialization checks, and zero exact cross-split overlaps after cleanup | Accept as prior verification evidence, not as a newly executed audit |
| Dataset record | Records 4,698 / 524 / 624 retained examples and missing patient IDs | Phase 1 must accept provenance, split policy, and limitations |
| Comparison/statistics | Per-run scores, summary scores, bootstrap outputs, and hardware records available | Adopt only after checking definitions and interpretation in Phase 5 |
| Earlier report | Complete pilot report is available | Use as a draft and evidence reference; final report acceptance remains Phase 6 |

Current training-source SHA-256: `8e40d09dc07cdbe8ebd335d876feac7bfc8330204e54a3f8b95e7f8db49dafd1`.

Current analysis-source SHA-256: `4795ee36a18a0c545053ff18d076940423f4296f4d00df81dc811812bf75ff7f`.

## 4. Reuse, inspect, or rerun

**Reuse now as reference assets:** original proposal, reaudit, literature references, dataset file, recorded environment, existing code snapshot, configurations, saved model weights, predictions, logs, figures, and numerical results. Preserve the dated archive.

**Inspect before formal acceptance:** dataset checksum/split policy (Phase 1), runtime and implementation readiness (Phase 2), detailed protocol consistency (Phase 3), run-level evidence (Phase 4), uncertainty and resource-measurement interpretation (Phase 5), and final report claims (Phase 6).

**Rerun only for a concrete reason:** a failed integrity check, missing artifact, inconsistent protocol, reproducibility requirement, genuine execution error, or an explicitly versioned new experiment. A different or more attractive score is not a reason to rerun selectively.

**Deferred:** new architectures, large hyperparameter searches, broad compression studies, adult multi-label classification, cross-hospital generalization, and application UI work.

## 5. Evaluation-history decision

The pilot test scores were already examined. The next deliverable is a documented benchmark study using that disclosed evidence history. It is not a newly blinded confirmatory test.

Do not use additional test inspection to choose model settings or thresholds. Any new development choices should use validation evidence. If a later claim requires untouched external validation, design a separate evaluation with an appropriate new dataset. A new external dataset is not required to finish the current educational comparison.

The protocol lock's historical `test_predictions_generated: false` field describes its state when originally written. It does not mean the test remains unseen now.

## 6. Permitted conclusions

- Describe observed performance and measured costs for these pretrained packages under this benchmark protocol.
- Report seed variability and exploratory paired uncertainty; preserve inconclusive model differences.
- Discuss partial fine-tuning and mixed precision using the measured scope of each experiment.
- State which tested workloads fit the GPU, without claiming an absolute optimization or memory ceiling.

Do not infer clinical readiness, adult or cross-hospital accuracy, universal architecture superiority, statistical equivalence from close scores, or guaranteed long-run acceleration from a short synthetic timing probe.

## 7. Phase 0 exit checklist

- [x] Project root is explicit.
- [x] Working title, task, dataset, model list, and hardware are adopted.
- [x] Comparison has priority over optimization.
- [x] Existing evidence has been inventoried and its reuse route decided.
- [x] The previously examined test-set history is recorded.
- [x] Reference protocol and permitted claims are explicit.
- [x] Later acceptance checks and concrete rerun triggers are identified.
- [x] User-owned next actions are written in `YOUR_NEXT_STEPS.md`.

**Next technical milestone:** Phase 1 — dataset and split acceptance. Only Phase 0 is marked complete by this decision record.