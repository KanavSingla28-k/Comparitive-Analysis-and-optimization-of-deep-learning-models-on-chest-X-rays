# Implementation milestones

Project: comparative evaluation of pretrained CNNs for pediatric pneumonia classification.

Prepared: 22 October 2026. Project home: `C:\Users\kanaa\OneDrive\Desktop\irs`.

## Execution priorities

**Required outcome:** a defensible comparison of ResNet-18, DenseNet-121, and MobileNetV3-Small, with comparable training conditions, predictive results, resource measurements, uncertainty, and a written interpretation.

**Secondary outcome:** a small optimization proof of concept with measurements, followed by a short explanation of the tested hardware envelope. Do not let optional optimization prevent completion of the comparison.

The earlier pilot is preserved in `archive/pilot_2026-10-22/`. Review its evidence before deciding which tasks need new execution. A milestone can be completed by accepting verified existing evidence; rerunning everything is not a default requirement. This document is a plan for subsequent work. No implementation or experimentation is authorized by an unchecked box alone.

## Phase overview

| Phase | Milestone | Depends on | Priority | Planning allowance if work is needed |
|---|---|---|---|---|
| 0 | Scope and evidence review | None | Required | 10 minutes |
| 1 | Dataset and split acceptance | 0 | Required | 20 minutes |
| 2 | Environment and pipeline readiness | 1 | Required | 30–45 minutes |
| 3 | Freeze the comparison protocol | 0–2 | Required | 10 minutes |
| 4 | Complete the model comparison runs | 3 | Required | 20–40 minutes, guided by measured runtime |
| 5 | Comparative evaluation and statistics | 4 | Required | 25–35 minutes |
| 6 | Comparison report and acceptance | 5 | Required | 25–35 minutes |
| 7 | Limited optimization proof of concept | 6; design specified in 3 | Secondary | 20–30 minutes |
| 8 | Final package and handoff | 6; include 7 if completed | Required | 10–15 minutes |

These are work allocations, not runtime guarantees. Existing pilot outputs can shorten several phases. Use relative timings rather than the superseded clock schedule in the original implementation plan. Finish Phases 0–6 first when working to the end-of-day deadline.

## Phase 0 — Scope and evidence review

**Completed on 22 October 2026.** See `01_planning/PHASE_0_DECISION_RECORD.md`. Remaining phases retain their acceptance checks.

**Purpose:** agree exactly what this college project will claim and identify reusable work.

**Inputs:** `initial.md`, the reaudit, the original implementation plan, and the archived pilot report and verification records.

**Tasks:**

- Read the revised task definition: pediatric pneumonia versus normal, rather than adult fourteen-finding classification.
- Confirm the three architectures and the RTX 4050 laptop resource constraint.
- Review the pilot report and its saved verification record; identify evidence that can be adopted, evidence requiring inspection, and work requiring a new run.
- Record that the previous test scores have been seen. Decide whether the next deliverable is a documented reproduction of that benchmark or a study with a new evaluation set.
- Choose a final comparison protocol before further model selection; leave architecture invention and broad compression searches outside the current scope.

**Deliverables:** completed decision record in `01_planning/`, plus a reuse/rerun note in the milestone tracker.

**Complete when:** the task, model list, dataset, evidence status, and permitted claims are explicit. No new experiment is needed for this phase.

## Phase 1 — Dataset and split acceptance

**Completed on 22 October 2026.** See `03_data/PHASE_1_DATA_ACCEPTANCE.md` and the verified manifests in `03_data/manifests/`.

**Purpose:** establish the data provenance and the exact examples used by the accepted study.

**Tasks:**

- Review the archived PneumoniaMNIST+ 224 file, official source, and checksum record before considering another download.
- Confirm the normal/pneumonia label mapping, resolution, and split counts.
- Review exact-duplicate findings. The pilot excluded ten training examples identical to validation images, leaving 4,698 training, 524 validation, and 624 test images.
- Record the three duplicate excess images within the preserved test split and the absence of patient IDs in the packaged arrays.
- Preserve original data; keep any accepted filtering as documented index records.
- Carry the split policy into the formal data note. Do not silently randomize or merge the existing splits.

**Deliverables:** accepted data note and split manifest in `03_data/`; links to the original dataset and accepted audit/index files.

**Complete when:** dataset version, checksum, labels, working counts, overlap checks, and patient-level limitations are documented. Use the archived dataset directly during review; unnecessary duplicate copies are not required.

## Phase 2 — Environment and pipeline readiness

**Completed on 22 October 2026.** See `04_implementation/PHASE_2_READINESS.md` and the current verification records.

**Purpose:** establish that the later implementation can run consistently on the available hardware.

**Tasks for a future execution turn:**

- Review the recorded CUDA environment and exact library versions.
- Verify the selected Python interpreter uses the RTX 4050 and that required imports and a model training step succeed.
- Review the archived pipeline before deciding whether to reuse or reorganize it.
- Confirm consistent image preprocessing, one-logit binary outputs, loss/label shapes, and model initialization.
- Confirm which weights update in head-only and partial fine-tuning modes; verify frozen batch-normalization statistics.
- Check checkpoint loading and prediction saving before scheduling comparison runs.
- Keep the first formal batch at 16 unless a common alternative is chosen before all runs.

**Deliverables:** environment record, pipeline-readiness checklist, and implementation-location decision. Any future active implementation belongs in `04_implementation/`; preserve the pilot snapshot.

**Complete when:** readiness evidence covers all three architectures. Folder preparation does not execute these checks or write implementation code.

## Phase 3 — Freeze the comparison protocol

**Completed on 22 October 2026.** Protocol `cxr_compare_v1` is adopted and hash-pinned; see `01_planning/PHASE_3_PROTOCOL.md`.

**Purpose:** prevent changing experimental choices in response to test scores.

**Proposed defaults to adopt or revise before execution:**

| Decision | Proposed setting |
|---|---|
| Models | ResNet-18, DenseNet-121, MobileNetV3-Small |
| Initialization | Recorded ImageNet-1K V1 weights |
| Input | Actual 224-resolution images, three repeated grayscale channels, ImageNet normalization |
| Augmentation | Training only: rotation within ±5° |
| Training schedule | Two head-only epochs, then three partial fine-tuning epochs |
| Seeds | Three prespecified seeds; 42, 43, 44 for exact pilot reproduction |
| Batch size | 16 across models |
| Optimizer | AdamW; head LR 1e-3, adaptation LR 1e-4, weight decay 1e-4 |
| Loss | Weighted binary cross entropy; weight from training labels only |
| Precision | Recorded CUDA AMP training configuration |
| Primary metric | Per-checkpoint test AUROC, summarized across seeds |
| Other metrics | AP, F1, precision, recall, specificity, accuracy, balanced accuracy |
| Threshold | 0.5, fixed before evaluation |
| Checkpoint | Fixed final epoch |
| Costs | Parameters, weights size, training duration, latency, and GPU allocation |
| Uncertainty | Seed variability plus paired bootstrap; account for exact-image test duplicates |

**Tasks:** freeze the run matrix, metric definitions, timing scopes, uncertainty method, resource settings, and stopping rules. Specify the optional ResNet frozen control and FP32/AMP probes now, before using further results to select them.

**Deliverables:** dated protocol decision record in `01_planning/` and an experiment manifest in `05_experiments/` when execution begins.

**Complete when:** all runs can be identified in advance and existing evidence can be matched to this protocol. Changed settings require a new experiment version; do not mix them into one ranking.

## Phase 4 — Complete the comparative runs

**Completed:** all nine core runs accepted; all twelve pilot checkpoints replayed exactly. A separate fresh full run also reproduced all original checkpoint files and test probabilities exactly. See `05_experiments/PHASE_4_RUN_ACCEPTANCE.md` and `04_implementation/REPRODUCTION_VALIDATION.md`.

**Purpose:** obtain all required comparison evidence for the three architectures.

**Required matrix:** three models × three seeds = **nine comparison checkpoints**. The extra three ResNet frozen-control checkpoints support Phase 7, giving twelve configurations if the proof of concept is included.

**Tasks:**

- For accepted pilot reuse, review all nine comparison manifests, histories, predictions, and checkpoint hashes against the frozen protocol.
- For new execution, run the same accepted recipe for every model/seed pair and preserve epoch logs, durations, memory records, and final checkpoints.
- Preserve each ResNet epoch-2 checkpoint if the paired optimization experiment will be completed later.
- Keep training exposure comparable. Label interrupted or invalid runs rather than quietly comparing different budgets.
- Capture failures and deviations; repair only genuine execution errors and rerun affected comparisons consistently.

**Deliverables:** nine accepted run records, associated checkpoints and histories, and a completed matrix in `05_experiments/`.

**Complete when:** every required cell has usable evidence, settings match, and incomplete runs are explicitly resolved. High scores are not an acceptance criterion.

## Phase 5 — Comparative evaluation and statistics

**Purpose:** turn model outputs into an auditable comparison.

**Tasks:**

- Produce or adopt fixed-checkpoint predictions under the protocol's evaluation precision and batch settings.
- Calculate AUROC and AP from probabilities; calculate thresholded metrics at the fixed threshold.
- Summarize all three seeds with means and seed standard deviations. Preserve individual seed rows.
- Calculate paired uncertainty for declared model differences; distinguish test-sample uncertainty from limited seed variability.
- Review latency and resource measurement scopes. Separate single-image latency, batched throughput, and training cost.
- Build ROC/PR curves, confusion matrices, training curves, and a performance–cost comparison.
- Check tables against predictions and manifests; do not insert published literature scores into the local results table.

**Deliverables:** accepted per-run and summary tables, statistical note, figures, and evidence-check record in `06_results/`.

**Complete when:** all three models are compared using the same definitions, uncertainty is visible, and every reported number has a traceable source. Do not claim equivalence or superiority solely from close point estimates.

## Phase 6 — Complete the comparative-analysis report

**Purpose:** finish the core college-project deliverable before optional optimization work.

**Tasks:**

- Write the aim, corrected related work, dataset, shared protocol, predictive results, resource results, and limitations.
- Explain why AUROC and F1 can favor different models.
- Explain the observed parameter/storage/latency trade-off without assuming that fewer parameters guarantees faster inference.
- Include the previously examined test-set history, duplicate cleanup, missing patient IDs, short schedule, and single-dataset scope.
- Separate empirical observations from recommendations. Avoid clinical or broad generalization claims.
- Check that the title and abstract describe the pediatric binary task actually studied.

**Deliverables:** comparison report and figure/table references in `07_report/`.

**Complete when:** the report answers which models performed best on which metrics, at what measured cost, under what uncertainty, and with what limitations. This is the **comparative-analysis completion gate**.

## Phase 7 — Limited optimization proof of concept

**Purpose:** demonstrate a measured optimization direction without making it the whole project.

**A. Learning optimization:** compare ResNet head-only continuation with partial fine-tuning from the same epoch-2 checkpoint, matching seeds, data order, augmentation, learning rate, and additional epochs. Summarize paired AUROC differences and uncertainty.

**B. Hardware optimization:** compare FP32 and AMP on the same weights and inputs. Measure numerical differences, latency, memory, and selected batch capacities. Label synthetic full-gradient timing probes as microbenchmarks rather than end-to-end training improvements.

**Tasks:** accept and clearly cite suitable archived proof-of-concept evidence or execute only the prespecified missing measurements. Report speedups below 1× as slowdowns. State the largest tested successful workload rather than claiming an absolute GPU limit.

**Deliverables:** a short optimization note, paired statistics, resource table, and a next-experiment recommendation.

**Complete when:** at least one optimization has measured evidence and limitations. Compression, quantization, distillation, architecture search, and cross-hospital studies remain later extensions.

## Phase 8 — Package and hand off

**Purpose:** make the accepted project easy to review and continue.

**Tasks:**

- Complete the milestone tracker with actual dates, evidence paths, decisions, and deviations.
- Confirm report tables and figures agree with the accepted outputs.
- Include source references, environment information, protocol version, and reproduction instructions for the accepted implementation.
- Keep pilot material identifiable and preserve accepted study artifacts without overwriting them.
- Record one concrete next optimization experiment and its acceptance criterion.

**Deliverables:** final report, evidence index, completed tracker, and handoff note.

**Complete when:** a reviewer can understand the comparison and locate the evidence without relying on chat history.

## Defer when time is short

Protect the three-model comparison, consistent metrics, traceable outputs, and report. Defer extra architectures, extensive tuning, application UI work, new compression methods, large datasets, and external validation to a later study. An incomplete optional probe should be labeled as such, not presented as completed optimization.
