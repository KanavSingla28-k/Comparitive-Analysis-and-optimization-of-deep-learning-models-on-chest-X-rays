# IRS research — start here

Project home: `C:\Users\kanaa\OneDrive\Desktop\irs`.

**Current priority:** complete and document the three-model comparative analysis, then include a small, measured optimization proof of concept. This preparation step creates the folder structure and milestone documents. It does not implement code, run training, or generate new experimental results.

## Start with these documents

1. [Milestone phases](MILESTONES.md) — tasks, deliverables, dependencies, and completion criteria.
2. [Milestone tracker](MILESTONE_TRACKER.md) — execution checklist and evidence log.
3. [Scope and decisions](01_planning/SCOPE_AND_DECISIONS.md) — proposed study defaults and decisions to freeze before further experiments.
4. [Original proposal](initial.md) — preserved source document.

## Folder map

| Folder | Purpose |
|---|---|
| `01_planning/` | Scope, decisions, previous audit, and original implementation plan |
| `02_literature/` | Reading notes and citation records |
| `03_data/` | Active-study dataset documentation and audited split records |
| `04_implementation/` | Future implementation work; currently a preparation note only |
| `05_experiments/` | Future run configurations, logs, and checkpoints |
| `06_results/` | Accepted comparison tables, figures, and statistical analysis |
| `07_report/` | College report draft and submission materials |
| `archive/pilot_2026-10-22/` | Preserved earlier code, dependencies, dataset, pretrained weights, checkpoints, reports, and measured outputs |

The dated archive is a snapshot of the earlier workspace, excluding its Git metadata and this temporary folder-preparation staging area. The original `initial.md` at the project root is retained. The previous workspace is kept as a recovery copy.

## Status and reuse

The archive contains an earlier completed pilot comparison with twelve model/seed configurations and verification records. Those files are evidence available for review; the new tracker does not silently treat every formal milestone as signed off. Review and reuse satisfactory evidence rather than automatically repeating expensive work.

The archived test results have already been examined. Any new study using that same test split must disclose this history; do not describe it as a newly untouched test set. New optimization decisions should use validation data. A new confirmatory generalization claim needs an appropriate untouched evaluation set.

All project materials are consolidated under this project home. The shared CUDA Python environment remains at `C:\Users\kanaa\Documents\Codex\ml-gpu-env`; it is a system resource referenced by the pilot, not relocated during folder preparation.

**Phases 0, 1, and 2 are complete.** Read [the scope decision](01_planning/PHASE_0_DECISION_RECORD.md), [dataset acceptance](03_data/PHASE_1_DATA_ACCEPTANCE.md), [pipeline readiness](04_implementation/PHASE_2_READINESS.md), and [your checklist](YOUR_NEXT_STEPS.md). The next milestone is **Phase 3 — freeze the comparison protocol**.

**Root policy:** all subsequent project work and outputs stay under `C:\Users\kanaa\OneDrive\Desktop\irs`. The previous Documents workspace is retained only as a recovery copy.
