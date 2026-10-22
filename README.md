# IRS research — executable comparison

**Final technical package:** [completed report](07_report/FINAL_COMPARATIVE_REPORT.md), [handoff](08_handoff/START_HERE.md) and [optimization evidence](06_results/PHASE_7_OPTIMIZATION.md). Phases 0–8 are technically complete.

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

A college-level comparison of ResNet-18, DenseNet-121 and MobileNetV3-Small on pediatric PneumoniaMNIST+ at 224 pixels, with three seeds, a paired fine-tuning control and measured RTX 4050 resource trade-offs.

## Start here

- [Runnable commands and environment recreation](04_implementation/README.md)
- [Pipeline, methodology and code map](04_implementation/PIPELINE.md)
- [Supervisor demonstration guide](07_report/SUPERVISOR_GUIDE.md)
- [Regenerated comparison report](06_results/reproduced_v1/comparative_analysis_report.md)
- [Model comparison CSV](06_results/reproduced_v1/comparison_summary.csv) and [per-seed CSV](06_results/reproduced_v1/metrics.csv)
- [Results with descriptions in the same JSON](06_results/reproduced_v1/results_explained.json)

## Run from this folder

```powershell
.\run.ps1 verify
.\run.ps1 reproduce-results -RunDir '05_experiments/demo_01'
.\run.ps1 full -RunDir '05_experiments/fresh_01'
```

`verify` checks the existing evidence. `reproduce-results` executes all twelve saved checkpoints in both precisions and regenerates statistics, figures and reports. `full` starts from official pretrained weights and executes the complete training, evaluation and hardware pipeline. Use a new run-directory name when repeating adoption or preparation. The tested CUDA environment is selected automatically on this laptop.

The implementation is in `04_implementation/src/`. Training choices are directly readable in **`settings.py`**. JSON is used for stored evidence and its explanations; readable CSV, Markdown, code, checkpoint files and raw prediction arrays are supplied alongside it.

## Project map

| Folder | Contents |
|---|---|
| `01_planning/` | Audit, scope, evidence-adoption protocol and decisions |
| `02_literature/` | Literature materials |
| `03_data/` | Accepted split indices, manifests and dataset documentation |
| `04_implementation/` | Active Python pipeline, methodology, requirements and run instructions |
| `05_experiments/` | Versioned runs, full checkpoints, histories, predictions, raw measurements and provenance |
| `06_results/` | Regenerated reports, CSV tables, explained JSON and figures |
| `07_report/` | Supervisor guide and report materials |
| `archive/pilot_2026-10-22/` | Preserved original code, data, dependencies, pretrained weights and measured pilot evidence |

All new project files and outputs remain under this root. The existing shared Python installation is a runtime dependency; instructions for making a new local environment are included. The archive is a preserved input, while active output paths are explicit and versioned.

## Scope and evidence history

The source proposal is [initial.md](initial.md). Its broad adult multi-finding theme was narrowed to an achievable pediatric binary benchmark for the one-evening college project. No novel architecture or clinical validation is claimed. The test scores had already been examined before formal protocol adoption; neither replay nor retraining makes that test split newly blinded.

[Milestone tracker](MILESTONE_TRACKER.md) records formal progress. [Your next steps](YOUR_NEXT_STEPS.md) explains what to demonstrate and review. The original pilot and new reproduction results have distinct provenance, so rerun timing measurements should not be mixed into the original tables.