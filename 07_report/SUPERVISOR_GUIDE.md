# Supervisor demonstration

The deliverable is executable Python plus measured artifacts. Start with [the pipeline and methodology](../04_implementation/PIPELINE.md), then [run instructions](../04_implementation/README.md).

Suggested demonstration sequence:

1. Show `04_implementation/src/settings.py` and `experiment.py`: model definitions, trainable stages, loss and training loop are directly inspectable.
2. Run `.\run.ps1 verify` from the project root. This recomputes scores and checks all twelve saved checkpoints and the accepted dataset.
3. Run `.\run.ps1 reproduce-results -RunDir '05_experiments/supervisor_demo_01'`. Use a new directory name if that one already exists. This executes every checkpoint on the real test set, recomputes statistics and regenerates the report.
4. Open [the regenerated comparative report](../06_results/reproduced_v1/comparative_analysis_report.md), [per-seed scores](../06_results/reproduced_v1/metrics.csv), [model comparison](../06_results/reproduced_v1/comparison_summary.csv), and the figure directory.
5. Open [explained JSON results](../06_results/reproduced_v1/results_explained.json). Numeric findings and their descriptions are stored together, with definitions and provenance.
6. Demonstrate full retraining with `.\run.ps1 full -RunDir '05_experiments/supervisor_training_01'`. Each new run gets its own checkpoints, logs, tables and explained JSON. This is optional for a live presentation because the full pipeline is already tested.

The regenerated saved-evidence report describes the original pilot measurements. Fresh-training evidence is stored separately under `05_experiments/retrained_v1` and, after completion, `06_results/retrained_v1`. Timing differences between sessions are expected. The verification/reproduction record summarizes what was actually executed.

Do not claim that a file format guarantees academic admissibility. The evidence is the source code, documented procedure, real data, trained checkpoints, saved predictions, logs and recomputable statistics. Confirm your college's required report format and disclosure rules yourself.
The complete fresh run has now passed. Read [executed reproduction validation](../04_implementation/REPRODUCTION_VALIDATION.md) and [the fresh-training report](../06_results/retrained_v1/comparative_analysis_report.md). All twelve checkpoint files and predictions matched the original exactly on this laptop; short hardware speed ratios varied between sessions and are reported separately.

## Completed technical handoff

Phases 5–8 add the [final comparative report](FINAL_COMPARATIVE_REPORT.md), [accepted statistics](../06_results/PHASE_5_STATISTICAL_REVIEW.md), [optimization note](../06_results/PHASE_7_OPTIMIZATION.md) and [evidence index](../08_handoff/EVIDENCE_INDEX.md). Team and supervisor details come from the user-supplied `PROJECT_DETAILS.md`; confirm the name/roll-number mapping before submission. No additional installation or retraining is needed for the demonstration.

Use `finalize.ps1 verify-package` to check file integrity. Use `finalize.ps1 all` to regenerate the final review package from saved evidence. The finalizer is readable Python and does not train models.