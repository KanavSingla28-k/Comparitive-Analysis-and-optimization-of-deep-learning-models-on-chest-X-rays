# Accepted results and supporting evidence

Primary report: [final comparative analysis](../07_report/FINAL_COMPARATIVE_REPORT.md). Handoff: [start here](../08_handoff/START_HERE.md).

- [Independent statistical acceptance](PHASE_5_STATISTICAL_REVIEW.md), [894 checks](statistics/phase_5_checks.csv)
- [Accepted comparison CSV](tables/accepted_comparison.csv), [precision consistency](tables/precision_consistency.csv), [both hardware sessions](tables/hardware_sessions.csv)
- [Optimization proof](PHASE_7_OPTIMIZATION.md)
- [Final explained JSON](../07_report/final_results_explained.json), with descriptions beside results
- [Fresh-session report](retrained_v1/comparative_analysis_report.md), [pilot regeneration](reproduced_v1/comparative_analysis_report.md)

The original result bundles and SHA-256 sums are preserved. Complete checkpoints, raw predictions, timing samples and source provenance are in their corresponding `05_experiments/` directories. Inference wall-throughput denominators were not separately retained; rate aggregation and event timing summaries are checked, and this limitation is disclosed. No new model training was needed for the remaining acceptance phases.