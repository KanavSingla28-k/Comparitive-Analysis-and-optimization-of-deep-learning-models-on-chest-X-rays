# Phase 5 — evaluation and statistics accepted

894 numerical checks passed across both sessions. Primary report session: `retrained_v1`; the pilot regeneration remains separate reproduction evidence.

An independent implementation rebuilt all 2,000 paired bootstrap draws using integer case multiplicities and positive-negative pair counting with half-credit for tied probabilities, rather than calling the original rank-sum bootstrap. The draws matched within absolute tolerance 1e-14. Means, sample SD (ddof=1), interval endpoints, per-seed confusion counts and precision discrepancies were recomputed from real prediction arrays.

Accepted tables are in `06_results/tables/`; detailed checks and independent draws are in `06_results/statistics/`. Figure sources are mapped in `figure_provenance.csv`. Original and fresh predictive results agree; hardware measurements remain separate sessions.

The primary endpoint is AUROC. Other metrics use threshold 0.5. Three models are compared using three seeds each; frozen ResNet is an extra paired control. Aggregate scores are means of per-model scores, not a probability ensemble. SD across three seeds and the exploratory seed-and-image-cluster bootstrap represent different variability summaries. Unknown patient dependence remains unmeasured; intervals are not multiplicity adjusted.

The ResNet fine-tuning interval is above zero within this benchmark. Both other declared architecture-difference intervals include zero, so close means do not establish ordering or equivalence. No literature scores are mixed into local result tables.

Raw CUDA-event medians, means, p95 and interleaved precision summaries were checked. Synthetic synchronized training-step medians and throughput were independently recomputed. Inference wall-throughput denominators were not separately saved; those rates can be traced to raw measurement records and code, while aggregation is verified against the recorded rates. Event sums must not be substituted for synchronized wall duration.

Five original result figures have code/source mappings. They use seed 42 for representative test curves/confusion matrices and all seeds for training/summary plots. No extra GPU experiment was needed for acceptance.
