# Phase 4 — comparison runs accepted

All nine core model/seed runs pass `cxr_compare_v1` acceptance: three architectures, three seeds, fixed five-epoch training. The three frozen ResNet controls passed the same integrity checks; optimization interpretation remains separately reviewable.

The active Python pipeline independently verified the dataset checksum and accepted split membership; all checkpoint SHA-256 values matched manifests; histories contained the correct head/partial schedule; all raw test probabilities were finite and in [0,1] and aligned with the true labels and indices. Predictive metrics were recomputed, including independent rank-based AUROC. ResNet branches shared identical epoch-2 states and histories.

Beyond byte and metadata checks, every saved checkpoint was strictly loaded and executed on all 624 real test images in AMP and FP32. Replay probabilities matched the stored predictions exactly in this environment (maximum absolute difference zero for all twelve models and both precisions).

Evidence:

- `06_results/reproducibility_checks.json` and `.md`
- `05_experiments/run_acceptance_v1.json`
- `05_experiments/reproduced_v1/artifacts/checkpoint_replay.json`
- `04_implementation/src/pipeline.py` (`verify`, `replay`)

The Phase 3 matrix stays immutable; this separate record updates acceptance without changing the hash-pinned specification. Results were already examined before protocol adoption. No newly blinded evaluation claim is made.

The user additionally requested supervisor-reviewable executable code and complete reproduction support. Active Python settings, training, analysis, hardware probes, a command launcher, Markdown methods, CSV outputs and result explanations inside the JSON are now provided. A separate full retraining run is recorded under `05_experiments/retrained_v1`; its validation record distinguishes numerical reproduction from changing device timing.

Next formal milestone: Phase 5 evaluation/statistics acceptance. Regenerated analysis is available for that review; it does not require another default training run.