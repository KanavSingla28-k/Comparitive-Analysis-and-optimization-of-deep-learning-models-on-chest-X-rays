# Next optimization experiment — validation-only batch-size pilot

Status: proposed, **not executed**. This is separate from the completed `cxr_compare_v1` comparison. New protocol identifier: `cxr_batch32_v2_pilot`.

Use ResNet-18 with seeds 42,43,44. Both branches start from each saved `resnet18_shared_seed{seed}.pt` epoch-2 state. Compare physical batch 16 against batch 32 for three additional partial-fine-tuning epochs at learning rate 1e-4. Keep trainable modules, loss/class weight, AdamW decay, AMP, fixed BN, rotations, input size and final-checkpoint rule unchanged. Match per-seed image order and image exposure; larger physical batches necessarily change optimizer update count, which must be documented.

Evaluate validation AUROC and F1, finite losses, adaptation time including validation and peak allocated memory. Do not load test predictions/labels for model selection. Store new versioned source/settings, hashes, logs, checkpoints and validation probabilities under `05_experiments/batch32_v2_pilot/`. Do not mix v1 and v2 outputs or describe the old test split as untouched.

Engineering criteria fixed before execution:

- Complete all three seeds without OOM and with finite losses.
- Keep peak allocated memory below 70% of physical VRAM; also inspect total GPU use for practical headroom.
- Mean validation AUROC reduction no greater than 0.005, and no individual paired-seed reduction greater than 0.010.
- Median paired adaptation-time speedup at least 1.10x under comparable laptop load; repeat timing if unstable.

These are practical pilot criteria, not a statistical non-inferiority or equivalence test. If they fail, retain batch 16 and report the failed optimization. A new generalization claim needs suitable untouched external evaluation.

The existing pipeline supplies reusable model, data and training-phase functions. A dedicated v2 branch runner and validation-only export must be implemented before launching this proposal. No v2 code execution or results are claimed by completing the v1 handoff.