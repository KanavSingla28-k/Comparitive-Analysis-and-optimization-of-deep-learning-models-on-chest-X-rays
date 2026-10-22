# Phase 3 — frozen comparison protocol

**Status: COMPLETE.** Protocol ID: `cxr_compare_v1`. Adopted: 22 October 2026 (IST).

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

## 1. What is frozen

The detailed recipe and evidence matrix are fixed for acceptance of the existing pediatric pneumonia benchmark. The machine-readable specification is `05_experiments/configs/comparison_protocol_v1.json`; the twelve run records are listed in `run_matrix_v1.json` beside it. A SHA-256 freeze record pins these specifications and their source evidence.

This is adoption of existing, previously examined results. It is not prospective preregistration and does not turn the already viewed test set into a new blinded evaluation. Phase 4 still accepts the actual run artifacts; Phase 5 still accepts the analysis. No training, inference, or new statistics were run during this phase.

## 2. Data and run matrix

Use the accepted 224-resolution PneumoniaMNIST+ file in the dated archive and the Phase 1 retained index arrays in `03_data/manifests/`. The working counts are 4,698 / 524 / 624. Labels are 0 normal and 1 pneumonia. Preserve remaining within-split duplicates and disclose missing patient IDs. The test has 621 exact-image hash clusters.

| Configuration | Seeds | Schedule | Role |
|---|---|---|---|
| ResNet-18 partial fine-tuning | 42, 43, 44 | Two head-only epochs, three partial epochs | Core comparison |
| DenseNet-121 partial fine-tuning | 42, 43, 44 | Two head-only epochs, three partial epochs | Core comparison |
| MobileNetV3-Small partial fine-tuning | 42, 43, 44 | Two head-only epochs, three partial epochs | Core comparison |
| ResNet-18 frozen control | 42, 43, 44 | Two shared head-only epochs, three head-only continuation epochs | Optimization control |

There are nine core comparison checkpoints and three control checkpoints. The matrix specifies evidence to inspect, not a request to rerun it. The existing run IDs, architecture names, seeds, five-epoch histories, phase order, class weights, and paired initial-state hashes were checked for protocol compatibility. Complete checkpoint/prediction integrity acceptance follows in Phase 4.

## 3. Training recipe

Use recorded ImageNet-1K V1 packages and one-logit binary heads. The common recipe is:

- Actual 224-pixel grayscale images repeated into three channels, scaled by 255, normalized with ImageNet channel means/stds.
- Training-only rotations within ±5°, bilinear sampling, border padding, no flips or additional crops.
- Batch 16, two head-only epochs at learning rate 1e-3, then three adaptation epochs at 1e-4.
- AdamW with weight decay 1e-4 and the recorded library defaults for remaining arguments.
- BCEWithLogitsLoss with training-only positive weight 1,214 / 3,484; the reference stores it in float32.
- CUDA FP16 autocast with FP32 parameters and GradScaler initial scale 1024.
- Fixed batch-normalization running statistics. BN affine parameters inside unfrozen modules may update.
- Fixed final checkpoint. No early stopping, validation-best checkpoint selection, per-architecture hyperparameter search, or test-selected threshold.

ResNet unfreezes `layer4`; DenseNet unfreezes the final dense block and final normalization; MobileNet unfreezes feature modules 11 and 12. The classifier remains trainable. The reference puts the model in evaluation mode and the classifier in training mode, with gradients enabled only for the declared modules.

For the ResNet control, both branches use the identical epoch-2 state, fresh adaptation optimizers, the same learning rate, three additional epochs, and the same batch/augmentation schedule. The control is a matched learning experiment, not a smaller deployed architecture.

## 4. Evaluation and statistical definitions

Prediction uses evaluation/inference mode, no augmentation, batch 64 with a smaller final batch, and AMP for primary results. FP32 predictions on the same real images are a separate numerical-agreement check.

Primary endpoint: pneumonia AUROC. Secondary endpoints: average precision, F1, precision, recall, specificity, accuracy, and balanced accuracy. Thresholded predictions use probability **greater than or equal to 0.5**. AP is not relabeled as trapezoidal PR-AUC.

Average per-seed metric values over seeds 42, 43, and 44; show sample standard deviation across seeds. Do not average probabilities into an unreported ensemble. Use seed 42 for the representative curves and confusion matrices, rather than selecting the best-looking seed.

Use the recorded 2,000-replicate hierarchical bootstrap with RNG seed 20261022: resample the three training seeds and the 621 exact-image test clusters with replacement. Case resampling is not stratified. Use the same sampled cases and seed indices for all compared models. Report percentile 95% intervals for mean AUROC and these declared differences:

1. ResNet partial minus ResNet frozen.
2. ResNet partial minus DenseNet partial.
3. MobileNet partial minus DenseNet partial.

The intervals are exploratory, with no multiplicity adjustment and no adjustment for unknown patient clusters. Close scores or intervals spanning zero do not establish equivalence. Other comparisons may be described as point estimates; adding new inferential tests requires explicit analysis versioning or a post hoc label.

## 5. Resource-measurement definitions

Use the recorded RTX 4050 setup, four PyTorch CPU threads, TF32 off, cuDNN deterministic mode, and cuDNN benchmarking disabled.

**Training:** wall time includes training and scheduled validation, excluding setup/downloads. Count the common ResNet head phase once when calculating actual total compute, while reporting each configuration's full logical training cost separately. Peak CUDA allocation includes the live objects in that phase; it is not reserved memory or total driver/system GPU usage. Weights-file size excludes optimizer/scaler state.

**Batch-1 latency:** use seed-42 checkpoints, three FP32/AMP-interleaved rounds, 30 warm-up forwards and 200 timed forwards per case, CUDA events and synchronization, and device-resident 1 × 3 × 224 × 224 inputs. Report the median of round medians and the median paired-round speedup ratio plus its range. Timing excludes decoding, preprocessing, transfer, and application overhead.

**Important wording correction:** the reference creates a fresh random input for each precision/round case. The latency comparison therefore uses the same shape and model weights, not identical input values. This differs from the FP32/AMP numerical check, which uses the same real images. Earlier broad “same inputs” wording must be corrected in the accepted final report; preserve the historical archive.

**Batched inference:** AMP batches 16, 32, 64, and 128; five warm-ups and 30 timed forwards per case. Throughput uses synchronized wall time.

**Synthetic full-gradient steps:** batches 16 FP32/AMP and 32/64 AMP, three warm-ups and ten measured steps, all parameter gradients enabled, fixed BN statistics, alternating synthetic labels, unweighted binary loss, and AdamW at 1e-4 with remaining library defaults. Updated copies are discarded. This probe has a different loss/input context from predictive training and cannot establish equal accuracy or a guaranteed end-to-end acceleration.

For the capacity probes, stop the current architecture after an OOM or a completed case above 70% of physical VRAM in peak allocation; honor the listed maximum tested batch. A successful probe establishes a tested fit, not an absolute capacity ceiling. Timing repeats are technical repeats within a laptop session, not independent hardware experiments.

## 6. Compatibility review

- The archived experiment configuration matches its original training-lock configuration.
- Phase 1 dataset checks and Phase 2 readiness passed.
- All twelve metadata records have the expected architecture, seed, epoch count, and head/adaptation phase order.
- Recorded training weights match the accepted training class ratio within float32 tolerance.
- Each ResNet control/partial pair records the same epoch-2 initialization hash.
- The training and analysis reference hashes remain pinned by their previous records and the new freeze record.
- One documentation clarification is carried forward: random timing inputs are shape-matched, not value-matched. No model or numerical artifact was changed.

This compatibility review does not replace Phase 4 checkpoint/prediction acceptance or Phase 5 numerical verification.

## 7. Paths and change control

Preserve the archive. Active specifications are under `05_experiments/configs/`. Later acceptance records should be new files, rather than edits to the frozen matrix. New experiments, only if justified, must use a new versioned directory under `05_experiments/`; accepted results go under `06_results/` and the report under `07_report/`.

The archived CLI retains archive-relative write defaults. Explicitly bind active data/index/output/cache paths before any future mutating invocation. No new runner is needed for the current evidence-adoption route.

A change to data/splits, architecture/weights, seeds, training recipe, checkpoint selection, threshold, precision, timing, or statistical method requires a recorded new version. Record any concrete integrity or execution failure; rerun affected comparisons consistently, not selected seeds chosen by score. No new results overwrite the accepted record.

The frozen specifications, source hashes, and data-index hashes identify version 1. Preserve them and create a superseding version if a correction is later required.

## 8. Exit checklist

- [x] Dataset, model list, run IDs, seeds, and training schedule fixed.
- [x] Prediction precision, threshold, metrics, and checkpoint rule fixed.
- [x] Seed aggregation, bootstrap, and declared comparisons fixed.
- [x] Timing scopes, memory definitions, and probe stop rules fixed.
- [x] Test-exposure history and reporting limitations explicit.
- [x] Archive-preserving output paths and change rules recorded.
- [x] Machine-readable protocol, run matrix, compatibility record, and SHA-256 freeze written.

**Next milestone:** Phase 4 — accept the nine core comparison runs. Existing eligible artifacts are the starting evidence; no fresh training is required by this protocol freeze alone.