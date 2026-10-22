# Phase 2 — environment and pipeline readiness

**Status: COMPLETE / READY for the accepted benchmark workflow.** Verified: 22 October 2026 (IST).

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

## 1. Decision

Reuse the existing CUDA environment and archived pipeline functions. No package installation, architectural rewrite, or new comparative training run is required to establish readiness.

Fresh checks passed for all three architectures using accepted training data. The checks used disposable in-memory model copies; the original trained checkpoints, implementation source, and raw dataset were preserved.

**Next milestone:** Phase 3 — freeze the detailed comparison protocol. Later run/evaluation acceptance remains separate from this technical readiness check.

## 2. Verified runtime

| Item | Verified value |
|---|---|
| Python executable | `C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe` |
| Python | 3.13.7 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Reported VRAM | 6,140.5 MiB |
| CUDA available | Yes |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| CUDA runtime | 12.8 |
| NumPy | 2.5.4 |
| scikit-learn / SciPy | 1.9.1 / 1.18.1 |
| Matplotlib / Pillow | 3.11.2 / 13.0.0 |
| PyTorch CPU threads | 4 |

The checked package versions match the archived lock. NumPy, torchvision, and supporting analysis/plotting libraries resolve from the dependencies preserved inside this project. PyTorch resolves from the pre-existing shared CUDA environment. Supporting imports and the noninteractive Matplotlib Agg backend passed.

Use the explicit GPU Python interpreter for later execution; an unqualified system Python may select the separate CPU-only installation. No shared environment or external package installation was modified during this phase. Verification caches and outputs are under `04_implementation/verification/`.

## 3. Accepted pipeline location and path policy

Reference implementation: `archive/pilot_2026-10-22/src/experiment.py`.

Canonical raw data: `archive/pilot_2026-10-22/data/pneumoniamnist_224.npz`.

Accepted indices: `03_data/manifests/train_indices.npy`, `val_indices.npy`, and `test_indices.npy`.

The verification imported the existing functions, bound data-index reads to the accepted manifests, and redirected generated outputs/caches to the active verification folder. This tested the existing data/model/prediction behavior without rewriting the implementation or mutating the pilot's artifacts.

The archived command-line entry points retain their original archive-relative output defaults. Do not invoke mutating `prepare`, `train`, or `evaluate` commands against the archive as the default way to create a new study. If a later phase requires new runs, give them explicit versioned active paths under `05_experiments/` and accepted result paths under `06_results/`. Reusing the completed evidence does not require creating a new runner now.

The path decision is also recorded in `phase_2_runtime_paths.json`.

## 4. Data and preprocessing checks

A balanced 16-image smoke batch was selected from accepted training indices, with eight normal and eight pneumonia examples. No test inference was performed.

- Loaded training/validation indices match the accepted Phase 1 index files.
- Loaded counts match 4,698 training and 524 validation examples.
- Training class counts match 1,214 normal and 3,484 pneumonia examples.
- Model inputs have shape **16 × 3 × 224 × 224**.
- Targets are floating-point binary values of shape **16**.
- Grayscale repetition, scaling by 255, and ImageNet normalization exactly match an independent reference calculation.
- Nonaugmented preprocessing is deterministic.
- Seeded rotation augmentation is reproducible, changes the inputs, preserves labels, and produces finite values.
- Source inspection confirms training uses rotation while prediction uses nonaugmented batches.

These are pipeline checks, not new data-policy or predictive-performance experiments.

## 5. GPU forward/backward and freezing checks

One AMP/gradient-scaled optimizer update was performed per architecture/mode on disposable in-memory copies. Each mode started from the saved checkpoint state. The weighted binary loss used the Phase 1 training-only positive weight, 0.3484500574.

| Architecture | Head-only trainable parameters | Partial fine-tuning trainable parameters | Both modes |
|---|---:|---:|---|
| ResNet-18 | 513 | 8,394,241 | PASS |
| DenseNet-121 | 1,025 | 2,161,153 | PASS |
| MobileNetV3-Small | 591,873 | 942,417 | PASS |

For all six checks:

- The output shape was **16 × 1**.
- Logits, loss, and computed gradients were finite.
- Intended trainable tensors updated.
- Every frozen parameter remained unchanged.
- Batch-normalization running statistics and other registered buffers remained unchanged.
- Batch 16 fitted on the GPU without an out-of-memory error.

The saved diagnostic peak allocations include verification overhead and resident data. They are not comparable replacement measurements for the formal hardware benchmark.

## 6. Checkpoint and prediction checks

For one existing seed-42 partial-fine-tuning checkpoint per architecture:

1. Its file hash matched the saved run manifest.
2. Strict state loading into the correct architecture succeeded.
3. An in-memory serialization round trip preserved every state tensor.
4. Training-smoke predictions were identical before and after reload.
5. Saved NPZ probabilities, labels, and original source indices loaded correctly.
6. The original checkpoint file hash was unchanged after the checks.

The training source hash still matches its recorded protocol hash. The raw dataset SHA-256 still matches Phase 1. These representative checkpoint checks establish readiness; Phase 4 will separately accept the full run matrix.

## 7. Evidence files

All paths below are relative to `04_implementation/`.

| Record | Purpose |
|---|---|
| `verification/phase_2_readiness.json` | Detailed six-mode smoke checks, representative checkpoint checks, environment, and preservation checks |
| `verification/environment.json` | Current runtime, device, versions, package locations, and cache location |
| `verification/supporting_import_checks.json` | Analysis/plotting dependency imports and backend check |
| `verification/preprocessing_smoke_inputs.npz` | Original training source indices and labels used for readiness |
| `verification/*_training_smoke_predictions.npz` | Small training-batch prediction serialization checks, not comparison results |
| `phase_2_runtime_paths.json` | Accepted data/reference/active-output path policy |
| `PIPELINE_REUSE_DECISION.md` | Implementation reuse and archive-preservation decision |

## 8. Completion boundary

Six single-minibatch smoke updates were executed on disposable copies. **Zero formal comparison training runs were started.** No test split inference or new comparative metrics were generated. No implementation source was created or changed, and the original model checkpoints and dataset were not overwritten.

Phase 2 is complete. Phase 3 must now freeze the exact protocol and evidence-version decision before any later new execution.