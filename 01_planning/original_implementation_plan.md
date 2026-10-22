# Tonight's implementation plan

Prepared: 22 October 2026, Asia/Kolkata (IST, UTC+05:30).

## 1. Fixed project decision

**Project:** Comparative Evaluation and Fine-Tuning of Pretrained CNNs for Pediatric Pneumonia Classification on Chest X-Rays.

**Dataset:** PneumoniaMNIST+, actual 224 × 224 release. **Hardware:** one RTX 4050 Laptop GPU with approximately 6 GB VRAM. **Target:** finish the implementation, limited experiments, and written report tonight. Novelty is not a requirement of this plan.

Deliver a reproducible college project with measured results. Do not attempt a new network, multi-hospital validation, segmentation, pruning, quantization, distillation, or a web application tonight. These do not help answer the chosen question within the time available.

## 2. Dataset and access

Use the [official MedMNIST+ release](https://zenodo.org/records/10519652), file **pneumoniamnist_224.npz**. The release lists approximately **214.4 MB** and MD5 **d6a3c71de1b945ea11211b03746c1fe1**. Download only this file, not the complete collection.

The official split sizes are **4,708 / 524 / 624** for training / validation / test. The task is binary pediatric pneumonia versus normal. Verify the label mapping from the dataset metadata before treating pneumonia as the positive class. [Dataset overview](https://medmnist.com/), [dataset construction paper](https://www.nature.com/articles/s41597-022-01721-8)

Use the supplied train, validation, and test arrays. Do not concatenate and randomly re-split them. The test split is reserved for the final evaluation; do not use its performance to choose models, epochs, thresholds, or preprocessing.

Required data audit:

- Verify the downloaded file checksum and record its source and resolution.
- Inspect array keys, dimensions, counts, labels, and a contact sheet of training images.
- Count normal and pneumonia labels in each split; report actual counts rather than guessed class proportions.
- Hash raw image arrays to detect exact duplicates within and across splits. This does not detect all repeat patients or near duplicates.
- If cross-split duplicates exist, document their counts and remove overlapping development examples while preserving test examples; record revised split sizes and the deviation from the official protocol. Do not silently relabel the result as the untouched official benchmark.
- Check whether patient identifiers are available. If they are absent, explicitly state that patient separation could not be independently verified. Image hashing cannot establish patient independence.

The selected package is a standardized processed dataset. Restrict conclusions to this benchmark; it does not establish adult or cross-hospital performance. Its creators explicitly describe MedMNIST as unsuitable for clinical use. [Official dataset statement](https://medmnist.com/)

## 3. Experiment matrix

Use ImageNet-pretrained weights from one recorded torchvision release. Replace each classification head with **one output logit**. Record the exact weights identifier instead of an unversioned “latest” setting.

| ID | Architecture | Epochs 1–2 | Epochs 3–5 | Purpose |
|---|---|---|---|---|
| R0 | ResNet-18 | Train new head, freeze backbone | Continue head only | Frozen-feature control |
| R1 | ResNet-18 | Same shared checkpoint as R0 after epoch 2 | Train head and final residual stage | Fine-tuning comparison |
| D1 | DenseNet-121 | Train new head, freeze backbone | Train head and final dense block / final normalization | Historical reference architecture |
| M1 | MobileNetV3-Small | Train new head, freeze feature extractor | Train head and last two feature modules | Compact architecture |

**Core deliverable:** three architectures and four configurations. Run one predefined seed initially, for example 42. Report that this is a single-seed comparison. Do not report a standard deviation across training runs when only one run was performed.

R0 and R1 branch from exactly the same epoch-2 model, including the head. Reset the optimizer for both branches at that point, use the same phase-2 learning rate, replay the same shuffled batches and augmentation seeds, and give both branches three additional epochs. The intended difference is which weights can be updated.

For the ResNet comparison, keep backbone batch-normalization statistics fixed in both branches, including during partial fine-tuning. Apply the same frozen-BN-statistics policy to the other architectures. Document this choice. Calling `model.train()` indiscriminately can otherwise change statistics in supposedly frozen layers.

Use the fixed final epoch checkpoint for the planned comparisons. Validation curves are diagnostic, not an opportunity to add unplanned tuning runs. If a run is invalid because of a genuine code error, repair it and rerun the affected configurations consistently; record the correction.

This compares a shared recipe under a short training budget. It does not establish the maximum achievable performance of each architecture.

## 4. Common preprocessing and training recipe

| Setting | Chosen default |
|---|---|
| Input | Actual 224 × 224 dataset images |
| Channels | Repeat grayscale into three channels for ImageNet weights |
| Scaling | Convert to float in [0,1] |
| Normalization | ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| Training augmentation | Mild random rotation up to 5 degrees only |
| Validation / test | Deterministic; no random augmentation |
| Physical batch size | Start at 16; use the same value across configurations |
| Data loader | Start with 0 workers on Windows; increase only if measured useful |
| Precision | CUDA automatic mixed precision with gradient scaling |
| Optimizer | AdamW; weight decay 1e-4 |
| Epochs 1–2 | Head only; learning rate 1e-3 |
| Epochs 3–5 | Learning rate 1e-4; train the modules listed above |
| Loss | BCEWithLogitsLoss; one float target per image |
| Class imbalance | Set positive weight = training negatives / training positives; record it |
| Checkpoint | Fixed final epoch; also preserve the shared ResNet epoch-2 checkpoint |
| Initial seed | 42 across Python, NumPy, PyTorch, and data shuffling |

Do not apply sigmoid before BCEWithLogitsLoss. Apply sigmoid when generating probabilities for metrics. Do not combine a weighted sampler with positive-weighted loss tonight; that would complicate interpretation.

No vertical flips, horizontal flips, strong crops, synthetic image generation, or preprocessing search. The dataset is already spatially processed. A mild rotation is enough to demonstrate augmentation without creating a large tuning task.

When freezing the feature extractor, disable its gradients and retain evaluation behavior for frozen modules. Keep the classifier in training mode. Verify the actual trainable parameter names and counts before every phase. Do not assume a blanket training-mode call preserves the freeze policy.

If batch 16 fails the memory smoke test, choose batch 8 for **all** configurations before the formal runs. Lower batch size first rather than changing image resolution. Train one model at a time and release the previous model and optimizer from GPU memory.

Mixed precision is a resource setting shared by all runs. It is not a separate claimed compression contribution. The main optimization experiment is R0 versus R1.

## 5. Environment gate: complete this first

The hardware check succeeded, but the shell currently selects a CPU-only Python installation. Reuse the GPU environment configured earlier if it is available. Do not reinstall the global Python stack just because this shell selected the wrong interpreter.

Before downloading model weights or launching formal runs, save an environment check containing:

1. Python executable path and Python version.
2. PyTorch and torchvision versions, and a successful torchvision import.
3. `torch.cuda.is_available()` = true.
4. CUDA device name identifying the RTX 4050.
5. A successful model forward and backward pass on that device.
6. Installed-package lock or freeze file and selected pretrained-weight identifiers.

If the earlier environment cannot be located quickly, create a dedicated local environment with a compatible CUDA-enabled PyTorch / torchvision pair from the official installation guidance. Avoid ad hoc upgrades or version guessing. The success condition is the actual CUDA training smoke test, not just `nvidia-smi`.

Allocate about 20 minutes to the environment and dataset gate. If CUDA is still unavailable after that, switch the project to two frozen-backbone baselines with cached features and document the change; do not allow setup to consume the entire evening. That fallback is a different, smaller experiment and must not be presented as the full four-configuration plan.

## 6. Implementation tasks and files

Create a compact project in this workspace, with one shared implementation used by every architecture:

```text
data/                         downloaded dataset; excluded from version control
src/data.py                   load arrays, transforms, split and duplicate audit
src/models.py                 model creation, heads, freeze policies
src/train.py                  shared training loop and checkpoint branching
src/evaluate.py               predictions and classification metrics
src/benchmark.py              parameter, timing, memory, and checkpoint measures
src/plots.py                  curves, confusion matrices, comparison figures
configs/experiment.json       fixed model, data, seed, and schedule settings
artifacts/environment.txt
artifacts/dataset_audit.json
artifacts/experiment_manifest.json
artifacts/metrics.csv
artifacts/predictions/         probabilities, targets, and stable sample indices
artifacts/history/             per-epoch training and validation logs
artifacts/checkpoints/
artifacts/figures/
report.md
README.md
requirements-lock.txt
```

Do not put the dataset, environments, or large checkpoints in Git. Record how to retrieve or regenerate them. No interface or application server is required.

Suggested implementation order:

1. Build the dataset loader and audit; confirm the class mapping visually and from metadata.
2. Implement ResNet-18 and the shared loop first. Validate one minibatch before adding models.
3. Add the other model constructors and assert their output is `[batch, 1]`.
4. Implement parameter freezing, log trainable names, and verify that a backward step changes the intended weights only.
5. Implement the ResNet checkpoint branch; confirm both branches start with identical model-state hashes.
6. Add evaluation, prediction saving, and resource measurement.
7. Run the fixed configurations sequentially, then create plots and the report from saved outputs.

Keep verification focused: shapes and class mapping, finite losses, GPU execution, correct frozen weights, checkpoint reload, and reproducible metric calculation. A broad test suite is unnecessary for tonight.

## 7. Time and compute budget

There were roughly four hours until midnight when this plan was prepared. Use the relative schedule below; the example assumes implementation starts at **20:10 IST on 22 October 2026**, ending around **23:50 IST**. If starting later, apply the reduction rules rather than promising the same scope.

| Elapsed | Example IST | Work and completion gate |
|---|---|---|
| 0:00–0:20 | 20:10–20:30 | Activate GPU environment; download and verify dataset; inspect counts |
| 0:20–1:05 | 20:30–21:15 | Implement the shared loader, models, training loop, evaluation |
| 1:05–1:20 | 21:15–21:30 | Smoke-test all configurations; benchmark speed; freeze scope |
| 1:20–2:10 | 21:30–22:20 | Formal training; save logs and checkpoints |
| 2:10–2:45 | 22:20–22:55 | Final evaluation, resource measurements, and figures |
| 2:45–3:25 | 22:55–23:35 | Write report from actual outputs; assemble reproducibility files |
| 3:25–3:40 | 23:35–23:50 | Verify report numbers, paths, rerun instructions, and limitations |

These are planning allocations, not measured runtime promises. Start the report skeleton while downloads run; reserve reporting time even if experiments take longer.

**Estimate before committing.** The shared ResNet head phase plus the two branches and two other models total approximately **18 training passes** over 4,708 images, or **84,744 training-image exposures**. Validation adds **9,432 image exposures** across those 18 epoch evaluations, reusing validation results for the shared first phase. Final evaluation of four configurations adds 2,496 test-image exposures.

Measure throughput over representative warm batches, including data loading and backward computation, separately for each architecture and phase. Use `sum(phase epochs × training images / phase throughput)` plus measured validation and checkpoint overhead. The often simpler estimate `84,744 / throughput` is only illustrative: at 50 training images/s it is about 28 minutes before overhead; at 20 images/s it is about 71 minutes. Neither rate is a benchmark of this laptop.

Freeze a smaller scope **before** formal training if the estimate exceeds the training window:

1. Omit DenseNet-121 first. Keep ResNet-18, MobileNetV3-Small, and the paired ResNet ablation; revise the report to two architectures.
2. If necessary, choose one head epoch plus two phase-2 epochs for every configuration. Keep branch exposure matched.
3. If only CPU is available, use two frozen pretrained feature extractors, cache features once, and train lightweight heads. Drop the fine-tuning claim.

Do not quietly compare a completed five-epoch run with a truncated one as if budgets matched. Label interrupted runs, exclude them from the main ranking, and retain the logs.

No cloud GPU is needed. Leave the laptop plugged in; use the same power conditions for the timing comparison. If there is spare time, repeat the **paired ResNet experiment** with one additional seed before adding another model. Keep the first comparison single-seed unless every relevant configuration was repeated.

## 8. Evaluation protocol

The primary endpoint is **test AUROC for pneumonia**. Also report average precision (AP), F1, precision, recall/sensitivity, specificity, accuracy, balanced accuracy, and the confusion matrix. Identify AP explicitly; do not silently treat it as identical to trapezoidal PR-AUC.

Use threshold **0.5** for the main thresholded metrics, fixed in advance. AUROC and AP use unthresholded probabilities. If there is time for an optional threshold analysis, select the maximum-F1 threshold using validation data only, define a tie rule, freeze it, and report its test results separately from the 0.5 results. Never choose the threshold from test labels.

Include a simple majority-class baseline for accuracy and report positive prevalence as the no-skill reference for AP. A high accuracy alone is not enough evidence that the model learned a useful classifier.

For uncertainty, obtain approximate 95% intervals for AUROC and the paired R1–R0 AUROC difference using 1,000 stratified bootstrap resamples of the test examples. Use the same sampled indices for both models in the paired difference. These are conditional on the trained checkpoints and the image-level benchmark; they do not measure training-seed variability or solve unknown patient clustering. If this cannot be implemented correctly in the available time, omit the interval and state the limitation.

Avoid a “statistically better” claim from a tiny score difference. A single seed and a small, processed test set support descriptive comparisons with limited uncertainty analysis, not a general architecture ranking.

## 9. Resource measurements

Measure all models with the same batch, precision, device, input shape, and timing method. Separate training cost from inference cost.

| Quantity | Measurement definition |
|---|---|
| Total parameters | All model parameters after replacing the head |
| Trainable parameters | Separately for head-only and partial fine-tuning phases |
| Training duration | Wall-clock time including training and scheduled validation; log phases separately |
| Training memory | Reset CUDA peak statistics and record peak allocated memory during a comparable phase |
| Inference latency | Batch 1, evaluation mode, inference mode, 224 × 224 input, shared precision |
| Timing repetitions | 30 warm-up forwards and 200 timed forwards; report median and 95th percentile |
| CUDA timing | Synchronize around timed operations or use correctly synchronized CUDA events |
| Timing scope | Device-resident model forward only; label explicitly, excluding download and preprocessing |
| Model file size | Weights-only state dictionary in a common dtype, excluding optimizer and scaler |

Do not call the complete optimizer checkpoint size “model size.” Do not infer speed from parameters alone. If adding an end-to-end timing, report it as a separate measurement including preprocessing and data transfer.

Select a practical recommendation from the measured accuracy–cost trade-off. A small model within 0.01 AUROC of the strongest observed model can be highlighted as a **descriptive heuristic**, chosen in advance, if it offers measured cost savings. This is not an equivalence or noninferiority test. If the compact model loses substantially, say so.

## 10. Report and submission package

Write a short report with: abstract; revised aim; corrected related work; dataset; model and training protocol; results; fine-tuning ablation; resource comparison; limitations; conclusion; references.

Minimum figures:

1. Training/validation loss and validation AUROC across epochs.
2. Test ROC curves for completed configurations.
3. Confusion matrices at threshold 0.5.
4. Test AUROC versus measured inference latency, annotated with model names.

Minimum result table:

| Configuration | AUROC | AP | F1 | Recall | Specificity | Parameters | Training minutes | Inference ms | Weights MB |
|---|---|---|---|---|---|---|---|---|---|
| R0: ResNet frozen | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| R1: ResNet partial FT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| D1: DenseNet partial FT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| M1: MobileNet partial FT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Every number must come from saved predictions or measurement logs. Replace pending entries only when completed. Literature scores belong in the related-work section with their own task and protocol, not in the experimental comparison table.

The limitations must name: pediatric binary scope; processed 224-resolution data; no external validation; uncertain independently verified patient separation if IDs are absent; short schedule; limited hyperparameter tuning; single-seed results where applicable; and hardware-specific latency. Record differences in the original ImageNet pretraining recipes as well: using available pretrained weights makes this a comparison of pretrained model packages under our protocol, not an isolated causal test of architecture alone.

Grad-CAM is optional only after the core outputs exist. A few heatmaps are illustrations, not evidence of validated localization or absence of shortcuts. Do not spend the reporting window adding them.

## 11. Definition of finished

- Dataset provenance, resolution, label mapping, split counts, and duplicate findings are recorded.
- A common pipeline runs on the verified GPU environment, or the documented smaller fallback.
- Completed configurations have saved weights, logs, predictions, and resource measurements.
- Thresholds and selection decisions were made without test-based tuning.
- Tables and plots agree with saved outputs; no missing results are invented.
- The report states the scope actually completed and explains the observed trade-offs.
- README instructions reproduce data preparation, training, evaluation, and figures using the recorded environment.

**First next action:** activate the existing CUDA environment and verify one training step, then acquire `pneumoniamnist_224.npz`. The audit and plan are complete; implementation and experimental results remain to be produced.
