# Phase 7 — measured optimization proof of concept

Partial ResNet fine-tuning changed mean test AUROC by **+0.019417**, with exploratory paired 95% interval **[+0.011256, +0.029072]**. Both branches start from the same epoch-2 state and use the same seeds, image order, rotation draws, additional epochs and adaptation learning rate. The trainable modules differ. This supports a within-benchmark learning improvement, not compression or clinical validation.

## Hardware evidence

| Session | Architecture | Synthetic AMP speedup | Allocation reduction | AMP batch-64 MiB |
| --- | --- | --- | --- | --- |
| reproduced_v1 | resnet18 | 2.037x | 25.2% | 1001.2 |
| reproduced_v1 | densenet121 | 2.003x | 44.4% | 4273.6 |
| reproduced_v1 | mobilenet_v3_small | 0.859x | 34.9% | 628.9 |
| retrained_v1 | resnet18 | 1.299x | 25.3% | 1001.7 |
| retrained_v1 | densenet121 | 1.028x | 44.3% | 4273.5 |
| retrained_v1 | mobilenet_v3_small | 2.959x | 34.8% | 629.4 |

Above-one speedup means AMP was faster. Ten-step technical measurements varied substantially across the two sessions, including MobileNet's direction. They do not establish a stable model-specific acceleration or an independent run-to-run confidence interval. Allocation fell in both sessions, supporting an AMP memory-efficiency direction for these workloads.

All three models completed tested full-gradient AMP batch 64. DenseNet used approximately 4.17 GiB of allocated tensors, leaving less practical headroom on this 6 GB GPU. Driver allocations, unused reservations, data residency and long-run behavior are additional. These are feasible tested workloads, not absolute maximums. Batch 16 remains the validated real-training setting.

FP32 was faster for isolated batch-1 inference in both sessions. Timing excludes image decoding, preprocessing and transfer. Timing inputs were fresh random tensors of matching shape, not identical values; numerical consistency below used the SAME real test images and checkpoints. This timing limitation remains disclosed.

## Precision consistency

| Configuration | Max probability change | Threshold changes / 1872 seed predictions | Max AUROC change |
| --- | --- | --- | --- |
| resnet18_frozen | 0.002760 | 0 | 0.000044 |
| resnet18_partial | 0.004156 | 1 | 0.000038 |
| densenet121_partial | 0.005754 | 0 | 0.000055 |
| mobilenet_v3_small_partial | 0.080860 | 8 | 0.000575 |

MobileNet's probability changes and eight changed decisions across 1872 seed predictions show why AMP cannot be called numerically identical or automatically accuracy-preserving. Do not tune thresholds on these test differences. Training precision, isolated inference precision and predictive fine-tuning are separate choices.

## Next experiment

[The batch-32 validation pilot](../05_experiments/NEXT_OPTIMIZATION_EXPERIMENT.md) is a proposed next experiment, not an executed result. There is no measured theoretical optimization ceiling or guarantee that a batch change improves real training.

Phase 7 is technically complete through the paired learning signal and disclosed memory/precision probes. Quantization, pruning, distillation and architecture search remain future work.
