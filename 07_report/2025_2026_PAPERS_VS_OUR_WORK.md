# Comparative Analysis of 2025–2026 Chest X-ray Deep Learning Papers and Our Work

**Prepared:** 22 October 2026.  
**Project:** Comparative Analysis and Optimization of Deep Learning Models on Chest X-rays.  
**Local experiment:** `cxr_compare_v1`; primary measurements from `retrained_v1`.

## Executive finding

Our project is a reproducible, hardware-constrained comparison of pretrained CNNs for **pediatric pneumonia versus normal**, with an initial optimization experiment. Recent prominent research tackles broader problems: multi-disease foundation models, efficient representation learning and auditable predictions. Our AUROC is numerically high, but it does **not** establish superiority over those models on their different tasks.

The strongest supported local improvement is the paired ResNet-18 comparison: **AUROC 0.96950 → 0.98891**, an increase of **1.9417 AUROC percentage points**, with an exploratory paired 95% interval for the increase of **1.1256 to 2.9072 points**. Our highest fixed-threshold mean F1 belongs to MobileNetV3-Small, not the model with the highest mean AUROC. [Local comparison](../06_results/retrained_v1/comparison_summary.csv), [paired evidence](../06_results/retrained_v1/paired_differences.csv).

This report presents each selected method, its baseline and improvement, its relationship to our experiment, and an explanation of the numbers in the same document. No literature models were newly trained or tested by us for this report.

## 1. Selection and comparison rules

The publication window is **1 January 2025 through 22 October 2026**. The selection prioritizes recognized publication venues and primary-source result tables. “Prominent” here refers to publication venue and relevance, **not a verified citation-count ranking**. Recent 2026 papers cannot automatically be described as long-established classics. This is a curated comparison, not an exhaustive systematic review.

| Selected study | Verified publication | Why included | Relationship to our experiment |
|---|---|---|---|
| Ark+ | Nature; published online 11 June 2025 | High-profile, open chest-radiograph foundation model | Broader multi-dataset, multi-disease representation learning |
| CheXWorld | CVPR 2025; April 2025 preprint | Major computer-vision conference; anatomical world-model pretraining | Different tasks and data, but informative representation-learning baselines |
| EVA-X | npj Digital Medicine; 17 November 2025 | Medical foundation-model and efficiency study | Relevant comparison of CNN/transformer size and learned representations |
| CLEAR | Nature Biomedical Engineering; 22 July 2026 | Auditable concept-based foundation model | Adds interpretability and external evaluation that our study does not establish |
| TinyCNN knowledge-distillation benchmark | BMC Research Notes; 11 May 2026 | Closest task/benchmark match; included for relevance, not claimed to be equally famous | PneumoniaMNIST, ResNet-18 teacher and lightweight students; still a different protocol |

Sources: [Ark+](https://www.nature.com/articles/s41586-025-09079-8), [CVPR record](https://cvpr.thecvf.com/virtual/2025/poster/34038), [EVA-X](https://www.nature.com/articles/s41746-025-02032-z), [CLEAR](https://www.nature.com/articles/s41551-026-01741-4), [distillation study](https://link.springer.com/article/10.1186/s13104-026-07793-8).

Older model names in their baseline tables are intentional: the **selected papers** are from 2025–2026; a new paper can compare itself against an older baseline.

### 1.1 Three kinds of numerical comparison

1. **Within-paper benchmark comparison:** a proposed method versus a competing reference on the paper's evaluation. Useful, but architecture, pretraining and resources may differ.
2. **Within-study intervention or ablation:** change a component or training strategy within that study. Our paired ResNet branches are an example.
3. **Across-paper juxtaposition:** display scores from separate studies. Without matching data, splits, labels, preprocessing and evaluation, the difference is descriptive, not evidence of an improvement caused by our method.

A common dataset name alone does not establish an identical evaluation. Binary pneumonia AUROC, mean AUROC over 14 adult findings, zero-shot classification and supervised fine-tuning are different experiments.

## 2. Analysis of the recent models and their baseline improvements

### 2.1 Ark+: learning from heterogeneous expert labels

**Method.** Ark+ accrues knowledge cyclically from differently labelled datasets, using a Swin-transformer-based encoder and reusable task components. It supports linear probing and fine-tuning. The Nature study covers broader disease recognition and transfer than our single pediatric task. [Paper and framework](https://www.nature.com/articles/s41586-025-09079-8), [evaluation caption](https://www.nature.com/articles/s41586-025-09079-8/figures/2).

**Verified results.** The authors' supplementary data give ChestX-ray14 mean AUC **81.41 → 82.32** for CXR-FM versus Ark+ under linear probing (**+0.91 points**), and **83.54 → 84.43** for RAD-DINO versus Ark+ under fine-tuning (**+0.89 points**). Within Ark+, linear probing to fine-tuning changes **82.32 → 84.43**, or **+2.11 points**. These are separate contrasts, not one shared causal comparison. [Published supplementary data](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09079-8/MediaObjects/41586_2025_9079_MOESM4_ESM.xlsx).

**Meaning for us.** Our partial-fine-tuning result is conceptually related to adaptation depth, but it uses a different model, binary task and dataset. Neither result proves that deeper adaptation always improves external generalization. Our 0.98891 must not be ranked against Ark+'s 0.8443 across different endpoints.

### 2.2 CheXWorld: learning anatomy and domain variation

**Method.** A ViT-Base learns local anatomical structure, global layout and image-domain variation through self-supervised prediction. Its scale is approximately 0.5 million pretraining X-rays. [Methods and results](https://arxiv.org/html/2504.13820#S5.T1).

**Verified results.** On VinDr-CXR, Table 1 reports MAE **92.76 → CheXWorld 95.24 AUROC**, or **+2.48 points**. Table 3's 1%-label ablation changes local-only **84.71 → full unified model 90.53**, or **+5.82 points**. The ablation and full-data benchmark are different experiments. RSNA classification values in the paper are **accuracy**, not AUROC. [Table 1](https://arxiv.org/html/2504.13820#S5.T1), [Table 3](https://arxiv.org/html/2504.13820#S5.T3).

**Meaning for us.** This studies the representation learned before adaptation; our experiment reuses ImageNet weights. Its reported pretraining uses eight RTX 4090 GPUs for 16 hours. That cost is not comparable to our short downstream training time, which excludes ImageNet pretraining. Reusing a released encoder is a more plausible extension than reproducing foundation pretraining on our 4050. [Implementation details](https://arxiv.org/html/2504.13820#S5).
### 2.3 EVA-X: medical self-supervision across model sizes

**Method.** EVA-X combines semantic and geometric self-supervision in a vision-transformer family. Reported sizes are approximately 6M, 22M and 86M parameters. [Model description](https://www.nature.com/articles/s41746-025-02032-z#Sec2).

**Verified results.** In the ChestX-ray14 comparisons, XProtoNet with a DenseNet-121 backbone has **82.2 mean AUC**, while EVA-X-Ti has **82.4**: **+0.20 points**. Medical MAE's ViT-S has **82.3**, versus EVA-X-S at **83.3**: **+1.00 point**. The paper also reports EVA-X-B at **83.5 mAUC**. The tiny-model baseline is an XProtoNet system, not our plain DenseNet checkpoint. [Classification results](https://www.nature.com/articles/s41746-025-02032-z).

**Meaning for us.** Representation learning and pretraining can matter as much as depth. The 6M-versus-8M comparison in that paper cannot establish that its model is faster on our GPU. Our task-specific parameter counts also differ from standard backbones with larger classification heads. Parameter efficiency, predictive performance and measured latency need separate evidence.

### 2.4 CLEAR: auditable concept-based prediction

**Method.** CLEAR maps images into a clinical concept space using image/report pretraining and language embeddings. Its novelty includes inspecting and intervening on predictive concepts, rather than simply increasing a classifier score. [Method](https://www.nature.com/articles/s41551-026-01741-4).

**Verified results.** For zero-shot VinDr-CXR classification, CheXzero **75.0 → CLEAR 78.2 mean AUROC**, a **+3.20-point** difference. On PadChest, **66.8 → 70.0** is also **+3.20 points**. A distinct enlarged-cardiomediastinum intervention changes supervised AUROC **0.727 → 0.784**: **+5.70 points**, or approximately **7.84% relative**. These are different tasks and evaluation regimes. [Results and Fig. 3 intervention](https://www.nature.com/articles/s41551-026-01741-4#Sec2).

**Meaning for us.** We have not demonstrated comparable concept-level auditing or external validation. Our higher binary AUROC does not make our model better than CLEAR. Its main methodological lesson is to examine failure mechanisms; our first step should be documented false-positive/false-negative review, without calling an explanatory heatmap clinical validation.

### 2.5 The 2026 PneumoniaMNIST distillation study: compression can lose discrimination

**Method.** Neo compares a ResNet-18 teacher with a 60,642-parameter TinyCNN, vanilla training, response distillation and FitNets. It uses five seeds and reports discrimination, calibration and resource measurements. [Methods](https://link.springer.com/article/10.1186/s13104-026-07793-8).

**Verified results.** Table 1 reports AUROC **0.8818 → 0.8760** for vanilla versus response-KD (**−0.58 points**), and **0.8818 → 0.8820** for FitNets (**+0.02 points**). The paper finds no consistent distillation benefit. Its teacher scores **0.9735**; vanilla TinyCNN therefore loses **9.17 AUROC points**, while Table 2 gives approximately **184.3× fewer parameters** and **3.73× faster batch-1 CPU inference**. Compression succeeds as a size/runtime trade-off, not as preserved discrimination. [Tables 1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13330209/#Tab1).

**Meaning for us.** This is the closest benchmark match, but not a controlled comparison against our run. Different seeds, schedules, losses, augmentation and source-file/preprocessing details prevent attributing the score gap to one technique. Paper timings on its CPU/T4 setup cannot be directly compared with our RTX 4050 latency.

## 3. Baselines versus improvements: numerical comparison

**Units:** AUROC is expressed on a 0–100 scale below. A change of 1.00 point equals an AUROC change of 0.0100. Each row is a contrast within its own study; rows do not form one leaderboard.

| Study and experiment | Baseline/reference | Improved or alternative configuration | Baseline → alternative AUROC / mean AUROC | Absolute change, points | What the contrast establishes |
|---|---|---|---|---|---|
| Ark+, ChestX-ray14 linear probe | CXR-FM | Ark+ | 81.41 → 82.32 | +0.91 | Reported benchmark comparison; different pretrained systems |
| Ark+, ChestX-ray14 fine-tuning | RAD-DINO | Ark+ | 83.54 → 84.43 | +0.89 | Same evaluation regime, different pretrained systems |
| Ark+ adaptation comparison | Ark+ linear probe | Ark+ fine-tuning | 82.32 → 84.43 | +2.11 | Adaptation-strategy contrast within the paper |
| CheXWorld, VinDr-CXR | MAE | CheXWorld | 92.76 → 95.24 | +2.48 | Representation-learning benchmark comparison |
| CheXWorld, VinDr-CXR, 1% labels | Local-only pretraining | Full unified pretraining | 84.71 → 90.53 | +5.82 | Component ablation; not the full-data experiment |
| EVA-X, ChestX-ray14 | XProtoNet / DenseNet-121 | EVA-X-Ti | 82.20 → 82.40 | +0.20 | Competing methods with different architectures |
| EVA-X, ChestX-ray14 | Medical MAE / ViT-S | EVA-X-S | 82.30 → 83.30 | +1.00 | Competing representation-learning methods |
| CLEAR, VinDr-CXR zero-shot | CheXzero | CLEAR | 75.00 → 78.20 | +3.20 | Mean across multiple findings, without target-task supervised adaptation |
| CLEAR, concept intervention | Before intervention | After concept filtering/refit | 72.70 → 78.40 | +5.70 | One particular finding; not its overall mean score |
| Neo, PneumoniaMNIST student | Vanilla TinyCNN | Response-KD TinyCNN | 88.18 → 87.60 | −0.58 | No observed discrimination improvement |
| Neo, PneumoniaMNIST student | Vanilla TinyCNN | FitNets TinyCNN | 88.18 → 88.20 | +0.02 | Tiny point difference; no consistent gain established |
| **Our paired ResNet-18** | Head-only continuation | Partial fine-tuning | **96.9496 → 98.8913** | **+1.9417** | Matched epoch-2 start and seed schedule; positive exploratory paired interval |

Table sources: [Ark+ supplementary data](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09079-8/MediaObjects/41586_2025_9079_MOESM4_ESM.xlsx), [CheXWorld Tables 1 and 3](https://arxiv.org/html/2504.13820#S5.T1), [EVA-X results](https://www.nature.com/articles/s41746-025-02032-z), [CLEAR results](https://www.nature.com/articles/s41551-026-01741-4), [Neo Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13330209/#Tab1), [our paired statistics](../06_results/retrained_v1/paired_differences.csv).

**How to read this table.** A smaller change in a difficult multi-label benchmark may be more important than a larger change on an easier binary benchmark. Absolute change alone does not rank the quality of research contributions. We computed arithmetic differences from the displayed paper values; these differences are not newly estimated confidence intervals or significance tests. Some literature baseline entries are references to prior methods rather than fully matched retraining ablations.
## 4. Analysis of each model in our project

### 4.1 Shared local protocol

All three architectures use official ImageNet-1K V1 pretrained weights, actual **224 × 224 PneumoniaMNIST+** images and the same two head-training plus three partial-fine-tuning epochs. Batch size is 16, optimizer AdamW, training loss class-weighted BCE, and training uses AMP with fixed batch-normalization running statistics. Seeds are 42, 43 and 44. Threshold metrics use 0.5; final checkpoints are selected by the fixed schedule, not test performance. [Executable settings](../04_implementation/src/settings.py), [training and inference](../04_implementation/src/experiment.py).

Exact duplicate cleanup removes ten training rows that overlap validation. Accepted sizes are **4,698 training / 524 validation / 624 test**. The test set has 234 normal and 390 pneumonia cases. Patient identifiers are unavailable. Scores had already been examined before formal protocol adoption, so this is not a newly blinded confirmatory study. [Dataset acceptance](../03_data/PHASE_1_DATA_ACCEPTANCE.md), [adopted protocol](../01_planning/PHASE_3_PROTOCOL.md).

### 4.2 Predictive comparison

AUROC and F1 below are on the **0–1 scale**. “±” is sample standard deviation across three seeds. The AUROC interval is an exploratory paired seed-and-image-cluster bootstrap interval, not that SD multiplied by a normal critical value.

| Configuration | AUROC mean ± seed SD | Exploratory 95% AUROC interval | AP | F1 mean ± seed SD | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| ResNet-18, head-only control | 0.96950 ± 0.00088 | 0.95606–0.98064 | 0.97957 | 0.91309 ± 0.00290 | 0.97863 | 0.72507 | 0.88355 |
| ResNet-18, partial fine-tuning | **0.98891 ± 0.00210** | 0.98143–0.99504 | **0.99214** | 0.93352 ± 0.02864 | 0.98889 | 0.78063 | 0.91079 |
| DenseNet-121, partial fine-tuning | 0.98552 ± 0.00118 | 0.97540–0.99310 | 0.98948 | 0.93884 ± 0.01181 | **0.99487** | 0.79202 | 0.91880 |
| MobileNetV3-Small, partial fine-tuning | 0.98225 ± 0.00076 | 0.97268–0.99006 | 0.98829 | **0.94289 ± 0.00941** | 0.98632 | **0.82336** | **0.92521** |

Source: [accepted numerical table](../06_results/tables/accepted_comparison.csv). These are averages of separate model scores, not a probability ensemble. Local changes are calculated from unrounded stored means; displayed values are rounded for readability. Best point estimates do not alone establish statistical superiority.

### 4.3 ResNet-18: strongest observed ranking performance and a paired improvement

ResNet-18 has **11,177,025 parameters** in our one-logit implementation. We train its classifier first, then unfreeze `layer4` and the classifier. Its mean AUROC is the highest observed in our primary comparison. Its fixed-threshold F1 is lower than MobileNet's, and its F1 seed SD is relatively large: approximately **2.864 percentage points**. High ranking performance therefore coexists with operating-point variability.

The frozen and partial branches start from the identical epoch-2 weights and use matching additional epochs, learning rate, seed schedule, ordering and rotations. This is our most useful controlled optimization evidence. It does not reduce the deployed network's parameter count.

### 4.4 DenseNet-121: a strong comparator with the highest sensitivity

DenseNet-121 has **6,954,881 parameters** after replacing its ImageNet classifier with one logit. We adapt its final dense block, final normalization and classifier. It achieves the highest mean sensitivity, **99.487%**, but lower specificity than MobileNet: more normal cases are classified as pneumonia at the chosen threshold.

Its greater depth does not imply more parameters than our ResNet. Its observed AUROC is 0.003397 below ResNet's, but the declared paired interval crosses zero. This experiment does not establish that DenseNet is inferior. We did not run a matched five-epoch head-only DenseNet control, so we cannot claim a separately measured fine-tuning improvement for this architecture.

### 4.5 MobileNetV3-Small: the best observed fixed-threshold balance and smallest model

MobileNetV3-Small has **1,518,881 parameters**. We train its classifier and later its last two feature modules. It uses **86.41% fewer parameters** than our ResNet and has the highest mean F1, specificity and accuracy at 0.5. Relative to ResNet, its mean F1 is **0.9374 points higher**, while mean AUROC is **0.6666 points lower**. These are descriptive local contrasts, not a declared significance test between those two models.

Its smaller size did not produce the lowest measured GPU batch-1 latency. Network operations and execution overhead matter as well as parameter count. As with DenseNet, a matched head-only MobileNet ablation is absent; its own fine-tuning gain has not been independently measured.

### 4.6 Our actual baseline-to-improvement table

| Metric | ResNet head-only continuation | ResNet partial fine-tuning | Absolute change, percentage points |
|---|---|---|---|
| AUROC | 0.96950 | 0.98891 | **+1.9417** |
| Average precision | 0.97957 | 0.99214 | +1.2571 |
| F1 | 0.91309 | 0.93352 | +2.0428 |
| Precision | 0.85581 | 0.88601 | +3.0196 |
| Sensitivity | 0.97863 | 0.98889 | +1.0256 |
| Specificity | 0.72507 | 0.78063 | +5.5556 |
| Accuracy | 0.88355 | 0.91079 | +2.7244 |
| Balanced accuracy | 0.85185 | 0.88476 | +3.2906 |

Source and calculations: subtract paired configuration means in [the local comparison](../06_results/retrained_v1/comparison_summary.csv), then multiply by 100. The paired interval was computed for **AUROC**, not independently for every metric in this table. We do not claim statistical significance for each secondary improvement.

A majority-label reference that always predicts pneumonia would have **62.5% accuracy** on this test set, with constant-score AUROC 0.5. This is an analytical reference, not another trained model. It illustrates why accuracy alone is insufficient on an imbalanced dataset.

## 5. Previous papers versus our work

| Dimension | Selected recent foundation-model studies | Closest 2026 distillation study | Our work |
|---|---|---|---|
| Main aim | Broader representations, transfer, efficiency or auditable reasoning | Compact student and calibration/efficiency trade-offs | Reproducible three-CNN comparison under a short budget |
| Target task | Typically multiple adult findings; some zero-shot settings | Binary pediatric pneumonia | Binary pediatric pneumonia |
| Learning source | Large chest-image collections; some use reports or heterogeneous labels | PneumoniaMNIST teacher/student training | ImageNet pretraining, then local adaptation |
| Model family | Swin/ViT or concept-based representations | ResNet-18 teacher and TinyCNN | ResNet-18, DenseNet-121, MobileNetV3-Small |
| Primary comparison | Paper-specific baselines and task definitions | Five seeds, teacher 8 epochs, students 12 epochs | Three seeds, 5 epochs, paired ResNet control |
| Generalization evidence | Broader datasets and/or external tasks, depending on study | External probes and calibration measurements | No independent external validation or calibration study |
| Main strength relative to our project | Breadth, representation learning or auditability | Explicit compression/calibration assessment | Inspectable code, matched budget, paired learning evidence and measured local hardware |
| What ours does not replace | A foundation-model validation study | A controlled distillation/calibration replication | Those experiments remain outside our completed scope |

Sources: [Ark+](https://www.nature.com/articles/s41586-025-09079-8), [CheXWorld](https://arxiv.org/html/2504.13820), [EVA-X](https://www.nature.com/articles/s41746-025-02032-z), [CLEAR](https://www.nature.com/articles/s41551-026-01741-4), [distillation methods](https://link.springer.com/article/10.1186/s13104-026-07793-8), [our protocol](../04_implementation/PIPELINE.md).

### 5.1 Closest numerical juxtaposition: the PneumoniaMNIST teacher

| Configuration | Reported mean AUROC | Comparison status |
|---|---|---|
| Neo's ResNet-18 teacher | 0.97350 | Published study: five seeds, different training/preprocessing details |
| Our head-only ResNet-18 | 0.96950 | Local control under our accepted recipe |
| Our partially fine-tuned ResNet-18 | 0.98891 | Local adapted model under our accepted recipe |

Our partial model's point estimate is **1.5413 AUROC percentage points above** the published teacher. This is **not** a demonstrated 1.5413-point methodological improvement over that paper. The study describes MedMNIST v2 images resized to 224 × 224; its methods do not establish that the exact native 224-pixel NPZ and our duplicate-cleaned membership were used. Seeds, epochs, optimizer, augmentation and uncertainty definitions also differ. [Published methods](https://link.springer.com/article/10.1186/s13104-026-07793-8), [our data acceptance](../03_data/PHASE_1_DATA_ACCEPTANCE.md).

A fair direct comparison would require their trained protocol and our protocol to be executed on the same verified input arrays, accepted splits and evaluation definition, with uncertainty computed for the new paired experiment. Until then, the defensible local improvement remains the **paired 1.9417-point ResNet contrast**, not an external state-of-the-art claim.
## 6. Explaining the numbers

### 6.1 AUROC is ranking ability, not percent correct

AUROC measures how well positive cases rank above negative cases across thresholds, with ties receiving half-credit. An AUROC of **0.98891** corresponds to roughly **98.891% correctly ordered positive–negative pairs** for the empirical evaluation, averaged across our trained seeds. It does not mean 98.891% of individual images were classified correctly. Our ResNet accuracy at threshold 0.5 is **91.079%**.

Mean AUROC over many disease labels is another endpoint: it averages per-label ranking performance. A mean adult multi-finding AUROC of 0.84 cannot be subtracted from our pediatric pneumonia AUROC to establish model superiority.

### 6.2 Why AUROC and F1 choose different local models

ResNet has the highest mean AUROC, while MobileNet has the highest mean F1 at threshold 0.5. These observations can coexist because AUROC assesses ranking over thresholds, whereas F1 uses one operating point.

For a concrete **single-seed** example, our seed-42 ResNet partial model produced **TN=215, FP=19, FN=11, TP=379**. Its accuracy is `(215 + 379) / 624 = 95.192%`; its F1 is `2 × 379 / (2 × 379 + 19 + 11) = 96.193%`. Those are not the three-seed means. Different seed operating points explain why the mean F1 is lower, even while ranking remains strong. [Individual results](../06_results/retrained_v1/metrics.csv).

F1 is calculated separately for each seed and then averaged. Recomputing F1 from the averaged precision and recall will not necessarily reproduce the mean F1, because the operation is nonlinear.

### 6.3 Sensitivity and specificity represent different errors

- **Sensitivity / recall = TP / (TP + FN):** fraction of pneumonia examples detected.
- **Specificity = TN / (TN + FP):** fraction of normal examples correctly rejected.
- **Precision = TP / (TP + FP):** fraction of pneumonia predictions that are labelled pneumonia.

DenseNet's mean sensitivity of **99.487%** means few pneumonia cases are missed at our threshold. Its specificity of **79.202%** means the average normal-case false-positive rate is about **20.798%**. High sensitivity alone therefore does not mean a near-perfect classifier. These operating characteristics apply to this benchmark and threshold, not an untested hospital population.

### 6.4 Average precision, class balance and calibration

Our AP is average precision from the precision–recall curve. Its reference for an uninformative score is related to positive prevalence; our test prevalence is **390 / 624 = 0.625**. AP or precision values from datasets with different prevalence are not directly interchangeable. A paper's “AUPRC” may also use a different numerical definition from average precision; check the implementation before pooling values.

High AUROC or AP does not establish calibrated probabilities. A model can rank images well while its probability values are overconfident. Our class-weighted training loss gives another reason to measure calibration rather than assume it. The Neo study evaluates calibration; our completed pipeline does not yet establish comparable calibration performance. [Calibration study](https://link.springer.com/article/10.1186/s13104-026-07793-8).

### 6.5 Percentage points versus relative percentage improvement

For a metric on the 0–1 scale:

```text
Absolute change in percentage points = 100 × (new − baseline)
Relative percentage change          = 100 × (new − baseline) / baseline
```

Our AUROC change is `0.98891263 − 0.96949558 = 0.01941705`. That is **+1.9417 points**, or about **+2.0028% relative**. It is not a 1.9417% increase in classification accuracy.

CLEAR's intervention illustrates the same distinction: `0.784 − 0.727 = 0.057`, or **+5.70 points**. Dividing by 0.727 gives **7.84% relative**, consistent with the paper's rounded 7.8% relative description. [Intervention results](https://www.nature.com/articles/s41551-026-01741-4#Fig3).

### 6.6 Standard deviation, intervals and significance

Our “± seed SD” summarizes variability across **three trained seeds**. Our 95% intervals use **2,000 paired bootstrap draws** over training seeds and **621 exact-image test clusters**. These groups account for exact duplicate images, not unknown patient dependence. Three seeds remain a limited sample, and the intervals are exploratory with no multiplicity adjustment. [Statistical review](../06_results/PHASE_5_STATISTICAL_REVIEW.md).

| Declared local comparison | Mean AUROC difference, points | Exploratory paired 95% interval, points | Interpretation |
|---|---|---|---|
| ResNet partial minus ResNet frozen | +1.9417 | +1.1256 to +2.9072 | Positive within-benchmark signal under the paired recipe |
| ResNet partial minus DenseNet partial | +0.3397 | −0.2267 to +0.9940 | Direction is not established by this interval |
| MobileNet partial minus DenseNet partial | −0.3269 | −0.8991 to +0.2069 | Direction is not established by this interval |

Source: [paired statistics](../06_results/retrained_v1/paired_differences.csv). ResNet versus MobileNet was not one of these prespecified interval contrasts. The declared comparisons do not establish a complete ordering of all three architectures. An interval containing zero is also not proof of equivalence.

The Neo paper's intervals use a Student-t interval for the mean across five seeds, whereas ours combine seed and exact-image-cluster resampling. Their widths must not be interpreted as the same type of uncertainty. Across independent studies, interval overlap or non-overlap does not repair an unmatched evaluation protocol. [Published uncertainty method](https://link.springer.com/article/10.1186/s13104-026-07793-8).

## 7. Resource costs and optimization: what is actually demonstrated

| Our adapted model | Total parameters | Saved weights, MiB | Mean five-epoch training minutes | Batch-1 FP32 forward ms |
|---|---|---|---|---|
| ResNet-18 | 11,177,025 | 42.71 | 0.323 | 1.524 |
| DenseNet-121 | 6,954,881 | 27.09 | 0.695 | 10.713 |
| MobileNetV3-Small | 1,518,881 | 5.92 | 0.270 | 15.970 |

Sources: [model/resource table](../06_results/retrained_v1/comparison_summary.csv), [fresh-session latency](../06_results/retrained_v1/hardware_summary.csv).

These are one laptop's measured workloads. Training duration includes scheduled validation, but excludes setup, weight download and the original ImageNet pretraining. The comparison is not equivalent to training a medical foundation model from scratch. Saved checkpoint size is not peak GPU memory. Latency excludes image decoding, preprocessing and transfer. Smaller parameter count did not guarantee lower measured latency.

At synthetic full-gradient batch 16, AMP reduced peak tensor allocation by approximately **25.3% for ResNet, 44.3% for DenseNet and 34.8% for MobileNet** in the fresh session. However, AMP step-speed ratios changed substantially between sessions; MobileNet even changed from slower to faster. A **speedup below 1× means a slowdown**, and no stable end-to-end training speedup has been established. FP32 was faster for isolated batch-1 inference in both measured sessions. [Both-session evidence](../06_results/tables/hardware_sessions.csv).

All three full-gradient AMP batch-64 probes completed; DenseNet used approximately **4.17 GiB of allocated tensors**. This is a tested fit, not the GPU's absolute maximum or a guaranteed long-running training configuration. The probes use disposable model copies, synthetic inputs and short step sequences; they do not demonstrate preserved predictive performance at a larger training batch. [Optimization evidence](../06_results/PHASE_7_OPTIMIZATION.md).

The Neo study's TinyCNN provides a distinct compression trade-off: much smaller and faster within its own measurement setup, but a substantial loss in teacher AUROC. Our partial fine-tuning improves predictive ranking without compressing the architecture. These are different meanings of “optimization,” and both the gain and the cost must be reported.

## 8. What the literature suggests we should do next

1. **Retain the completed CNN comparison as the college project's core.** Describe it as a reproducible benchmark and learning-efficiency study, not a new foundation model or external state-of-the-art result.
2. **Measure calibration and review errors before adding complexity.** Use validation data to develop temperature scaling or threshold choices. Keep previous test exposure disclosed; do not tune on the test differences reported here.
3. **Try a released medical encoder through a small downstream experiment.** A frozen encoder and linear head would test the representation-learning direction suggested by the foundation-model papers without repeating their expensive pretraining. Compatibility, weights availability, overlap and actual VRAM use must be checked first.
4. **Treat distillation as a measured trade-off, not a promised gain.** Start with a vanilla small-model baseline, then compare distillation under matching data, seeds and budgets. Include AUROC, calibration, parameter count and measured runtime. A negative result is valid evidence.
5. **Use a truly appropriate external set for new generalization claims.** A dataset with the same parent source may still overlap. Verify provenance and independence before calling an evaluation external.

These are recommendations inferred from the combined evidence, not additional experiments already completed.

## 9. Report-ready conclusion

Recent 2025–2026 chest-radiograph research emphasizes broader pretraining, transfer, efficiency and interpretability. Our study addresses a narrower but appropriate college-project question: how three standard pretrained CNNs compare under a common short recipe on an auditable pediatric pneumonia benchmark.

ResNet-18 produced the highest observed mean AUROC, while MobileNetV3-Small produced the highest mean F1 at threshold 0.5 and used substantially fewer parameters. The paired ResNet experiment supports a local AUROC improvement from partial fine-tuning. The declared architecture intervals do not establish a complete ranking, and the literature's different datasets and evaluation regimes prevent claiming that our larger AUROC is a superior clinical or foundation-model result.

Our contribution is the reproducible comparison and measured trade-off, with explicit uncertainty and hardware limits. The next useful extensions are calibration, independent generalization checks and a controlled medical-encoder or compression experiment.
## 10. References and extraction audit

### 10.1 Primary papers

1. **Ma, D., Pang, J., Gotway, M. B. and Liang, J. (2025).** *A fully open AI foundation model applied to chest radiography.* Nature 643, 488–498. DOI: 10.1038/s41586-025-09079-8. [Publisher](https://www.nature.com/articles/s41586-025-09079-8), [public Fig. 2 caption](https://www.nature.com/articles/s41586-025-09079-8/figures/2), [published supplementary data](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09079-8/MediaObjects/41586_2025_9079_MOESM4_ESM.xlsx).
2. **Yue, Y., Wang, Y., Tao, C., Liu, P., Song, S. and Huang, G. (2025).** *CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning.* CVPR 2025. Numerical tables read in arXiv:2504.13820v1, 18 April 2025. [Official conference record](https://cvpr.thecvf.com/virtual/2025/poster/34038), [primary paper text](https://arxiv.org/html/2504.13820v1).
3. **Yao, J. et al. (2025).** *EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning.* npj Digital Medicine. Published 17 November 2025. DOI: 10.1038/s41746-025-02032-z. [Publisher and result descriptions](https://www.nature.com/articles/s41746-025-02032-z).
4. **Han, T. et al. (2026).** *CLEAR: an auditable foundation model for radiology grounded in clinical concepts.* Nature Biomedical Engineering. Published 22 July 2026. DOI: 10.1038/s41551-026-01741-4. [Publisher, Results and Fig. 3](https://www.nature.com/articles/s41551-026-01741-4).
5. **Neo, R. Y. S. (2026).** *Benchmarking knowledge distillation for lightweight pneumonia detection: a multi-seed calibration study on PneumoniaMNIST.* BMC Research Notes 19, 270. Published 11 May 2026. DOI: 10.1186/s13104-026-07793-8. [Publisher](https://link.springer.com/article/10.1186/s13104-026-07793-8), [open full text and tables](https://pmc.ncbi.nlm.nih.gov/articles/PMC13330209/).

### 10.2 Where the numbers were checked

| Evidence | Exact locator used | Scope of verification |
|---|---|---|
| Ark+ | Supplementary workbook, `Sec. 2.1 ChestX-ray14`, C6:C7 and C10:C11; published Fig. 2 caption | Point mean AUCs and probing/fine-tuning regimes; full subscription main text was not required or claimed to have been read |
| CheXWorld | April 2025 v1, Table 1 and Table 3; implementation paragraph | Benchmark and ablation means; metric distinction and reported pretraining resources; conference status independently verified |
| EVA-X | Results: “Pre-training: performance, efficiency, and flexibility” and “Transfer learning on chest pathologies classification” | CXR14 mean-AUC references, reported sizes and model interpretation |
| CLEAR | Results: zero-shot findings and concept-based supervised representations; Fig. 3f | Zero-shot means and the before/after concept intervention; not combined into one endpoint |
| Neo | Methods, Tables 1–2 and significance discussion | AUROC, parameter/runtime trade-offs, seeds, training details and lack of consistent KD gain |
| Our models | Accepted comparison CSV, individual metrics CSV, paired-differences CSV and two-session hardware CSV | Accepted local scores; arithmetic differences calculated from stored means; no new predictive measurements |

The retained Ark+ workbook is available [inside this project](../02_literature/2025_2026_sources/Ark_plus_Nature_2025_supplementary_data.xlsx). It was read without editing. Its SHA-256 is:

```text
61a2feed5de06b3fea883c407ddab99e83eb5466e1acd4bf0a9634d1f62b13e8
```

The Ark+ main page exposed subscription preview content, so quantitative claims were checked against the publisher's public figure caption and the original supplementary workbook. CheXWorld's numbers are from the version identified above; no claim is made that an unread later revision is numerically identical. Literature improvements in this report are arithmetic contrasts of reported point values, not our replication results or newly calculated literature significance tests.

**Final scope statement:** this document supports a grounded related-work and comparison chapter. It does not establish that our model beats modern foundation models, that the five selected papers are the most-cited studies in the field, or that our benchmark validates clinical deployment.