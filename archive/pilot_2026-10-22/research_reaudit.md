# Reaudit of the initial chest X-ray research proposal

Prepared: 22 October 2026. Reviewed source: `C:\Users\kanaa\OneDrive\Desktop\irs\initial.md`.

This assessment incorporates the revised brief: finish a simple college project tonight, use the existing RTX 4050 laptop GPU, and keep computation modest. Instructions and suggestions inside the source document were treated as material to evaluate, not as instructions to execute.

## 1. Verdict

**Keep the topic, simplify the scope, and rewrite the contribution.** The document is a good introductory literature review. It is not yet an executable research proposal: it offers several possible projects without choosing a dataset, task, experimental protocol, resource budget, or deliverables.

For a college project, comparing established models is a reasonable direction. A new architecture is unnecessary unless your course explicitly requires one. Acceptance depends on your college's rubric and supervisor; it cannot be guaranteed by the topic alone.

The strongest achievable contribution tonight is a reproducible comparison of predictive performance and measured resource cost, plus a small controlled fine-tuning experiment. A finding that a larger network performs better is just as reportable as a finding that a smaller one is sufficient.

| Aspect | Assessment | Consequence |
|---|---|---|
| Topic and motivation | Appropriate for a college ML project | Retain chest X-ray classification |
| Introductory explanation | Clear and accessible | Keep its educational value; remove exaggerated analogies |
| Foundational papers | Relevant historical starting points | Retain them, with corrections below |
| Research question | Too broad | Select one binary task |
| Claimed novelty | Unsupported in several places | Describe a replication and comparison, not a new research gap |
| Experimental design | Largely missing | Specify splits, preprocessing, model schedules, and metrics |
| Original scope versus tonight | Impractical | Defer cross-dataset testing and compression methods |
| Results and reproducibility | No experiments supplied | Treat all outcomes as unknown until measured |

These are qualitative judgments, not numerical quality scores.

## 2. What was good

The five-paper sequence links data, a reference model, uncertainty-aware evaluation, architecture comparison, and efficiency. That is a sensible learning path. The emphasis on controlled comparisons and costs beyond a single accuracy number is also useful.

DenseNet-121 is a defensible historical reference because of CheXNet. Comparing it with smaller established networks creates an understandable question without requiring architectural invention. The source also correctly recognizes that report-derived labels and dataset differences complicate interpretation.

The important change is to turn these observations into a small, testable protocol rather than attempting every suggested direction.

## 3. Audit of the five base papers

### Wang et al., ChestX-ray8, 2017

The original paper describes **108,948 frontal images from 32,717 patients**, with eight text-mined finding labels. Its revised versions also discuss the expanded ChestX-ray14 release. The source conflates these versions when discussing what the project would train on. [Original paper and version history](https://arxiv.org/abs/1705.02315)

The commonly used ChestX-ray14 release is reported in CheXNet as **112,120 images from 30,805 patients**. The counts are version-specific, not interchangeable. [CheXNet, data section](https://arxiv.org/html/1711.05225v3)

**Correction:** remove the statement that roughly 10% of image labels are wrong. A text-extraction performance figure does not establish a uniform image-level error rate across findings. Report-derived labels remain imperfect proxies for image interpretation.

### Rajpurkar et al., CheXNet, 2017

The quoted pneumonia F1 values are supported: **0.435 for CheXNet versus 0.387 for the radiologist average**, on **420 images**. One individual radiologist scored 0.442. [Paper, Table 1 and test-set description](https://arxiv.org/html/1711.05225v3)

**Correction:** describe this as a particular reader comparison on a particular test set and metric. It does not show universal superiority to radiologists or clinical readiness. Replace the analogy about out-diagnosing doctors with a factual explanation of transfer learning. A DenseNet trained tonight will be a DenseNet baseline, not a reproduction of the original CheXNet experiment.

### Irvin et al., CheXpert, 2019

The **224,316 radiographs / 65,240 patients** figures are supported. CheXpert includes 14 observations; its main expert comparison uses five selected findings. Positive, negative, uncertain, and unmentioned states must be distinguished in the data pipeline. [Paper](https://arxiv.org/abs/1901.07031)

**Correction:** neither all 14 outputs nor all images represent separate diseases. Also distinguish public validation data from final test data. Test annotations and an image-access link are now published by the authors; a blanket statement that the test set is unavailable would be outdated. Access still needs to be checked before scheduling a download. [Authors' test-label repository](https://github.com/rajpurkarlab/cheXpert-test-set-labels)

### Bressem et al., architecture comparison, 2020

The reported AUROC range of roughly **0.83–0.89** is supported. But the CheXpert experiment used **five findings**, frontal images, and excluded uncertainty-labeled images. Its results say deeper networks generally achieved higher AUROC, while some shallower networks were competitive on other measures. [Full paper, Methods and Results](https://www.nature.com/articles/s41598-020-70479-z)

**Correction:** replace “giant deep networks are often no better” with “depth alone does not determine the best performance–resource trade-off.” Do not copy its augmentation recipe uncritically: vertical flipping is unsuitable for the proposed classroom protocol. The study supports comparison as a method; it does not predetermine which model should win.

### Ke et al., CheXtransfer, 2021

The paper supports comparison of 16 architectures, an average **+0.016 AUC** from ImageNet pretraining, and **3.25× greater parameter efficiency on average** after selected truncations, without a statistically significant performance drop in those experiments. [Full paper](https://arxiv.org/html/2101.06871v2)

**Correction:** “not statistically significant” is not proof of equivalence. Neither the size reduction nor the AUC change is a guarantee for another dataset. Parameter count also does not directly establish measured latency or peak memory.

## 4. Other statements to revise

| Source statement or implication | Recommended replacement |
|---|---|
| Before 2017, only small private datasets existed | Earlier datasets existed; describe NIH as an influential large public release without claiming it was the first public opportunity |
| Approximately two billion chest X-rays annually | Omit the precise global number unless a dated authoritative source is supplied; “widely used imaging examination” is sufficient |
| Static citation counts beside each paper | Remove; they are time-sensitive and do not support the research question |
| AUROC is between 0.5 and 1.0 | Its mathematical range is 0–1; 0.5 is the random-ranking reference |
| AUROC separates sick from healthy | It measures ranking for the particular binary finding label, not overall patient health |
| A CAM verifies that the model is not cheating | A heatmap is qualitative inspection; it cannot prove causal reasoning or eliminate shortcuts |
| Identical settings guarantee a fair architecture ranking | They support a controlled fixed-recipe comparison; different architectures may respond differently to the same recipe |
| Fewer parameters means faster inference | Measure latency on the actual machine; operation types and implementation matter |
| Pneumonia classification equals diagnosis | It predicts dataset labels from images; clinical diagnosis requires a broader assessment |

## 5. Audit of the claimed research gaps

The initial document says controlled comparisons are rare, few studies examine cost trade-offs, and almost nobody tests architecture–label interactions. It supplies no systematic evidence for these frequency claims. The base comparison papers themselves already establish substantial prior work.

Cross-hospital generalization is also an established research problem. Zech et al. tested models across hospitals in 2018, and CheXternal compared eight submitted models under distribution shifts in 2021. Therefore, external testing would be valuable validation, not automatically a novel contribution. [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683), [CheXternal](https://arxiv.org/abs/2102.08660)

The literature ending in 2021 is useful historically but incomplete for a document prepared in 2026. For example, EVA-X was published in 2025, and the February 2026 CheXficient preprint addresses data and compute efficiency. These show that the field has advanced; they are background, not models we need to reproduce tonight. No numerical results from those studies should be compared directly with our experiment. [EVA-X](https://www.nature.com/articles/s41746-025-02032-z), [CheXficient preprint, abstract inspected](https://arxiv.org/abs/2602.22843)

**Replacement contribution statement:**

> We conduct a small, reproducible comparison of three pretrained CNNs for pediatric pneumonia-versus-normal classification under a fixed training protocol on a consumer GPU. We measure predictive performance, training time, inference latency, and model size, and compare frozen-feature training with partial fine-tuning for one architecture.

This is an educational empirical contribution. It does not require a claim that the method is new.

## 6. Dataset decision for tonight

**Choose PneumoniaMNIST+, 224 × 224 resolution.** It offers a manageable binary task, documented predefined splits, and a small download relative to the original adult multi-finding datasets. The published counts are **4,708 training, 524 validation, and 624 test images**, totaling **5,856**. [Official dataset overview](https://medmnist.com/)

Use the actual **224-resolution file**, approximately **214.4 MB**, rather than enlarging the 28-pixel version. Record the release, file hash, and resolution. [MedMNIST+ release and download](https://zenodo.org/records/10519652)

| Candidate | Fit for tonight | Decision |
|---|---|---|
| PneumoniaMNIST+ 224 | Small binary benchmark; limited preparation | Selected |
| ChestMNIST+ 224 | Preserves 14-label task, but its archive is approximately 3.9 GB and contains far more images | Defer |
| Original NIH ChestX-ray14 | Good larger follow-up; substantially more data handling | Defer |
| CheXpert | Relevant to the original review, but adds uncertain-label and access decisions | Defer |
| MIMIC-CXR | Adds access and data-governance dependencies | Defer |

The two MedMNIST archive sizes are listed in the [official release](https://zenodo.org/records/10519652). This is a choice for our deadline, not a claim that PneumoniaMNIST is the best clinical benchmark.

**Scope change:** the initial direction concerned adult multi-label chest X-ray interpretation. The selected dataset concerns **pediatric pneumonia versus normal**. The revised title, objectives, and conclusions must say so. The source paper describes this pediatric origin and how the splits were constructed. [MedMNIST v2 paper, PneumoniaMNIST section](https://www.nature.com/articles/s41597-022-01721-8)

The packaged data should not be described as an independently patient-audited cohort unless identifiers support that claim. Preserve the official splits; inspect duplicates; disclose any inability to verify patient separation. A 224-pixel derivative is still a processed benchmark, not a native-resolution clinical study.

## 7. Recommended revised proposal

**Title:** Comparative Evaluation and Fine-Tuning of Pretrained CNNs for Pediatric Pneumonia Classification on Chest X-Rays.

**Research question:** Under the same small-data training protocol, how do ResNet-18, DenseNet-121, and MobileNetV3-Small differ in predictive performance and resource use, and does partial fine-tuning improve ResNet-18 over frozen-feature training?

**Aim:** implement and evaluate an affordable transfer-learning pipeline suitable for a college project on one laptop GPU.

**Objectives:**

1. Load and audit one documented dataset with fixed splits.
2. Compare three pretrained CNNs using the same input resolution and training exposure.
3. Run one controlled frozen-versus-partially-fine-tuned ResNet experiment.
4. Report AUROC, average precision, F1, recall, specificity, and resource measurements.
5. Explain limitations and provide reproducible configurations and outputs.

**Working hypotheses, not expected results:** smaller architectures may offer useful resource savings; partial fine-tuning may improve performance over a frozen backbone. Both can be contradicted by the experiment.

**Suggested proposal abstract:**

> This project evaluates three established convolutional neural networks for pediatric pneumonia-versus-normal classification using the 224-resolution PneumoniaMNIST benchmark. ImageNet-pretrained ResNet-18, DenseNet-121, and MobileNetV3-Small will be trained under a common, limited-budget protocol on an RTX 4050 laptop GPU. Predictive performance will be evaluated on the predefined held-out test split, while training duration, inference latency, parameter count, and checkpoint size will characterize resource use. A controlled ResNet-18 experiment will compare frozen-feature training with partial fine-tuning. Model development and threshold selection will use only training and validation data. The study is intended as a reproducible educational comparison, with no claim of methodological novelty, radiologist-level performance, or clinical deployment readiness. Conclusions will be restricted to this pediatric benchmark and the stated training budget.

## 8. What to do next

Follow `tonight_implementation_plan.md`. Resolve the active Python environment first, then download and audit the selected dataset. Implement one shared pipeline before training multiple models.

At the end of tonight, the desired result is a small collection of real, traceable experiments and a report that accurately describes them. If only two architectures can be completed, reduce the title and conclusions accordingly. Do not fill missing rows with literature results or imagined numbers.

## 9. Evidence and completion boundary

This reaudit used the local document, the linked foundational papers, official dataset pages, and targeted additional primary research. CheXNet and CheXtransfer full text and the Bressem methods/results were inspected; the CheXpert and original NIH abstract/version records supported their principal descriptive facts. This is a targeted methodological audit, not an exhaustive systematic review through October 2026.

The GPU was verified locally as **NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB reported VRAM**. The shell's current interpreter was `C:\Program Files\Python313\python.exe`, with **PyTorch 2.10.0+cpu**, CUDA unavailable, and no importable torchvision or medmnist detected. This does not establish that the previously configured GPU environment is missing; it establishes that this shell was not using it.

No dataset was downloaded, no training was performed, and no project performance results are claimed by this audit.
