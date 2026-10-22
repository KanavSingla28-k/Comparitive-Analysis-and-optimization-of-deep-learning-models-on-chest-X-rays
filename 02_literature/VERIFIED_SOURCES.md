# Verified primary sources

Checked on 22 October 2026. Literature scores are not local measurements.

- [R1] Wang et al. (2017). [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/abs/1705.02315). The original release describes report-derived labels for eight findings and 108,948 frontal images. Later ChestX-ray14 updates must be distinguished from that original release. Our pediatric binary dataset is different.

- [R2] Rajpurkar et al. (2017). [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225). The reported radiologist comparison concerns pneumonia F1 on a specific test and annotation setup. It motivates a DenseNet baseline but does not establish universal clinical superiority or validate our models.

- [R3] Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031). CheXpert provides report-derived observation labels with uncertainty and a focused expert comparison. We do not use that dataset or claim to reproduce its multi-finding evaluation.

- [R4] Bressem et al. (2020). [Comparing different deep learning architectures for classification of chest radiographs](https://www.nature.com/articles/s41598-020-70479-z). The authors compared sixteen architectures on CheXpert and a separate COVID-19 collection. Comparable shallower models motivate measuring resource costs; their dataset-specific scores are not our results.

- [R5] Ke et al. (2021). [CheXtransfer: Performance and Parameter Efficiency of ImageNet Models for Chest X-Ray Interpretation](https://arxiv.org/abs/2101.06871). The study evaluates transfer and truncation on CheXpert, reporting an average parameter-efficiency improvement without a statistically significant drop in its comparison. That does not prove equivalence; our fine-tuning control does not reproduce truncation.

- [R6] Yang et al. (2023). [MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification](https://www.nature.com/articles/s41597-022-01721-8). The 2023 paper describes the standardized MedMNIST v2 collection at 28-pixel resolution. Our actual 224-pixel MedMNIST+ file is identified separately by its release record and checksum, rather than being an upscaled copy of the paper's 28-pixel images.

The MedMNIST paper was published in 2023 despite the DOI containing 2022. The actual 224-pixel file is separately identified by the official MedMNIST+ release and Phase 1 checksum, rather than being an upscaled 28-pixel image array.
