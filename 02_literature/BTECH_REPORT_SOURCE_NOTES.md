# Source notes for the BTech report

Reviewed and checked on 19 November 2026. These notes support `07_report/BTECH_RESEARCH_REPORT.md`; they are not an additional report chapter. The review focused on architectural ideas, relevant methods and reported results. Access to a primary abstract or selected sections is not presented as a complete line-by-line reading of every paper.

| Report reference | Primary material consulted | Claims used |
| --- | --- | --- |
| R1 | Official CVPR paper text/abstract exposed by the publisher search index | Residual learning and the architectural basis of ResNet |
| R2 | Official CVPR abstract and paper text exposed by the publisher index | Dense connectivity and feature reuse |
| R3 | Official ICCV paper abstract/text exposed by the publisher index | Hardware-aware search, mobile CPU focus, Small/Large variants |
| R4 | arXiv:1711.05225 abstract and PDF text | DenseNet-121 chest X-ray application; scope of radiologist comparison |
| R5 | arXiv:1901.07031 abstract and PDF text | Dataset counts, uncertainty-label approach and expert-evaluation setting |
| R6 | Publisher full text: Abstract, Methods and article citation | Sixteen-architecture comparison, datasets and resource motivation |
| R7 | arXiv:2101.06871 abstract and PDF text | Transfer-learning evaluation and truncation; not equated with our fine-tuning |
| R8 | Publisher abstract, official release and earlier accepted dataset audit | Standardized benchmark; original 28-pixel collection distinguished from later 224-pixel release |
| R9 | Publisher abstract, open full text at PMC13657249 and retained supplementary workbook | Ark+ method and CXR14 fine-tuning contrast: 83.54% versus 84.43% |
| R10 | arXiv:2504.13820v1: method, Table 1 and Appendix C | CheXWorld method; VinDr mean AUROC 92.76% versus 95.24% |
| R11 | Publisher full text: transfer-learning results | EVA-X method; CXR14 82.3 versus 83.3 mAUC comparison |
| R12 | Publisher indexed full text: Abstract and zero-shot Results | Concept-based auditing; VinDr 75.0% versus 78.2% mean AUROC |
| R13 | Publisher full text: Abstract, Methods and result tables | Five-seed teacher/student comparison and lack of consistent distillation gain |

All primary links and bibliographic details are in the report references. Some direct publisher URL fetches returned access or cookie errors; accessible primary text, publisher-indexed content and open paper copies supplied the claims above. No numerical claim was taken from a generic blog. No citation-count or most-famous-paper ranking is claimed.

## Local numerical evidence

- Main means, standard deviations, parameters and training costs: `06_results/retrained_v1/comparison_summary.csv`.
- AUROC differences and intervals: `06_results/retrained_v1/paired_differences.csv`.
- Original inference latency and precision evidence: `07_report/FINAL_COMPARATIVE_REPORT.md` and its linked primary CSV tables.
- Accepted data counts and exact-overlap findings: `03_data/PHASE_1_DATA_ACCEPTANCE.md`.
- Optimization means, costs and update counts: `05_experiments/batch32_v2_pilot/summary.csv`.
- Paired timing decision and its explanation: `05_experiments/batch32_v2_pilot/results_explained.json`.

The literature table's relevance column is an assessment of applicability to this project. It does not imply that the original authors failed to address an issue outside their study's scope. Literature scores and local scores are not pooled, and validation results are kept separate from the original test evaluation.
