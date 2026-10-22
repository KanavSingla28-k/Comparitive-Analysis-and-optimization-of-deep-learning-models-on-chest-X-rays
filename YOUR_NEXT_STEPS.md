# Your next steps after Phases 0, 1, and 2

Phases 0, 1, and 2 are complete. Dataset acceptance and the existing CUDA/pipeline setup have been checked. Your optional parallel tasks remain the scope/submission review below; no installation or terminal work is assigned to you at this stage.

## 1. Read the agreed scope — about 3 minutes

Open [the Phase 0 decision record](01_planning/PHASE_0_DECISION_RECORD.md). Be comfortable explaining:

> We compare ResNet-18, DenseNet-121, and MobileNetV3-Small on a pediatric pneumonia-versus-normal benchmark. We compare predictive performance and resource costs, then add a small optimization experiment.

The scope is pediatric binary classification. The original broader title may need an accurate subtitle or scope statement in the college report.

## 2. Add the college details you already have — about 5 minutes

Fill [PROJECT_DETAILS.md](01_planning/PROJECT_DETAILS.md) with your team identifiers, course/guide details, official title, submission format, and exact submission instructions if known.

Save any assignment brief, rubric, or report template in `07_report/requirements/`. Check for actual mandatory requirements such as a specified dataset, a minimum model count, a demo, or novelty. If you already have guidance accepting the topic, record it. These administrative details can be completed alongside the next technical milestone; seeking a new approval is not a prerequisite imposed by this plan.

## 3. Read the result summary and limitations — about 5 minutes

Open [the archived comparison report](archive/pilot_2026-10-22/comparative_analysis_report.md). Read the abstract, comparative-results table, and limitations.

Keep three observations clear:

- ResNet had the highest observed mean AUROC; that alone does not establish statistically reliable superiority over DenseNet.
- MobileNet had the highest mean F1 at the fixed threshold and the fewest parameters. A model can lead on one metric and not another.
- The existing test scores are already known, so this is documented benchmark evidence rather than a new blinded evaluation.

## 4. Prepare a short explanation for your guide or viva

Try answering these in your own words:

1. **What is predicted?** A dataset label: pneumonia versus normal on pediatric chest X-ray images.
2. **Why pretrained CNNs?** They provide reusable image features and keep training practical for a small college project.
3. **Why both AUROC and F1?** AUROC describes ranking across thresholds; F1 summarizes precision and recall at a selected threshold.
4. **What makes the comparison controlled?** A recorded common data/preprocessing/training recipe, multiple seeds, and traceable outputs.
5. **What is the optimization proof?** A matched ResNet fine-tuning comparison and measured precision/resource probes, each with stated limits.

You do not need to claim a new architecture or doctor-level diagnosis.

## What happens next

**Phase 3 — freeze the comparison protocol** is next. [Dataset acceptance](03_data/PHASE_1_DATA_ACCEPTANCE.md) and [GPU/pipeline readiness](04_implementation/PHASE_2_READINESS.md) are complete. No manual dataset download or environment installation is needed. Keep the laptop connected to power during any later GPU execution.

No training or further test inspection is needed for your manual tasks above. Keep all new project documents in this root.