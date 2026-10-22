# Scope and decisions — adopted in Phase 0

Phase 0 completed: 22 October 2026 (IST). Full record: [Phase 0 decision](PHASE_0_DECISION_RECORD.md).

## Project root

`C:\Users\kanaa\OneDrive\Desktop\irs` is the root for all subsequent project work and outputs. The former Documents workspace remains a recovery copy. The dated pilot archive stays preserved inside this root.

## Adopted scope

**Working title:** Comparative Analysis of Pretrained CNNs for Pediatric Pneumonia Classification with an Initial Efficiency Study.

**Research question:** How do three pretrained CNNs compare in pediatric pneumonia-versus-normal classification and resource use under a shared short training protocol, and what does a limited optimization experiment demonstrate?

| Item | Adopted decision |
|---|---|
| Dataset | PneumoniaMNIST+, actual 224 × 224 release |
| Task | Pediatric binary pneumonia versus normal |
| Models | ResNet-18, DenseNet-121, MobileNetV3-Small |
| Hardware | Existing RTX 4050 Laptop GPU, approximately 6 GB VRAM |
| Study route | Reuse suitable archived benchmark evidence following milestone acceptance checks |
| Reference repeats | Seeds 42, 43, 44 for exact pilot reproduction |
| Reference schedule | Two head-only epochs, three partial fine-tuning epochs |
| Main metric | AUROC; also report AP, fixed-threshold metrics, and resource costs |
| Threshold | 0.5 |
| Core matrix | Nine comparison runs; three additional frozen ResNet controls support optimization |
| Priority | Comparison and report first; limited optimization second |
| Contribution | Reproducible educational comparison; architectural novelty is not a requirement of this working plan |

## Evidence and evaluation history

Phase 0 found twelve run manifests with matching checkpoint/prediction files. The archived training and analysis source hashes match their recorded hashes. The previous verification record is available. These observations support reuse in principle; Phases 1–6 still perform their defined acceptance reviews.

The pilot test results have already been examined. The report must describe that history and must not call the next use of this split a newly blinded confirmation. A later external-generalization claim would need a separately scoped evaluation.

## Decisions completed

- [x] Adopt the pediatric binary scope and working title.
- [x] Use the existing assets rather than repeat every run by default.
- [x] Preserve comparison priority and limited optimization scope.
- [x] Define the previously examined test-set policy.
- [x] Carry the archived technical recipe forward as the Phase 3 reference.

## Administrative details still useful

Team/student identifiers, official course title, guide details, report template, marking rubric, and exact submission instructions can be filled in `PROJECT_DETAILS.md` when available. Their absence does not prevent the next technical milestone. No college or supervisor approval is inferred.

## Next milestone

Phases 1 and 2 are complete; see dataset acceptance and `../04_implementation/PHASE_2_READINESS.md`. Next: Phase 3 — freeze the detailed comparison protocol. No new comparison training is required merely to advance that review.