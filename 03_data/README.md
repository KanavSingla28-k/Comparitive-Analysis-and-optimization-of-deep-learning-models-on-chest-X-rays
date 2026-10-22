# Accepted data workspace

**Phase 1 is complete.** Read [dataset and split acceptance](PHASE_1_DATA_ACCEPTANCE.md).

Accepted dataset: PneumoniaMNIST+ 224, pediatric pneumonia versus normal. Official mapping: 0 normal, 1 pneumonia.

| Split | Accepted images | Normal | Pneumonia |
|---|---:|---:|---:|
| Training | 4,698 | 1,214 | 3,484 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

The existing raw file is preserved at `../archive/pilot_2026-10-22/data/pneumoniamnist_224.npz`. It remains the canonical raw asset; another copy or download is unnecessary. MD5: `d6a3c71de1b945ea11211b03746c1fe1`.

Use the accepted index arrays and CSV manifests in `manifests/`. They reproduce the pilot's cleanup: ten training examples identical to validation images were excluded. There is zero exact cross-split overlap in the retained data. Sixteen training duplicate excess images and three test duplicate excess images remain, explicitly preserved for reproduction.

Patient identifiers are absent, and the test scores were already examined. Keep those limitations in subsequent analysis and reporting.

The next technical milestone is Phase 2. No models were executed during dataset acceptance.