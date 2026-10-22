# Phase 1 — dataset and split acceptance

**Status: COMPLETE / ACCEPTED for the defined college benchmark.** Verified: 22 October 2026 (IST).

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

## 1. Acceptance decision

Accept the existing PneumoniaMNIST+ 224 file and the retained split indices used by the pilot. Reuse these assets for the documented pediatric pneumonia-versus-normal comparison. No new download, random re-split, raw-data rewrite, training, or model inference was needed for this phase.

This acceptance is limited to the recorded benchmark policy. It does not certify patient independence, absence of near duplicates, clinical validity, or a newly unseen test set.

## 2. Provenance and integrity

The [official MedMNIST metadata](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/medmnist/info.py) identifies PneumoniaMNIST as pediatric binary classification, with `0 = normal`, `1 = pneumonia`, one channel, and original split sizes of 4,708 / 524 / 624. It lists the 224-resolution release URL, checksum, and CC BY 4.0 license. The relevant retrieved fields are preserved in `manifests/official_source_metadata.json`.

- Dataset source: [MedMNIST+ release](https://zenodo.org/records/10519652).
- Construction reference: [MedMNIST v2 dataset paper](https://www.nature.com/articles/s41597-022-01721-8).
- Accepted local file, relative to the project root: `archive/pilot_2026-10-22/data/pneumoniamnist_224.npz`.
- Size verified locally: **214,384,716 bytes**.
- MD5 verified locally and matched to official 224 metadata: `d6a3c71de1b945ea11211b03746c1fe1`.
- SHA-256 recorded locally: `2a6faaca43872b4bcf4154a79625631ab29683b41e265ab7936d0513351fe591`.
- Image arrays: grayscale `uint8`, 224 × 224, observed values 0–255.
- Label arrays: `uint8`, shape N × 1, values restricted to 0 and 1.
- Exactly the six expected image/label arrays are present; no patient identifier array is provided.

The MD5 was checked before and after the audit. The raw file was unchanged.

## 3. Accepted splits and class balance

| Split | Original images | Accepted images | Normal (0), accepted | Pneumonia (1), accepted | Excluded images |
|---|---:|---:|---:|---:|---:|
| Training | 4,708 | 4,698 | 1,214 | 3,484 | 10 |
| Validation | 524 | 524 | 135 | 389 | 0 |
| Test | 624 | 624 | 234 | 390 | 0 |
| Total | 5,856 | 5,846 | 1,583 | 4,263 | 10 |

Class mapping is accepted from the official metadata, rather than inferred from image appearance. Training loss class weighting, if retained in Phase 3, must use accepted training counts only: `1,214 / 3,484 = 0.3484500574` for the positive-class weight. Validation/test prevalence must not determine training weights.

## 4. Exact-duplicate findings

Image SHA-256 values were recomputed directly from the raw pixel arrays.

| Check | Original splits | Accepted splits |
|---|---:|---:|
| Distinct hashes shared by training and validation | 8 | 0 |
| Distinct hashes shared by training and test | 0 | 0 |
| Distinct hashes shared by validation and test | 0 | 0 |

Those eight shared training/validation hashes affected **ten training rows**, all carrying pneumonia labels. The retained training index file excludes precisely those rows. An exclusion manifest records every removed source index and its matching validation indices.

| Split | Original duplicate excess | Accepted duplicate excess | Accepted unique image hashes |
|---|---:|---:|---:|
| Training | 18 | 16 | 4,682 |
| Validation | 0 | 0 | 524 |
| Test | 3 | 3 | 621 |

“Duplicate excess” means image rows beyond the number of unique pixel hashes, not the number of repeated patients. No conflicting labels were found within exact-image hash groups in any split.

Within-split duplicates are preserved to reproduce the pilot's training exposure and benchmark evaluation. Removing the remaining training or test duplicates later would constitute a separately versioned data policy; the current results must not be relabeled as if trained on such a deduplicated dataset. Test uncertainty analysis should respect the 621 exact-image clusters, while acknowledging that unknown patient dependence remains.

## 5. Accepted split policy

1. Preserve the original NPZ file.
2. Preserve every original test row and its order.
3. Exclude any validation row identical to a test image; there were no such rows here.
4. Exclude training rows identical to preserved test or retained validation images; this removes ten rows here.
5. Preserve remaining within-split duplicates and disclose them.
6. Use explicit source-index files; do not merge splits or create a new random split.
7. Use training for fitting and validation for development decisions. The already examined test split remains a disclosed benchmark evaluation; further test-driven tuning is outside this accepted policy.

The saved indices were independently checked for integer type, bounds, uniqueness, increasing source order, expected counts, and exact agreement with the reconstructed policy. Test indices are exactly 0 through 623. Copies in `03_data/manifests/` match the archived originals by SHA-256.

## 6. Accepted records

Paths below are relative to `03_data/`.

| Record | Purpose |
|---|---|
| `manifests/phase_1_verification.json` | Fresh checksum, array, class-count, duplicate, and retained-index checks |
| `manifests/official_source_metadata.json` | Official label mapping, release URL, counts, and license snapshot |
| `manifests/train_indices.npy` | Accepted original training row indices |
| `manifests/val_indices.npy` | Accepted original validation row indices |
| `manifests/test_indices.npy` | Preserved original test row indices |
| `manifests/train_manifest.csv` | Training source index, label, hash, and array member |
| `manifests/val_manifest.csv` | Validation source index, label, hash, and array member |
| `manifests/test_manifest.csv` | Test source index, label, hash, and array member |
| `manifests/excluded_train_examples.csv` | Ten exclusions and matching held-out source indices |

The CSV source indices identify rows in the packaged arrays; they are not patient IDs. Manifest checksums are recorded in the verification JSON.

## 7. Required limitations and attribution

The data are pediatric, processed images for a binary task. Patient IDs are absent, so patient separation cannot be independently established. Exact hashes do not detect all repeat patients or near duplicates. The preserved test split contains repeated identical images and its scores have already been examined. Accepting this dataset does not support adult fourteen-finding or external-hospital claims.

The final report must cite MedMNIST, the official release, and the original source-data work identified by the MedMNIST paper. Retain CC BY 4.0 attribution and describe this benchmark's educational/research scope.

## 8. Exit checklist

- [x] Official provenance, task, label mapping, resolution, and license recorded.
- [x] Existing local checksum matched to the official 224-resolution checksum.
- [x] Arrays, dtypes, labels, image dimensions, and value ranges checked.
- [x] Original and accepted class counts recorded.
- [x] Retained indices independently checked against the cleanup policy.
- [x] Zero exact cross-split image overlap confirmed after cleanup.
- [x] Remaining within-split duplicates and missing patient IDs disclosed.
- [x] Raw dataset and archived split records preserved.
- [x] Active data manifests saved under the project root.

**Next milestone:** Phase 2 — environment and pipeline readiness. No additional manual dataset work is required from the user for Phase 1.