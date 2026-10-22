# Accepted split records

These records were accepted in Phase 1 on 22 October 2026.

- `phase_1_verification.json`: local dataset-integrity and split-policy checks.
- `official_source_metadata.json`: relevant fields verified against official MedMNIST metadata.
- `train_indices.npy`, `val_indices.npy`, `test_indices.npy`: unchanged copies of the accepted pilot indices.
- `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`: one row per accepted example.
- `excluded_train_examples.csv`: the ten removed training rows and their matching held-out indices.

CSV fields are `split`, `source_index`, `label`, `label_name`, `image_sha256`, and `npz_member`. `source_index` refers to the original row of the named image array in the canonical NPZ file. It is not a patient identifier.

The canonical NPZ stays at `archive/pilot_2026-10-22/data/pneumoniamnist_224.npz`, relative to the project root. Apply these indices to the original arrays; do not renumber source indices or regenerate a random split.

Within-split duplicates are preserved and disclosed. See `../PHASE_1_DATA_ACCEPTANCE.md` for the accepted policy.