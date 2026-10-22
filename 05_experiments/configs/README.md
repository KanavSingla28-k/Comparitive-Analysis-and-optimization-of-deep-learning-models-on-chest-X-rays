# Version-1 protocol records

`comparison_protocol_v1.json` records the complete adopted recipe and analysis/timing definitions. `run_matrix_v1.json` identifies the required evidence. `protocol_compatibility_v1.json` records the Phase 3 metadata review.

`protocol_freeze_v1.json` pins eighteen specification/source/evidence files by SHA-256. `protocol_freeze_verification_v1.json` records the verified freeze-file hash, pin count, unique run count, and matching original training/analysis source hashes.

Preserve version 1. A later correction or experimental change should create a superseding version and an explicit change note. Mutable milestone status and later run acceptance belong in separate records.

The archived data and source remain unchanged. Later report acceptance must clarify that synthetic latency inputs were shape-matched random draws per case, whereas real-image numerical precision checks used matched images.