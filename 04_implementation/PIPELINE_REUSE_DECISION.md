# Pipeline reuse decision — Phase 2

Decision date: 22 October 2026.

Adopt the existing archived pipeline as the technical reference for the benchmark, following the successful Phase 2 checks. Preserve its source and original evidence. A duplicate or rewritten implementation is not required merely to advance the milestones.

Reuse these behaviors: data-array loading with explicit retained indices, common preprocessing, architecture/head creation, trainable-module selection with fixed BN statistics, binary-loss training behavior, checkpoint loading, and probability saving.

Active project documents and generated evidence belong under the Desktop\irs root. The raw file and existing model weights can be read from the archive. New formal runs, only if later required, need versioned paths in `05_experiments/`; accepted analytical outputs belong in `06_results/`.

The archived CLI retains archive-relative write defaults. Future new execution must explicitly bind active output paths before using its mutating functions. This phase verified function-level reuse with redirected readiness outputs, rather than approving archive-overwriting CLI invocations.

The existing GPU Python environment is reused without modification. Project dependencies already reside in the preserved local `.deps` folder. Readiness caches are inside this root. No package installation or new production code was needed.

Phase 3 will match and freeze the detailed protocol and evidence version. Any later algorithmic change is a new experiment decision, not a silent repair of preserved results.