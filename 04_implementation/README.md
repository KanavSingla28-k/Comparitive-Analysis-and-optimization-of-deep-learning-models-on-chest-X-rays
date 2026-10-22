# Runnable implementation and reproduction guide

Project root: `C:\Users\kanaa\OneDrive\Desktop\irs`.

This project includes executable model training, inference, duplicate auditing, checkpoint verification, bootstrap statistics, plots and hardware probes. Study settings live in **`src/settings.py`**, not a JSON configuration. JSON records results and their descriptions; CSV, Markdown, PNG, NPZ and PT files provide the other evidence.

## Start on this laptop

Open PowerShell in the project root. The launcher selects the existing tested CUDA Python environment automatically; nothing needs to be installed on this laptop.

```powershell
Set-Location -LiteralPath 'C:\Users\kanaa\OneDrive\Desktop\irs'
.\run.ps1 verify
```

For a quick supervisor demonstration, replay all saved checkpoints and regenerate the figures, statistics and explained results without retraining. Choose a directory that does not already contain artifacts:

```powershell
.\run.ps1 reproduce-results -RunDir '05_experiments/supervisor_demo_01'
```

For a complete fresh training run, including the optimization probes:

```powershell
.\run.ps1 full -RunDir '05_experiments/supervisor_training_01'
```

Both commands preserve the pilot archive and use the canonical dataset already stored inside this project. A full run trains nine architecture/seed configurations plus three paired frozen ResNet controls. It uses batch 16 on the RTX 4050; no external cluster is required. Runtime depends on laptop load and power mode. Keep the laptop plugged in and avoid another GPU job during resource timing.

If local PowerShell policy prevents running a script, invoke Python directly without changing machine policy:

```powershell
& 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' -B -u '04_implementation/src/pipeline.py' full --run-dir '05_experiments/supervisor_training_02'
```

## Execute individual stages

Use the SAME run directory throughout a fresh run:

```powershell
.\run.ps1 prepare   -RunDir '05_experiments/manual_run_01'
.\run.ps1 train     -RunDir '05_experiments/manual_run_01'
.\run.ps1 evaluate  -RunDir '05_experiments/manual_run_01'
.\run.ps1 replay    -RunDir '05_experiments/manual_run_01'
.\run.ps1 benchmark -RunDir '05_experiments/manual_run_01'
.\run.ps1 hardware  -RunDir '05_experiments/manual_run_01'
.\run.ps1 analyze   -RunDir '05_experiments/manual_run_01'
```

`prepare` audits the data and runs six disposable smoke checks. `train` resumes completed model/seed units; an incomplete unit restarts its five-epoch recipe rather than resuming optimizer state mid-epoch. The saved final checkpoints contain model weights and buffers, not optimizer state. `evaluate` requires the entire declared run matrix before test prediction. `analyze` requires all predictive and hardware artifacts, because the report includes both. The launcher exits with an error when a stage fails.

Fresh preparation and saved-evidence adoption refuse to overwrite an existing artifact directory. Use a new name for another run. Later individual stages deliberately regenerate their own outputs inside the selected run. Do not change settings or source midway through a run; changing the research recipe requires a new protocol and run directory.

## Source map

| File | What the supervisor can inspect |
|---|---|
| `src/settings.py` | All training settings, models, seeds and analysis constants |
| `src/experiment.py` | `model_factory`, `configure_training`, `ImageArrays.batch`, `train_phase`, `train`, `evaluate`, `benchmark` |
| `src/analyze.py` | Paired hierarchical bootstrap, aggregation, plots and the full comparative report |
| `src/hardware_probe.py` | Disposable full-gradient synthetic training probes |
| `src/pipeline.py` | Command-line stages, dataset/checkpoint verification, checkpoint replay and provenance |
| `src/explained_results.py` | Generate result descriptions from measured numbers and export JSON/CSV/Markdown |
| `src/runtime.py` | Resolve all paths from this root and separate archive inputs from new outputs |

The training and analysis implementations are readable adaptations of the measured pilot source. Core numerical operations and training recipes were retained. New orchestration binds active output paths, replaces JSON-driven training configuration with Python settings and adds provenance and checks. The archive is preserved for comparison.

## Results produced

For run `example`, complete artifacts are under `05_experiments/example/`, and the readable result bundle is under `06_results/example/`.

| Output | Purpose |
|---|---|
| `results_explained.json` | Same JSON contains numeric results, `description` fields, metric definitions, pipeline stages, per-run histories, paired differences, hardware estimates and limitations |
| `comparative_analysis_report.md` | Full comparative analysis with tables, methods, interpretation and linked figures |
| `comparison_summary.csv` | Model-level means, seed SD, size and resource results |
| `metrics.csv` | Twelve individual model/seed scores and confusion counts |
| `epoch_history.csv` | Sixty recorded epoch rows, including duplicated shared ResNet warm-up rows identified by run |
| `paired_differences.csv` | AUROC differences and exploratory interval endpoints |
| `hardware_summary.csv` | Inference latency and synthetic training precision ratios |
| `figures/` | ROC/PR, confusion matrices, training curves, resource trade-offs and hardware scaling |
| `SHA256SUMS.txt` | Integrity hashes for the exported review bundle |

The full run folder additionally contains all twelve PT checkpoints, shared ResNet checkpoints, raw NPZ probabilities in both precisions, bootstrap samples, raw latency timings, manifests, environment versions and source hashes. Paths in the explained JSON point back to these actual artifacts. Explanations are generated deterministically from the measurements; they are not additional experimental evidence.

## Recreate an environment on another machine

The measured environment was Windows, Python 3.13.7, PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128. Exact installed versions are recorded in `requirements-lock.txt`. The existing preserved `.deps` is an offline fallback for this laptop; a new local `.venv` uses its own installed libraries.

After installing Python 3.13 and a compatible NVIDIA driver, run these from this project root:

```powershell
py -3.13 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
& '.\.venv\Scripts\python.exe' -m pip install -r '04_implementation/requirements-lock.txt' --extra-index-url https://download.pytorch.org/whl/cu128
.\run.ps1 verify
.\run.ps1 full -RunDir '05_experiments/new_machine_01'
```

The PyTorch version pair and CUDA wheel index are listed in [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). These installation commands are documented for rebuilding the environment; they were not executed here because the working GPU environment already existed. Exact versions may need accessible package indexes. The whole project archive, accepted split files and frozen protocol records are part of the reproduction package; do not copy only the Python directory.

If the canonical NPZ is absent, `pipeline.py download` downloads and verifies the official file into `03_data/raw/`. Then set `$env:IRS_DATA_FILE` to that absolute NPZ path before continuing. The normal complete package already includes the data and official pretrained weights, so normal reproduction does not need a dataset download. Training preparation copies cached pretrained weights into the active project cache; otherwise torchvision obtains its official weights.

## What reproduction means

1. Artifact verification checks that data, saved weights and outputs match recorded checksums and accepted membership.
2. Checkpoint replay executes all 624 test images for every checkpoint in AMP and FP32 and compares them with saved predictions within absolute tolerance 1e-6.
3. Analysis reproduction independently rebuilds the bootstrap draws, aggregate scores, CSV tables and figures from raw saved probabilities.
4. Fresh training repeats the full recipe from the official pretrained weights and records a separate set of measurements.

GPU timing changes with load and thermal/power state. Bitwise-identical training across different platforms or releases is not promised. Three seeds, deterministic cuDNN settings, disabled TF32 and fixed sampling schedules improve repeatability. See [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

The test scores were examined before the formal evidence-adoption protocol. No rerun changes that history. This is a college benchmark comparison with exploratory uncertainty, not a prospective clinical validation.
The project Git ignore rules exclude binary datasets, weights, dependency caches and logs. A source-only Git checkout therefore does not contain the whole offline evidence package. Preserve the complete project folder (including the dated archive and accepted indices), or obtain the official data/weights using the documented commands and retain the recorded evidence needed for saved-result reproduction.

## Final statistical review and report assembly

The next phases have a runnable CPU-only review/packaging entry point:

```powershell
.\finalize.ps1 review
.\finalize.ps1 all
.\finalize.ps1 verify-package
```

`review` verifies reproduction and independently recomputes predictive metrics, the paired bootstrap and timing summaries. `all` additionally generates the accepted report, optimization note, source index and file integrity manifest. `verify-package` checks that retained files still match that manifest. These commands do not retrain models or tune against test scores.

Review code: `src/review_statistics.py`. Assembly code: `src/finalize_project.py`. Final report: `07_report/FINAL_COMPARATIVE_REPORT.md`. Handoff: `08_handoff/START_HERE.md`. Model, paired and hardware results retain descriptions beside numerical evidence in `07_report/final_results_explained.json`; parallel CSV and Markdown are supplied.

Running `all` refreshes the package snapshot after legitimate file changes. Run `verify-package` before editing to detect unintended changes. Active generated cache, Git metadata, Python bytecode, a local virtual environment and manifest/verification outputs are excluded; preserved pilot dependencies and binary evidence are included.