# Pediatric chest X-ray CNN comparative analysis

This workspace implements the research protocol in `configs/experiment.json` and statistical analysis in `configs/analysis_protocol.json`. The task is **pediatric pneumonia versus normal**, using PneumoniaMNIST+ 224. It is an educational comparison, not an adult multi-label study or a clinical diagnostic system.

## Outputs

- `comparative_analysis_report.md`: generated report with measured scores, uncertainty, cost comparison, and initial optimization evidence; available after all stages complete.
- `artifacts/metrics.csv`: one row per model configuration and training seed.
- `artifacts/comparison_summary.csv`: aggregated comparison.
- `artifacts/statistical_summary.json`: bootstrap intervals, paired differences, and timing summary.
- `artifacts/dataset_audit.json`: checksum, label counts, and exact-duplicate cleanup.
- `artifacts/predictions/`: saved test labels, stable sample indices, AMP/FP32 probabilities.
- `artifacts/history/`: epoch logs, training costs, and checkpoint hashes.
- `artifacts/benchmarks/`: raw latency samples, tested inference batches, and synthetic training capacity probes.
- `artifacts/figures/`: report figures.
- `research_reaudit.md`: original proposal audit.
- `tonight_implementation_plan.md`: initial implementation plan. The execution protocol adds three seeds and inference precision/capacity probes; actual outputs and the locked config supersede planning estimates.

## Environment

The existing GPU runtime is `C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe`, with PyTorch 2.10.0+cu128. Missing libraries were installed into workspace-local `.deps`; the runtime imports this directory without altering the existing environment. `requirements-lock.txt` records resolved packages. GPU execution requires native CUDA access on the host.

For another machine, create a dedicated Python 3.13 environment, install PyTorch 2.10.0 and torchvision 0.25.0 from the official CUDA 12.8 index, and install NumPy, SciPy, scikit-learn, Matplotlib, Pillow, and tzdata with compatible locked versions. The original V1 ImageNet weights are downloaded by torchvision into workspace-local `.cache/torch`.

To recreate the local dependency layout with `uv` and the existing CUDA runtime:

```powershell
uv pip install --python 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' --target .deps numpy pillow scikit-learn matplotlib scipy psutil
uv pip install --python 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' --target .deps --no-deps torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

For exact reproduction, select versions from the saved lock rather than resolving new releases. Standard installations may install all dependencies in the same dedicated environment instead of using `.deps`.

## Dataset

Download only `pneumoniamnist_224.npz` from the [official release](https://zenodo.org/records/10519652) into `data/`. Required MD5: `d6a3c71de1b945ea11211b03746c1fe1`. The source train/validation/test sizes are 4,708/524/624. The audit excludes ten training examples identical to validation images, leaving 4,698/524/624. No exact overlap with the test images was found. Patient IDs are absent.

## Run

From this workspace:

```powershell
.\run.ps1 prepare
.\run.ps1 train
.\run.ps1 evaluate
.\run.ps1 benchmark
& 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' -u src\hardware_probe.py
& 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe' -u src\analyze.py
```

`prepare` verifies data, downloads pretrained weights, checks forward/backward updates and frozen weights, and writes the protocol lock. `train` uses seeds 42, 43, and 44 and skips configurations with existing completed manifests. An interrupted architecture can be rerun; remove or relocate its incomplete files only after inspection. Avoid using skip behavior to mix outputs from different configs or code revisions.

`evaluate` refuses to run until all twelve prespecified model/seed configurations have completed. It evaluates fixed final checkpoints, not validation-best or test-selected ones. Thresholded results use 0.5. Test results are saved once for postprocessing; re-running evaluation must not be used to drive additional model tuning.

`benchmark` uses seed-42 trained weights and synthetic device-resident inputs. Three FP32/AMP latency rounds are interleaved. Inference batches up to 128 are probed subject to a 70%-of-VRAM allocated-memory stop rule. `hardware_probe.py` additionally measures synthetic full-gradient training steps at selected batches. Those probes discard modified in-memory copies and never overwrite trained research checkpoints.

The fixed five-epoch comparison does not establish convergence or maximum possible accuracy. The precision proof of concept does not compress the architecture. Bootstrap intervals account for exact-image test clusters and resampled seeds, but cannot account for unavailable patient IDs.

Run `src/verify_results.py` with the recorded GPU Python interpreter to independently recompute AUROC from ranks, verify confusion counts/F1, check checkpoint hashes and shared branch initialization, and confirm the cleaned splits have no exact image overlap. The completed verification is saved in `artifacts/verification.json`.

## Data, weights, and source control

Data, workspace dependencies, cache, and checkpoints are excluded from Git. Saved numerical outputs and code document the experiments. Keep checkpoints locally for verification and future demonstrations; do not silently substitute external literature scores for local results.
