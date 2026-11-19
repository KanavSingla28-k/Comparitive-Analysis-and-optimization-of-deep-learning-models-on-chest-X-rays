# Optimization execution status — 19 November 2026

**Status: implemented; GPU execution blocked before training by Windows Application Control. No batch-32 optimization result has been measured or accepted.**

## What is ready

- `04_implementation/src/batch32_pilot.py`: ResNet-18, seeds 42/43/44, physical batch 16 versus 32, three adaptation epochs, two technical timing sessions with reversed order.
- `04_implementation/src/review_batch32_pilot.py`: independent AUROC pair counting, F1 confusion counts, timing/acceptance arithmetic, repeated-prediction checks, source/checkpoint integrity.
- `optimize.ps1`: preflight, training, independent review and final integrity verification; accepts an approved alternative Python path.
- Python syntax checks and the runner's `--help` command passed. GPU execution and numerical correctness have NOT been verified for the new implementation.

The existing v1 code/settings/checkpoints have not been edited. V2 pre-generates one angle vector for the entire epoch to guarantee paired per-image rotations across physical batch sizes, so it reruns its own batch-16 baseline. This deliberate RNG difference is recorded in the runner's protocol and future report.

## Exact runtime blocker

The existing Python executable works, but importing PyTorch fails with:

```text
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading ...\torch\lib\torch_python.dll or one of its dependencies.
```

Windows Code Integrity event 3077, 19 November 2026 at 14:51:46 local time, names the same DLL and policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. Event 3033 also records a signing-level rejection. Authenticode reports the DLL as not signed. The `VerifiedAndReputablePolicyState` registry value was read as 1, consistent with active Smart App Control; no registry values or security settings were changed. Relevant diagnostic messages are saved alongside this note.

This is an operating-system application-control rejection, not an out-of-memory error, invalid model, or rejection by Codex automatic approval review. Training never started and no pilot results/checkpoints were created.

Microsoft documents that Smart App Control currently has no per-application exception: [Microsoft Support FAQ](https://support.microsoft.com/en-gb/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions), [Microsoft signing guidance](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/overview).

Use a PyTorch/CUDA runtime accepted by the device policy, with help from the device administrator or software publisher. Do not rename/copy blocked DLLs to evade the policy or disable security for this experiment. An ordinary administrator shell alone does not supply the missing signing approval.

## Resume once a policy-permitted runtime is available

From `C:\Users\kanaa\OneDrive\Desktop\irs`:

```powershell
.\optimize.ps1
```

If the permitted runtime is at a different path:

```powershell
.\optimize.ps1 -Python 'C:\path\to\approved-environment\Scripts\python.exe'
```

Expected new evidence will be under `05_experiments/batch32_v2_pilot/`: `OPTIMIZATION_REPORT.md`, `runs.csv`, `summary.csv`, `paired_comparisons.csv`, `epoch_history.csv`, `results_explained.json`, checkpoints, validation predictions, source snapshots, a figure, independent review and integrity manifests. These files are prospective, not claimed to exist now.

A fresh run directory is required for every repeat. The review and verification commands accept `-RunDir` and `-Python` too:

```powershell
.\optimize.ps1 -Stage review
.\optimize.ps1 -Stage verify
```

## Decision criteria fixed before execution

Complete all seeds with finite losses; peak allocated memory below 70% of GPU VRAM; mean validation AUROC reduction no greater than 0.005 and each seed reduction no greater than 0.010; median paired adaptation speedup at least 1.10x. Both settings see 14,094 training-image presentations; batch 16 has 882 optimizer updates and batch 32 has 441. No test arrays or test predictions are used. Two timing sessions are technical repetitions, not six independent training seeds. Failing a criterion means retain batch 16 and report the measured failure.
