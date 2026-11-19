param(
    [ValidateSet('run','verify','review')][string]$Stage = 'run',
    [string]$RunDir = '05_experiments/batch32_v2_pilot',
    [string]$Python = 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    $runner = '04_implementation/src/batch32_pilot.py'
    $reviewer = '04_implementation/src/review_batch32_pilot.py'
    if ($Stage -eq 'verify') {
        & $Python -B $runner --verify --run-dir $RunDir
        if ($LASTEXITCODE -ne 0) { throw 'Pilot artifact verification failed.' }
    } elseif ($Stage -eq 'review') {
        & $Python -B $reviewer --run-dir $RunDir
        if ($LASTEXITCODE -ne 0) { throw 'Independent pilot review failed.' }
    } else {
        # Respect Windows application policy. Import failure stops before training.
        & $Python -B -c "import sys; sys.path.insert(0, '04_implementation/src'); import runtime; import numpy; import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch preflight failed. See 06_results/OPTIMIZATION_EXECUTION_STATUS.md. No security settings were changed.' }
        $log = '05_experiments/logs/batch32_optimization_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log'
        & $Python -B -u $runner --run-dir $RunDir 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "Pilot execution failed. See $log" }
        & $Python -B $reviewer --run-dir $RunDir
        if ($LASTEXITCODE -ne 0) { throw 'Independent pilot review failed.' }
        & $Python -B $runner --verify --run-dir $RunDir
        if ($LASTEXITCODE -ne 0) { throw 'Pilot artifact verification failed.' }
    }
} finally { Pop-Location }

