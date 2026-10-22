param(
    [Parameter(Position=0)][string]$Stage = 'verify',
    [string]$RunDir = '05_experiments/reproduced_v1',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$researchRoot = $PSScriptRoot
if (-not $Python) {
    $localPython = Join-Path $researchRoot '.venv\Scripts\python.exe'
    $sharedPython = 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python = $localPython }
    elseif (Test-Path -LiteralPath $sharedPython) { $Python = $sharedPython }
    else { $Python = 'python' }
}
$source = Join-Path $researchRoot '04_implementation\src\pipeline.py'
Push-Location -LiteralPath $researchRoot
try {
    & $Python -B -u $source $Stage --run-dir $RunDir
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }