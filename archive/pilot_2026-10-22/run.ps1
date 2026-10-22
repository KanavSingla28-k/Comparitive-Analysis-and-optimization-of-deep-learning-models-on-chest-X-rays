param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ExperimentArgs)
$ErrorActionPreference = 'Stop'
$researchRoot = $PSScriptRoot
$researchPython = 'C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe'
Set-Location -LiteralPath $researchRoot
& $researchPython -u (Join-Path $researchRoot 'src\experiment.py') @ExperimentArgs
exit $LASTEXITCODE
