param([Parameter(Position=0)][ValidateSet('all','review','verify-package')][string]$Stage='all',[string]$Python='')
$ErrorActionPreference='Stop'
$researchRoot=$PSScriptRoot
if (-not $Python) {
    $localPython=Join-Path $researchRoot '.venv\Scripts\python.exe'
    $sharedPython='C:\Users\kanaa\Documents\Codex\ml-gpu-env\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python=$localPython }
    elseif (Test-Path -LiteralPath $sharedPython) { $Python=$sharedPython }
    else { $Python='python' }
}
Push-Location -LiteralPath $researchRoot
try {
    & $Python -B -u (Join-Path $researchRoot '04_implementation\src\finalize_project.py') $Stage
    if ($LASTEXITCODE -ne 0) { throw "Finalizer failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }