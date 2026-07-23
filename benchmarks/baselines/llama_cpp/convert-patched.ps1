[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelDir,
    [Parameter(Mandatory = $true, Position = 1)][string]$OutputGguf,
    [Parameter(Position = 2, ValueFromRemainingArguments = $true)][string[]]$ConverterOptions = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$expectedCommit = (Get-Content -Raw (Join-Path $PSScriptRoot "commit.txt")).Trim()
$patchedSource = if ($env:LLAMA_CPP_PATCHED_SOURCE) { $env:LLAMA_CPP_PATCHED_SOURCE } else { Join-Path $repoRoot "third_party\cache\llama.cpp-mixed" }
$python = if ($env:LLAMA_CPP_CONVERT_PYTHON) { $env:LLAMA_CPP_CONVERT_PYTHON } else { Join-Path $repoRoot "third_party\cache\unsloth-nvfp4-env\Scripts\python.exe" }
$patchFile = Join-Path $PSScriptRoot "patches\0001-support-mixed-fp8-nvfp4-compressed-tensors.patch"

if (-not (Test-Path -LiteralPath (Join-Path $patchedSource ".git"))) {
    throw "Patched converter source was not found; run prepare-patched-source.ps1"
}
$actualCommit = (& git.exe -C $patchedSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Patched source is not based on $expectedCommit"
}
& git.exe -C $patchedSource apply --unidiff-zero --reverse --check $patchFile
if ($LASTEXITCODE -ne 0) {
    throw "Tracked mixed-precision patch is not applied exactly"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Converter Python was not found at $python"
}

$previousSource = $env:LLAMA_CPP_SOURCE
$previousPython = $env:LLAMA_CPP_CONVERT_PYTHON
try {
    $env:LLAMA_CPP_SOURCE = $patchedSource
    $env:LLAMA_CPP_CONVERT_PYTHON = $python
    & (Join-Path $PSScriptRoot "convert.ps1") $ModelDir $OutputGguf @ConverterOptions
} finally {
    $env:LLAMA_CPP_SOURCE = $previousSource
    $env:LLAMA_CPP_CONVERT_PYTHON = $previousPython
}
