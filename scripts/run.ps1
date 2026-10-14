param(
    [string]$PythonExecutable = 'python',
    [string]$Config = 'configs/default.yaml',
    [string[]]$Datasets = @(),
    [int]$MaxWindows = 0,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$entrypoint = Join-Path (Split-Path $PSScriptRoot -Parent) 'run.py'
$arguments = @('-B', $entrypoint, '--config', $Config)
if ($PSBoundParameters.ContainsKey('MaxWindows')) { $arguments += @('--max-windows', "$MaxWindows") }
foreach ($dataset in $Datasets) { $arguments += @('--dataset', $dataset) }
if ($Check) { $arguments += '--check' }
& $PythonExecutable @arguments
exit $LASTEXITCODE
