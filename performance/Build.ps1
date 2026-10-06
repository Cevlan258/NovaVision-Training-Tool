$ErrorActionPreference = 'Stop'
$toolRoot = Split-Path -Parent $PSScriptRoot
$dotnetPath = Join-Path $PSScriptRoot 'toolchain\dotnet\dotnet.exe'
if (-not (Test-Path -LiteralPath $dotnetPath)) { throw '缺少便携 .NET 9 SDK，请按 README 准备到 performance/toolchain/dotnet。' }
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
foreach ($project in @('NovaVision.Performance','Patcher','PerformanceHarness')) {
    $destination = switch ($project) { 'NovaVision.Performance' {'helper'} 'Patcher' {'patcher'} default {'harness'} }
    & $dotnetPath build (Join-Path $PSScriptRoot ($project + '.csproj')) -c Release -o (Join-Path $PSScriptRoot ('build\' + $destination)) --nologo
    if ($LASTEXITCODE -ne 0) { throw "构建失败: $project" }
}
$original = Join-Path $PSScriptRoot 'backups\assembly-original\NovaVision.dll'
if (-not (Test-Path -LiteralPath $original)) { throw '缺少首次保存的原程序集备份，拒绝猜测补丁基线。' }
$staging = Join-Path $PSScriptRoot 'staging'
New-Item -ItemType Directory -Force -Path $staging | Out-Null
& $dotnetPath (Join-Path $PSScriptRoot 'build\patcher\Patcher.dll') patch $original (Join-Path $PSScriptRoot 'build\helper\NovaVision.Performance.dll') (Join-Path $staging 'NovaVision.dll')
if ($LASTEXITCODE -ne 0) { throw '生成补丁失败' }
& $dotnetPath (Join-Path $PSScriptRoot 'build\patcher\Patcher.dll') verify $original (Join-Path $staging 'NovaVision.dll') (Join-Path $PSScriptRoot 'reports\assembly-audit.json')
if ($LASTEXITCODE -ne 0) { throw '程序集审计失败' }
& (Join-Path $toolRoot 'python.exe') -B (Join-Path $PSScriptRoot 'PublishRelease.py')
if ($LASTEXITCODE -ne 0) { throw '生成交付文件失败' }
Write-Output '构建及程序集审计完成；交付文件在 performance/release。安装前仍须进行对应运行回归。'
