param([string]$AppDirectory)
$ErrorActionPreference = 'Stop'
if (-not $AppDirectory) { $AppDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'App_NovaVision' }
$AppDirectory = (Resolve-Path -LiteralPath $AppDirectory).Path
$executable = Join-Path $AppDirectory 'NovaVision.exe'
if (Get-CimInstance Win32_Process -Filter "Name='NovaVision.exe'" | Where-Object { $_.ExecutablePath -eq $executable }) {
    throw '请先保存工作并正常关闭 NovaVision；脚本不会终止应用。'
}
$backupRoot = Join-Path $AppDirectory 'backups\performance-20261007'
$backupStatePath = Join-Path $backupRoot 'backup-state.json'
if (-not (Test-Path -LiteralPath $backupStatePath)) { throw '备份完成状态缺失，拒绝恢复。' }
$backupState = Get-Content -LiteralPath $backupStatePath -Raw | ConvertFrom-Json
if (-not $backupState.ready) { throw '备份未完成，拒绝恢复。' }
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release\manifest.json') -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot ('release\' + $entry.Name)) -Algorithm SHA256).Hash -ne $entry.Value -or
        (Get-FileHash -LiteralPath (Join-Path $AppDirectory $entry.Name) -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "当前程序或交付文件与本次优化不一致，拒绝恢复: $($entry.Name)"
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $backupRoot 'NovaVision.dll') -Algorithm SHA256).Hash -ne $manifest.original_dll_sha256 -or
    (Get-FileHash -LiteralPath (Join-Path $backupRoot 'NovaVision.deps.json') -Algorithm SHA256).Hash -ne $manifest.original_deps_sha256) {
    throw '回滚备份哈希不匹配，尚未改写安装文件。'
}
if ($backupState.helper_existed -and
    (-not (Test-Path -LiteralPath (Join-Path $backupRoot 'NovaVision.Performance.dll')) -or
    (Get-FileHash -LiteralPath (Join-Path $backupRoot 'NovaVision.Performance.dll') -Algorithm SHA256).Hash -ne $backupState.helper_sha256)) {
    throw '原有辅助程序集备份缺失或损坏，尚未改写安装文件。'
}
if (-not $backupState.helper_existed -and (Test-Path -LiteralPath (Join-Path $backupRoot 'NovaVision.Performance.dll'))) {
    throw '备份状态记录原先没有辅助程序集，但备份目录存在该文件，拒绝恢复。'
}
# 仅恢复本次安装的程序集和依赖清单，不回滚用户数据和其他项目改动。
$helperPath = Join-Path $AppDirectory 'NovaVision.Performance.dll'
try {
    Copy-Item -LiteralPath (Join-Path $backupRoot 'NovaVision.deps.json') -Destination $AppDirectory
    Copy-Item -LiteralPath (Join-Path $backupRoot 'NovaVision.dll') -Destination $AppDirectory
    if ($backupState.helper_existed) {
        Copy-Item -LiteralPath (Join-Path $backupRoot 'NovaVision.Performance.dll') -Destination $AppDirectory
    } elseif (Test-Path -LiteralPath $helperPath) { Remove-Item -LiteralPath $helperPath }
    if ((Get-FileHash -LiteralPath (Join-Path $AppDirectory 'NovaVision.dll') -Algorithm SHA256).Hash -ne $manifest.original_dll_sha256 -or
        (Get-FileHash -LiteralPath (Join-Path $AppDirectory 'NovaVision.deps.json') -Algorithm SHA256).Hash -ne $manifest.original_deps_sha256) {
        throw '回滚后哈希不匹配'
    }
} catch {
    $restorationFailure = $_
    $recoveryFailures = [System.Collections.Generic.List[string]]::new()
    foreach ($name in @('NovaVision.Performance.dll','NovaVision.deps.json','NovaVision.dll')) {
        $source = Join-Path $PSScriptRoot ('release\' + $name)
        $destination = Join-Path $AppDirectory $name
        try {
            if (-not (Test-Path -LiteralPath $destination) -or
                (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
                Copy-Item -LiteralPath $source -Destination $destination
            }
        } catch { $recoveryFailures.Add("$name : $($_.Exception.Message)") }
    }
    if ($recoveryFailures.Count) { throw "恢复失败且部分安装文件未还原，备份保留: $($recoveryFailures -join '; ')" }
    throw $restorationFailure
}
Write-Output '已恢复本次优化前的程序版本。Python桥接保持兼容精简版本。'
