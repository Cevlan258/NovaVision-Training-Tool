param([int]$WaitSeconds = 0, [string]$AppDirectory)
$ErrorActionPreference = 'Stop'
if (-not $AppDirectory) { $AppDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'App_NovaVision' }
$AppDirectory = (Resolve-Path -LiteralPath $AppDirectory).Path
$releaseRoot = Join-Path $PSScriptRoot 'release'
$manifest = Get-Content -LiteralPath (Join-Path $releaseRoot 'manifest.json') -Raw | ConvertFrom-Json
$dllPath = Join-Path $AppDirectory 'NovaVision.dll'
$depsPath = Join-Path $AppDirectory 'NovaVision.deps.json'
$helperPath = Join-Path $AppDirectory 'NovaVision.Performance.dll'
$executable = Join-Path $AppDirectory 'NovaVision.exe'

# 文件清单必须与已验证交付版本一致。
foreach ($entry in $manifest.files.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $releaseRoot $entry.Name) -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "交付文件哈希不匹配: $($entry.Name)"
    }
}
if ((Get-FileHash -LiteralPath $dllPath -Algorithm SHA256).Hash -eq $manifest.files.'NovaVision.dll' -and
    (Test-Path -LiteralPath $helperPath) -and
    (Get-FileHash -LiteralPath $helperPath -Algorithm SHA256).Hash -eq $manifest.files.'NovaVision.Performance.dll' -and
    (Get-FileHash -LiteralPath $depsPath -Algorithm SHA256).Hash -eq $manifest.files.'NovaVision.deps.json') {
    Write-Output '优化版本已安装，无需再次替换。'
    exit 0
}
if ((Get-FileHash -LiteralPath $dllPath -Algorithm SHA256).Hash -ne $manifest.original_dll_sha256 -or
    (Get-FileHash -LiteralPath $depsPath -Algorithm SHA256).Hash -ne $manifest.original_deps_sha256) {
    throw '当前程序与已验证基线不一致，拒绝覆盖。'
}
$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
while (Get-CimInstance Win32_Process -Filter "Name='NovaVision.exe'" | Where-Object { $_.ExecutablePath -eq $executable }) {
    if ([DateTime]::UtcNow -ge $deadline) { throw '请保存工作并正常关闭 NovaVision 后应用补丁；脚本不会终止应用。' }
    Start-Sleep -Milliseconds 500
}

$rollbackRoot = Join-Path $AppDirectory 'backups\performance-20261007'
$backupStatePath = Join-Path $rollbackRoot 'backup-state.json'
if (Test-Path -LiteralPath $rollbackRoot) {
    if (-not (Test-Path -LiteralPath $backupStatePath)) { throw '既有备份缺少完成状态，拒绝将部分备份视为完整备份。' }
    $backupState = Get-Content -LiteralPath $backupStatePath -Raw | ConvertFrom-Json
    if (-not $backupState.ready) { throw '此前备份未完成，请保留备份并检查；尚未改写安装文件。' }
    if ((Get-FileHash -LiteralPath (Join-Path $rollbackRoot 'NovaVision.dll') -Algorithm SHA256).Hash -ne $manifest.original_dll_sha256 -or
        (Get-FileHash -LiteralPath (Join-Path $rollbackRoot 'NovaVision.deps.json') -Algorithm SHA256).Hash -ne $manifest.original_deps_sha256) {
        throw '既有回滚备份与基线不一致，拒绝覆盖。'
    }
    if ($backupState.helper_existed -and
        (-not (Test-Path -LiteralPath (Join-Path $rollbackRoot 'NovaVision.Performance.dll')) -or
        (Get-FileHash -LiteralPath (Join-Path $rollbackRoot 'NovaVision.Performance.dll') -Algorithm SHA256).Hash -ne $backupState.helper_sha256)) {
        throw '原有辅助程序集备份缺失或损坏，拒绝覆盖。'
    }
    if (-not $backupState.helper_existed -and (Test-Path -LiteralPath (Join-Path $rollbackRoot 'NovaVision.Performance.dll'))) {
        throw '备份状态记录原先没有辅助程序集，但备份目录存在该文件，拒绝覆盖。'
    }
} else {
    New-Item -ItemType Directory -Path $rollbackRoot | Out-Null
    $helperExisted = Test-Path -LiteralPath $helperPath
    $backupState = [ordered]@{ ready=$false; helper_existed=$helperExisted; helper_sha256=if($helperExisted){(Get-FileHash -LiteralPath $helperPath -Algorithm SHA256).Hash}else{$null} }
    $backupState | ConvertTo-Json | Set-Content -LiteralPath $backupStatePath -Encoding utf8
    Copy-Item -LiteralPath $dllPath, $depsPath -Destination $rollbackRoot
    if ($helperExisted) {
        Copy-Item -LiteralPath $helperPath -Destination $rollbackRoot
        if ((Get-FileHash -LiteralPath (Join-Path $rollbackRoot 'NovaVision.Performance.dll') -Algorithm SHA256).Hash -ne $backupState.helper_sha256) {
            throw '原有辅助程序集备份哈希不匹配，尚未改写安装文件。'
        }
    }
    $backupState.ready = $true
    $backupState | ConvertTo-Json | Set-Content -LiteralPath $backupStatePath -Encoding utf8
}
try {
    foreach ($name in @('NovaVision.Performance.dll','NovaVision.deps.json','NovaVision.dll')) {
        Copy-Item -LiteralPath (Join-Path $releaseRoot $name) -Destination (Join-Path $AppDirectory $name)
        if ((Get-FileHash -LiteralPath (Join-Path $AppDirectory $name) -Algorithm SHA256).Hash -ne $manifest.files.$name) {
            throw "安装后哈希核验失败: $name"
        }
    }
    [ordered]@{ status='installed'; installed_at_utc=[DateTime]::UtcNow.ToString('o'); files=$manifest.files } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $rollbackRoot 'installation.json') -Encoding utf8
    Write-Output '优化版本已安装，重新打开 NovaVision 即可生效。'
} catch {
    $installationFailure = $_
    $rollbackFailures = [System.Collections.Generic.List[string]]::new()
    foreach ($name in @('NovaVision.deps.json','NovaVision.dll')) {
        $source = Join-Path $rollbackRoot $name
        $destination = Join-Path $AppDirectory $name
        try {
            if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
                Copy-Item -LiteralPath $source -Destination $destination
            }
        } catch { $rollbackFailures.Add("$name : $($_.Exception.Message)") }
    }
    if ($backupState.helper_existed) {
        Copy-Item -LiteralPath (Join-Path $rollbackRoot 'NovaVision.Performance.dll') -Destination $AppDirectory
    } elseif (Test-Path -LiteralPath $helperPath) { Remove-Item -LiteralPath $helperPath }
    if ($rollbackFailures.Count) { throw "安装失败且部分回滚未完成，备份保留: $($rollbackFailures -join '; ')" }
    throw $installationFailure
}
