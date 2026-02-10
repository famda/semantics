# Semantics CLI installer for Windows
# Usage: irm https://raw.githubusercontent.com/famda/semantics/main/docs/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo       = "famda/semantics"
$Image      = "famda/semantics:cli-latest"
$InstallDir = "$env:LOCALAPPDATA\semantics"
$BinaryName = "semantics.exe"
$Rid        = "win-x64"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Info  { param($msg) Write-Host "==> " -ForegroundColor Green -NoNewline; Write-Host $msg }
function Write-Err   { param($msg) Write-Host "error: " -ForegroundColor Red -NoNewline; Write-Host $msg; exit 1 }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  +=============================================+" -ForegroundColor Cyan
Write-Host "  |         Semantics CLI Installer             |" -ForegroundColor Cyan
Write-Host "  +=============================================+" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker is not installed.`n`n  Install Docker Desktop: https://docs.docker.com/get-docker/`n`n  Then re-run this installer."
}

$prevEAP = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
docker info *>$null
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    Write-Err "Docker daemon is not running.`n`n  Please start Docker Desktop and re-run this installer."
}

# ---------------------------------------------------------------------------
# Download gateway binary
# ---------------------------------------------------------------------------
Write-Info "Downloading Semantics CLI ..."

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$ReleaseUrl  = "https://api.github.com/repos/$Repo/releases/latest"

# Try local binary first (only when running from a repo checkout, not via irm | iex)
$LocalBinary = if ($PSScriptRoot) { Join-Path $PSScriptRoot $BinaryName } else { $null }
if ($LocalBinary -and (Test-Path $LocalBinary)) {
    Copy-Item $LocalBinary (Join-Path $InstallDir $BinaryName) -Force
    Write-Info "Installed from local build."
}
else {
    try {
        $release = Invoke-RestMethod -Uri $ReleaseUrl -UseBasicParsing
        $asset   = $release.assets | Where-Object { $_.name -eq "semantics-$Rid.exe" } | Select-Object -First 1
        if (-not $asset) {
            Write-Err "Could not find semantics-$Rid.exe in latest release."
        }
        $downloadUrl = $asset.browser_download_url

        $frames = [string[]]@([char]0x280B,[char]0x2819,[char]0x2839,[char]0x2838,[char]0x283C,[char]0x2834,[char]0x2826,[char]0x2827,[char]0x2807,[char]0x280F)
        $idx = 0
        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        $targetPath = Join-Path $InstallDir $BinaryName
        $webClient = [System.Net.WebClient]::new()
        $downloadTask = $webClient.DownloadFileTaskAsync($downloadUrl, $targetPath)

        while (-not $downloadTask.IsCompleted) {
            $elapsed = [math]::Floor($sw.Elapsed.TotalSeconds)
            Write-Host "`r  $($frames[$idx % $frames.Count]) Downloading binary ... (${elapsed}s)  " -NoNewline
            $idx++
            Start-Sleep -Milliseconds 80
        }
        $sw.Stop()
        Write-Host "`r$(' ' * 60)`r" -NoNewline

        if ($downloadTask.IsFaulted) {
            Write-Err "Download failed: $($downloadTask.Exception.InnerException.Message)"
        }

        $webClient.Dispose()
    }
    catch {
        Write-Err "Failed to download release binary: $_"
    }
}

# ---------------------------------------------------------------------------
# Pull Docker image
# ---------------------------------------------------------------------------
Write-Info "Pulling container image ..."

$psi = [System.Diagnostics.ProcessStartInfo]@{
    FileName               = "docker"
    Arguments              = "pull $Image"
    UseShellExecute        = $false
    RedirectStandardOutput = $true
    RedirectStandardError  = $true
    CreateNoWindow         = $true
}
$proc = [System.Diagnostics.Process]::Start($psi)
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()

$frames = [string[]]@([char]0x280B,[char]0x2819,[char]0x2839,[char]0x2838,[char]0x283C,[char]0x2834,[char]0x2826,[char]0x2827,[char]0x2807,[char]0x280F)
$idx = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()
while (-not $proc.HasExited) {
    $elapsed = [math]::Floor($sw.Elapsed.TotalSeconds)
    Write-Host "`r  $($frames[$idx % $frames.Count]) Pulling image ... (${elapsed}s)  " -NoNewline
    $idx++
    Start-Sleep -Milliseconds 80
}
$sw.Stop()
Write-Host "`r$(' ' * 60)`r" -NoNewline

$proc.WaitForExit()
[void]$stdoutTask.Wait()
[void]$stderrTask.Wait()

if ($proc.ExitCode -ne 0) {
    Write-Host ""
    Write-Host $stderrTask.Result -ForegroundColor Red
    Write-Err "Failed to pull $Image."
}
$proc.Dispose()

# Create CLI-specific shims: semantics-audio, semantics-video, semantics-research
foreach ($cli in @("audio", "video", "research")) {
    $shimContent = @"
@echo off
"%~dp0$BinaryName" $cli %*
"@
    Set-Content -Path (Join-Path $InstallDir "semantics-$cli.cmd") -Value $shimContent -Encoding ASCII
}

Write-Info "Installed: semantics, semantics-audio, semantics-video, semantics-research"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
& (Join-Path $InstallDir $BinaryName) version

# ---------------------------------------------------------------------------
# Add to PATH
# ---------------------------------------------------------------------------
if ($env:GITHUB_ACTIONS) {
    Write-Info "Adding to GITHUB_PATH for this workflow ..."
    $InstallDir | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
    $env:Path = "$InstallDir;$env:Path"
}
else {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$InstallDir*") {
        Write-Info "Adding $InstallDir to PATH ..."
        [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
        $env:Path = "$env:Path;$InstallDir"
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Available commands:"
Write-Host "    semantics audio     " -ForegroundColor White -NoNewline; Write-Host "- Audio processing"
Write-Host "    semantics video     " -ForegroundColor White -NoNewline; Write-Host "- Video analysis"
Write-Host "    semantics research  " -ForegroundColor White -NoNewline; Write-Host "- Web research"
Write-Host "    semantics update    " -ForegroundColor White -NoNewline; Write-Host "- Update to latest version"
Write-Host ""
Write-Host "  Restart your terminal" -ForegroundColor Yellow -NoNewline; Write-Host " or run:"
Write-Host "    `$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User')"
Write-Host ""
