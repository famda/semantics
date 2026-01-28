# Reset Platform CLI development environment
# Run this from the Platform directory on the host

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PlatformDir = Split-Path -Parent $ScriptDir

Write-Host "=== Platform CLI Dev Environment Reset ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will stop dev containers and optionally clear cached data."
Write-Host ""

Set-Location $PlatformDir

# Stop containers if running
Write-Host "Stopping dev containers..." -ForegroundColor Yellow
try { docker compose down 2>$null } catch {}

Write-Host ""
$response = Read-Host "Also clear workspaces data (.data/platform/workspaces)? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Clearing workspaces..."
    Remove-Item -Recurse -Force ".data\platform\workspaces\results\*" -ErrorAction SilentlyContinue
    Write-Host "Workspaces cleared (assets preserved)" -ForegroundColor Green
}

Write-Host ""
$response = Read-Host "Rebuild images from scratch? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Rebuilding images (this may take a while)..."
    docker compose build --no-cache
    Write-Host "Images rebuilt" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done! You can now reopen a devcontainer." -ForegroundColor Green
