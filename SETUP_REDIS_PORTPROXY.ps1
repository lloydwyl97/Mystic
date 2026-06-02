# ==============================================================================
# SETUP_REDIS_PORTPROXY.ps1
# Run this script AS ADMINISTRATOR to enable Windows → WSL2 Redis access
# ==============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "REDIS PORTPROXY SETUP FOR WSL2" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "✓ Running as Administrator`n" -ForegroundColor Green

# Get WSL2 IP
Write-Host "Detecting WSL2 IP address..." -ForegroundColor Yellow
$wslIp = (wsl -d Ubuntu hostname -I).Trim().Split()[0]

if (-not $wslIp) {
    Write-Host "ERROR: Could not detect WSL2 IP address!" -ForegroundColor Red
    Write-Host "Make sure Ubuntu WSL2 is running." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ WSL2 IP: $wslIp`n" -ForegroundColor Green

# Remove old portproxy entries
Write-Host "Removing old portproxy entries (if any)..." -ForegroundColor Yellow
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=6379 2>$null
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=6379 2>$null
Write-Host "✓ Old entries cleared`n" -ForegroundColor Green

# Create new portproxy
Write-Host "Creating portproxy: 127.0.0.1:6379 -> ${wslIp}:6379..." -ForegroundColor Yellow
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=6379 connectaddress=$wslIp connectport=6379

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Portproxy created successfully`n" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to create portproxy!" -ForegroundColor Red
    exit 1
}

# Verify portproxy
Write-Host "Current portproxy rules:" -ForegroundColor Yellow
netsh interface portproxy show all
Write-Host ""

# Add firewall rule
Write-Host "Adding firewall rule for Redis portproxy..." -ForegroundColor Yellow
New-NetFirewallRule -DisplayName "WSL2 Redis PortProxy 6379 Inbound" -Direction Inbound -LocalAddress 127.0.0.1 -LocalPort 6379 -Protocol TCP -Action Allow -Profile Any -ErrorAction SilentlyContinue

if ($?) {
    Write-Host "✓ Firewall rule added`n" -ForegroundColor Green
} else {
    Write-Host "⚠ Firewall rule may already exist (this is OK)`n" -ForegroundColor Yellow
}

# Test connection
Write-Host "Testing connection to localhost:6379..." -ForegroundColor Yellow
$testResult = Test-NetConnection -ComputerName 127.0.0.1 -Port 6379 -InformationLevel Quiet

if ($testResult) {
    Write-Host "✓ Connection successful!`n" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "PORTPROXY SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    Write-Host "Redis is now accessible at: redis://127.0.0.1:6379/0" -ForegroundColor Cyan
} else {
    Write-Host "⚠ Connection test failed" -ForegroundColor Yellow
    Write-Host "Redis may still be starting up, or there's a firewall issue." -ForegroundColor Yellow
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

