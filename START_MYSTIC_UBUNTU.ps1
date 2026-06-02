# MYSTIC Startup - Stops then starts app on Ubuntu VM and opens dashboard
# Copy this to your Windows Desktop and double-click to run
# Always runs stop+start on the VM so updates (e.g. code fixes) are picked up.

$UbuntuIP = "192.168.4.128"
$DashboardURL = "http://${UbuntuIP}:8000/dashboard/"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "       MYSTIC TRADING SYSTEM STARTUP        " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Stopping existing services, then starting via SSH..." -ForegroundColor Yellow
Write-Host ""

# Check if SSH is available
$sshPath = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $sshPath) {
    Write-Host "ERROR: SSH is not available on this system." -ForegroundColor Red
    Write-Host "Please install OpenSSH or start Mystic manually on the Ubuntu VM." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Manual start steps:" -ForegroundColor Cyan
    Write-Host "  1. Open Hyper-V Manager" -ForegroundColor White
    Write-Host "  2. Connect to Ubuntu VM" -ForegroundColor White
    Write-Host "  3. Run: cd /home/mystic/mystic && ./start_mystic.sh" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Start the app via SSH
Write-Host "Connecting to Ubuntu VM at $UbuntuIP..." -ForegroundColor Cyan
Write-Host "(You may be prompted for the mystic user password)" -ForegroundColor DarkGray
Write-Host ""

# Run the startup script on Ubuntu
ssh mystic@$UbuntuIP "cd /home/mystic/mystic && ./start_mystic.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "SSH connection failed. Please check:" -ForegroundColor Red
    Write-Host "  1. Ubuntu VM is running in Hyper-V" -ForegroundColor Yellow
    Write-Host "  2. SSH is enabled on Ubuntu (sudo apt install openssh-server)" -ForegroundColor Yellow
    Write-Host "  3. IP address is correct: $UbuntuIP" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Wait for app to fully start
Write-Host ""
Write-Host "Waiting for app to initialize..." -ForegroundColor Yellow
Start-Sleep 8

# Open the dashboard
Write-Host "Opening dashboard..." -ForegroundColor Green
Start-Process $DashboardURL

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "       MYSTIC STARTED SUCCESSFULLY!         " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard: $DashboardURL" -ForegroundColor Cyan
Write-Host ""
Start-Sleep 5
