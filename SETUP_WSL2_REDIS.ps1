#!/usr/bin/env pwsh
# WSL2 Ubuntu + Redis Setup Script
# This sets up WSL2 Ubuntu and Redis for Mystic Trading System

Write-Host "=== WSL2 UBUNTU + REDIS SETUP ===" -ForegroundColor Cyan
Write-Host "This will configure WSL2 Ubuntu and install Redis" -ForegroundColor Yellow
Write-Host ""

# Step 1: Check WSL2 Status
Write-Host "[1/5] Checking WSL2 Status..." -ForegroundColor Cyan
$wslStatus = wsl -l -v
Write-Host $wslStatus

# Step 2: Start Ubuntu (will trigger setup if needed)
Write-Host "[2/5] Setting up Ubuntu..." -ForegroundColor Cyan
Write-Host "Ubuntu will ask for username/password - use whatever you prefer" -ForegroundColor Yellow
Write-Host "Recommended: username=lloyd, password=mystic" -ForegroundColor Green
Write-Host ""
Write-Host "Press Enter to continue to Ubuntu setup..." -ForegroundColor Yellow
Read-Host

# Launch Ubuntu setup
wsl -d Ubuntu

# Step 3: Install Redis in Ubuntu  
Write-Host "[3/5] Installing Redis in Ubuntu..." -ForegroundColor Cyan
$redisInstallCommands = @"
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo service redis-server start
redis-cli ping
"@

Write-Host "Run these commands in the Ubuntu terminal that just opened:" -ForegroundColor Yellow
Write-Host $redisInstallCommands -ForegroundColor Green
Write-Host ""
Write-Host "Press Enter after Redis installation is complete..." -ForegroundColor Yellow
Read-Host

# Step 4: Test Redis Connection
Write-Host "[4/5] Testing Redis Connection..." -ForegroundColor Cyan
$wslIp = (wsl -d Ubuntu hostname -I 2>$null).Trim()
if ($wslIp) {
    Write-Host "WSL2 IP: $wslIp" -ForegroundColor Green
    $redisPing = redis-cli -h $wslIp ping 2>$null
    if ($redisPing -eq "PONG") {
        Write-Host "✅ Redis is working at $wslIp:6379" -ForegroundColor Green
    } else {
        Write-Host "❌ Redis not responding at $wslIp" -ForegroundColor Red
        Write-Host "Try: wsl -d Ubuntu -e sudo service redis-server start" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Cannot get WSL2 IP" -ForegroundColor Red
}

# Step 5: Ready to start Mystic
Write-Host "[5/5] Setup Complete!" -ForegroundColor Green
Write-Host "Now you can run: .\START_MYSTIC.ps1" -ForegroundColor Cyan
Write-Host ""

