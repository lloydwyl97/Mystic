# MYSTIC MIGRATION PACKAGE CREATOR
# Creates a clean package for transfer to new Hyper-V Windows Server
# Run this on your CURRENT machine to prepare for migration

$sourcePath = "C:\Users\lloyd\Mystic-Codebase"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$packageName = "Mystic_Migration_$timestamp"
$outputPath = "C:\Users\lloyd\Desktop\$packageName"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MYSTIC MIGRATION PACKAGE CREATOR" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create output directory
if (Test-Path $outputPath) {
    Remove-Item $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

Write-Host "Creating migration package at: $outputPath" -ForegroundColor Yellow
Write-Host ""

# Define what to EXCLUDE (don't copy these)
$excludeDirs = @(
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "*.egg-info",
    "logs",           # Can start fresh
    "audit_logs",     # Can start fresh
    "var"             # Runtime data
)

$excludeFiles = @(
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "backend_api_output.txt"
)

# CRITICAL FILES TO COPY
Write-Host "[1/6] Copying core codebase..." -ForegroundColor Green

# Use robocopy for efficient copying with exclusions
$excludeDirArgs = $excludeDirs | ForEach-Object { "/XD", $_ }
$excludeFileArgs = $excludeFiles | ForEach-Object { "/XF", $_ }

$robocopyArgs = @(
    $sourcePath,
    $outputPath,
    "/E",           # Copy subdirectories including empty ones
    "/NP",          # No progress percentage
    "/NFL",         # No file list
    "/NDL",         # No directory list
    "/MT:8"         # Multi-threaded
) + $excludeDirArgs + $excludeFileArgs

& robocopy @robocopyArgs | Out-Null

Write-Host "  Core files copied" -ForegroundColor DarkGreen

# Copy .env file explicitly (might be hidden)
Write-Host "[2/6] Copying .env configuration..." -ForegroundColor Green
$envFile = Join-Path $sourcePath ".env"
if (Test-Path $envFile) {
    Copy-Item $envFile -Destination $outputPath -Force
    Write-Host "  .env file copied" -ForegroundColor DarkGreen
} else {
    Write-Host "  WARNING: No .env file found - you'll need to create one!" -ForegroundColor Yellow
}

# Copy models folder (AI trained models - IMPORTANT)
Write-Host "[3/6] Copying trained AI models..." -ForegroundColor Green
$modelsPath = Join-Path $sourcePath "models"
$modelsOutput = Join-Path $outputPath "models"
if (Test-Path $modelsPath) {
    # Copy models excluding cache files
    robocopy $modelsPath $modelsOutput /E /NP /NFL /NDL /XF *.tmp *.log | Out-Null
    $modelCount = (Get-ChildItem $modelsOutput -Recurse -File).Count
    Write-Host "  $modelCount model files copied" -ForegroundColor DarkGreen
}

# Copy database if exists
Write-Host "[4/6] Copying database..." -ForegroundColor Green
$dbFile = Join-Path $sourcePath "mystic_trading.db"
if (Test-Path $dbFile) {
    Copy-Item $dbFile -Destination $outputPath -Force
    $dbSize = [math]::Round((Get-Item $dbFile).Length / 1MB, 2)
    Write-Host "  Database copied (${dbSize} MB)" -ForegroundColor DarkGreen
} else {
    Write-Host "  No database file (will be created fresh)" -ForegroundColor DarkYellow
}

# Create setup script for NEW server
Write-Host "[5/6] Creating setup script for new server..." -ForegroundColor Green

$setupScript = @'
# MYSTIC SETUP ON NEW SERVER
# Run this AFTER copying files to the new Hyper-V Windows Server

$repoPath = $PSScriptRoot  # Assumes script is in the Mystic folder

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MYSTIC NEW SERVER SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $repoPath "venv"
if (Test-Path $venvPath) {
    Write-Host "  venv already exists, skipping..." -ForegroundColor DarkYellow
} else {
    python -m venv $venvPath
    Write-Host "  venv created" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
$pipExe = Join-Path $venvPath "Scripts\pip.exe"
$reqFile = Join-Path $repoPath "requirements.linux.txt"

if (Test-Path $reqFile) {
    & $pipExe install -r $reqFile --quiet
    Write-Host "  Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  WARNING: requirements.linux.txt not found" -ForegroundColor Yellow
}

# Setup OpenSSH Server for Cursor remote editing
Write-Host "[4/5] Setting up SSH for Cursor remote editing..." -ForegroundColor Yellow
$sshCapability = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
if ($sshCapability.State -ne 'Installed') {
    Write-Host "  Installing OpenSSH Server..." -ForegroundColor DarkYellow
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
}
Start-Service sshd -ErrorAction SilentlyContinue
Set-Service -Name sshd -StartupType Automatic -ErrorAction SilentlyContinue
Write-Host "  SSH Server enabled" -ForegroundColor Green

# Configure firewall
Write-Host "[5/5] Configuring firewall..." -ForegroundColor Yellow
$rules = @(
    @{Name="Mystic Backend API"; Port=8000},
    @{Name="Mystic Redis"; Port=6379},
    @{Name="OpenSSH Server"; Port=22}
)

foreach ($rule in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if (-not $existing) {
        New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Port $rule.Port -Protocol TCP -Action Allow | Out-Null
        Write-Host "  Firewall rule added: $($rule.Name) (port $($rule.Port))" -ForegroundColor DarkGreen
    } else {
        Write-Host "  Firewall rule exists: $($rule.Name)" -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Update START_MYSTIC.ps1 line 2:" -ForegroundColor White
Write-Host "   `$repoPath = `"$repoPath`"" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Update Redis IP in START_MYSTIC.ps1 line 57:" -ForegroundColor White
Write-Host "   For local Redis: `$hyperVUbuntuIP = `"127.0.0.1`"" -ForegroundColor Yellow
Write-Host "   Or your Redis server IP" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. To connect Cursor from your other PC:" -ForegroundColor White
$serverIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress
Write-Host "   - Open Cursor" -ForegroundColor Yellow
Write-Host "   - Press Ctrl+Shift+P" -ForegroundColor Yellow
Write-Host "   - Type: Remote-SSH: Connect to Host" -ForegroundColor Yellow
Write-Host "   - Enter: $env:USERNAME@$serverIP" -ForegroundColor Yellow
Write-Host "   - Open folder: $repoPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Start Mystic:" -ForegroundColor White
Write-Host "   .\START_MYSTIC.ps1" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
'@

$setupScriptPath = Join-Path $outputPath "SETUP_NEW_SERVER.ps1"
$setupScript | Out-File -FilePath $setupScriptPath -Encoding UTF8
Write-Host "  Setup script created" -ForegroundColor DarkGreen

# Create summary
Write-Host "[6/6] Creating migration summary..." -ForegroundColor Green

$totalSize = [math]::Round((Get-ChildItem $outputPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 2)
$fileCount = (Get-ChildItem $outputPath -Recurse -File).Count

$summary = @"
MYSTIC MIGRATION PACKAGE
========================
Created: $timestamp
Total Size: $totalSize MB
Total Files: $fileCount

CONTENTS:
- Core codebase (backend, configs, scripts)
- Trained AI models
- Database (if exists)
- Environment configuration

TO TRANSFER TO NEW SERVER:
1. Copy this entire folder to the new Hyper-V Windows Server
2. Run SETUP_NEW_SERVER.ps1 as Administrator
3. Follow the on-screen instructions

CURSOR REMOTE EDITING:
After setup, connect from your local Cursor:
- Ctrl+Shift+P -> Remote-SSH: Connect to Host
- Enter: username@server-ip
- Open the Mystic folder

"@

$summaryPath = Join-Path $outputPath "README.txt"
$summary | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  MIGRATION PACKAGE READY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Location: $outputPath" -ForegroundColor Cyan
Write-Host "Size: $totalSize MB" -ForegroundColor Cyan
Write-Host "Files: $fileCount" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT: Copy the folder to your new server" -ForegroundColor Yellow
Write-Host "      Then run SETUP_NEW_SERVER.ps1" -ForegroundColor Yellow
Write-Host ""

# Open the folder
explorer $outputPath

Read-Host "Press Enter to close"





