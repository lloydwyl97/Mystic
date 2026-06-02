# MYSTIC STARTUP
$repoPath = "C:\Users\lloyd\Mystic-Codebase"
Set-Location $repoPath

# CRITICAL: Force venv Python and disable shim killing for stability
$env:MYSTIC_USE_BASE_PYTHON = "false"
$env:MYSTIC_KILL_PARENT_SHIMS = "false"

$pythonExe = Join-Path $repoPath "venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: venv python not found at $pythonExe" -ForegroundColor Red
    exit 1
}

$venvPath = Join-Path $repoPath "venv"
$venvSitePackages = Join-Path $venvPath "Lib\site-packages"
$basePrefix = & $pythonExe -c "import sys; print(sys.base_prefix)" 2>$null
$basePythonExe = if ($basePrefix) { Join-Path $basePrefix "python.exe" } else { $null }
if (-not $basePythonExe -or -not (Test-Path $basePythonExe)) {
    $fallbackBase = "C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe"
    if (Test-Path $fallbackBase) {
        $basePythonExe = $fallbackBase
    }
}
$useBasePython = $env:MYSTIC_USE_BASE_PYTHON -ne "false"
$servicePythonExe = if ($useBasePython -and $basePythonExe -and (Test-Path $basePythonExe)) { $basePythonExe } else { $pythonExe }

# DEBUG: Show which Python was selected
Write-Host "`n=== PYTHON SELECTION ===" -ForegroundColor Yellow
Write-Host "MYSTIC_USE_BASE_PYTHON = '$($env:MYSTIC_USE_BASE_PYTHON)'" -ForegroundColor White
Write-Host "useBasePython = $useBasePython" -ForegroundColor White
Write-Host "Venv Python: $pythonExe" -ForegroundColor White
Write-Host "Base Python: $basePythonExe" -ForegroundColor White  
Write-Host "SELECTED: $servicePythonExe" -ForegroundColor $(if ($servicePythonExe -eq $pythonExe) { "Green" } else { "Red" })
Write-Host "========================`n" -ForegroundColor Yellow

$env:VIRTUAL_ENV = $venvPath
if (Test-Path $venvSitePackages) {
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$repoPath;$venvSitePackages;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = "$repoPath;$venvSitePackages"
    }
    $env:PYTHONNOUSERSITE = "1"
}

# CRITICAL FIX: Set UTF-8 encoding for Python to handle emojis in logging
# Without this, services crash with cp1252 encoding errors on Windows
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# HYPER-V REDIS SETUP (WSL removed - all junk deleted)
Write-Host "Setting up Hyper-V Redis connection..." -ForegroundColor Cyan

# MANUAL CONFIGURATION: Set your Hyper-V Ubuntu VM IP address here
# To find it: Open Hyper-V Manager -> Connect to VM -> Login -> Run: hostname -I
$hyperVUbuntuIP = "CONFIGURE_YOUR_HYPERV_UBUNTU_IP_HERE"  # e.g., "192.168.4.50"

if ($hyperVUbuntuIP -eq "CONFIGURE_YOUR_HYPERV_UBUNTU_IP_HERE") {
    Write-Host "ERROR: You must configure the Hyper-V Ubuntu IP in START_MYSTIC.ps1" -ForegroundColor Red
    Write-Host "Edit line 53-56 and set `$hyperVUbuntuIP to your VM's IP address" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "To find it:" -ForegroundColor Yellow
    Write-Host "  1. Open Hyper-V Manager" -ForegroundColor White
    Write-Host "  2. Connect to your Ubuntu VM" -ForegroundColor White
    Write-Host "  3. Login and run: hostname -I" -ForegroundColor White
    Write-Host "  4. Copy the IP and paste it into START_MYSTIC.ps1" -ForegroundColor White
    Write-Host "" -ForegroundColor Yellow
    Write-Host "Active IPs on network (one of these might be your VM):" -ForegroundColor Cyan
    20..100 | ForEach-Object { $ip = "192.168.4.$_"; if (Test-Connection -ComputerName $ip -Count 1 -Quiet) { Write-Host "  $ip" -ForegroundColor White } }
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Testing Redis connection at ${hyperVUbuntuIP}:6379..." -ForegroundColor Yellow
$redisTest = Test-NetConnection -ComputerName $hyperVUbuntuIP -Port 6379 -WarningAction SilentlyContinue -InformationLevel Quiet

if (-not $redisTest) {
    Write-Host "ERROR: Cannot connect to Redis at ${hyperVUbuntuIP}:6379" -ForegroundColor Red
    Write-Host "" -ForegroundColor Yellow
    Write-Host "Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "  1. Verify Ubuntu VM is running in Hyper-V Manager" -ForegroundColor White
    Write-Host "  2. Connect to VM and check Redis: sudo service redis-server status" -ForegroundColor White
    Write-Host "  3. Start Redis if needed: sudo service redis-server start" -ForegroundColor White
    Write-Host "  4. Check Redis config allows remote: sudo nano /etc/redis/redis.conf" -ForegroundColor White
    Write-Host "     - Comment out 'bind 127.0.0.1' or change to 'bind 0.0.0.0'" -ForegroundColor White
    Write-Host "     - Restart: sudo service redis-server restart" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Redis connection successful!" -ForegroundColor Green

# Test Python Redis connection
Write-Host "Testing Python Redis connection..." -ForegroundColor Yellow
try {
    $testResult = & $pythonExe -c "import redis; r = redis.Redis(host='$hyperVUbuntuIP', port=6379, socket_connect_timeout=5); r.ping(); print('OK')" 2>$null
    if ($testResult -match "OK") {
        Write-Host "✓ Python Redis connection ready!" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Python Redis connection failed, app may have issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Python Redis test failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Configure Redis environment variables for Hyper-V
$env:REDIS_HOST = $hyperVUbuntuIP
$env:REDIS_PORT = "6379"
$env:REDIS_DB   = "0"
$env:REDIS_URL  = "redis://$($hyperVUbuntuIP):6379/0"
Write-Host "✓ Redis configured at ${hyperVUbuntuIP}:6379" -ForegroundColor Green

Write-Host "Using Python executable: $servicePythonExe" -ForegroundColor Yellow

$servicePatterns = @(
    "uvicorn",
    "backend.main:app",
    "live_data_collector.py",
    "start_signals.py",
    "start_orchestrator.py",
    "start_portfolio_engine_integration.py",
    "start_paper_autobuy.py",
    "start_agent_orchestrator.py",
    "start_ai_live_trading.py"
)

function Stop-MysticServices {
    Write-Host "STOPPING EXISTING MYSTIC SERVICES..." -ForegroundColor Red
    
    # Kill all Python processes 3 times to ensure they die
    Write-Host "Kill pass 1/3..." -ForegroundColor Yellow
    cmd /c "taskkill /F /IM python.exe /T 2>nul"
    cmd /c "taskkill /F /IM pythonw.exe /T 2>nul"
    Start-Sleep 2
    
    Write-Host "Kill pass 2/3..." -ForegroundColor Yellow
    cmd /c "taskkill /F /IM python.exe /T 2>nul"
    cmd /c "taskkill /F /IM pythonw.exe /T 2>nul"
    Start-Sleep 2
    
    Write-Host "Kill pass 3/3..." -ForegroundColor Yellow
    cmd /c "taskkill /F /IM python.exe /T 2>nul"
    cmd /c "taskkill /F /IM pythonw.exe /T 2>nul"
    Start-Sleep 3

    # Verify processes are actually dead
    Write-Host "Verifying processes stopped..." -ForegroundColor Yellow
    $remaining = Get-Process python -ErrorAction SilentlyContinue
    if ($remaining) {
        Write-Host "WARNING: $($remaining.Count) Python processes still running! Force killing..." -ForegroundColor Red
        Stop-Process -Name python -Force -ErrorAction SilentlyContinue
        Stop-Process -Name pythonw -Force -ErrorAction SilentlyContinue
        Start-Sleep 3
    }

    # Clean up orphaned/zombie shim processes
    Write-Host "Cleaning up orphaned shim processes..." -ForegroundColor Yellow
    $deadProcesses = Get-Process | Where-Object { 
        $_.ProcessName -eq "python" -and $_.Handles -eq 0 
    } -ErrorAction SilentlyContinue
    if ($deadProcesses) {
        Write-Host "Found $($deadProcesses.Count) zombie processes, cleaning up..." -ForegroundColor Yellow
        $deadProcesses | ForEach-Object {
            try {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                Write-Host "Cleaned up zombie process PID $($_.Id)" -ForegroundColor DarkYellow
            } catch {
                # Process already gone
            }
        }
    }

    Write-Host "STOP COMPLETE - All processes terminated" -ForegroundColor Green
    
    # MANDATORY DATA CLEANUP - RUNS EVERY START
    Write-Host "`nRUNNING MANDATORY DATA CLEANUP..." -ForegroundColor Cyan
    & $pythonExe "MANDATORY_CLEANUP.py"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nERROR: Data cleanup failed! Cannot start services with malformed data." -ForegroundColor Red
        Write-Host "Please check MANDATORY_CLEANUP.py output above." -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
    
    Write-Host "DATA CLEANUP COMPLETE - Safe to start services`n" -ForegroundColor Green
}

function Start-MysticService {
    param(
        [string]$Name,
        [string]$Arguments,
        [int]$DelaySeconds = 2
    )

    Write-Host "STARTING: $Name" -ForegroundColor Cyan
    
    # Start service directly - no cmd wrapper, no shim killing
    $proc = Start-Process -FilePath $servicePythonExe -ArgumentList $Arguments -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru
    
    if ($proc) {
        Write-Host "  Started PID $($proc.Id)" -ForegroundColor DarkGreen
    }
    
    Start-Sleep $DelaySeconds
}

Stop-MysticServices

Write-Host "STARTING: Backend API (lifespan owns system services)..." -ForegroundColor Cyan

$env:MYSTIC_USE_BASE_PYTHON = "false"
$env:MYSTIC_KILL_PARENT_SHIMS = "false"

$env:EXTERNAL_SUPERVISOR_MODE = "false"
$env:MYSTIC_LIFESPAN_AUTOSTART = "true"

Start-Process -FilePath $pythonExe -ArgumentList "-m uvicorn backend.main:app --host 0.0.0.0 --port 8000" -WorkingDirectory $repoPath -WindowStyle Normal -PassThru | Out-Null

Write-Host "Waiting for API to come up..." -ForegroundColor Yellow
Start-Sleep 12

Write-Host "STARTING: Data Collector (standalone)..." -ForegroundColor Cyan
Start-Process -FilePath $pythonExe -ArgumentList "live_data_collector.py" -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru | Out-Null

Write-Host "STARTING: Heavy AI (standalone, lifespan disabled)..." -ForegroundColor Cyan
$env:EXTERNAL_SUPERVISOR_MODE = "true"
$env:MYSTIC_LIFESPAN_AUTOSTART = "false"

Start-Process -FilePath $pythonExe -ArgumentList "start_ai_live_trading.py" -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru | Out-Null

Write-Host "`nSTARTUP COMPLETE" -ForegroundColor Green
Write-Host "Dashboard: http://localhost:8000" -ForegroundColor Green
Read-Host "Press Enter to close this window"
