# ============================================================================
# MYSTIC CLEAN START - "NUKE FROM ORBIT"
# ============================================================================
# This script performs a COMPLETE clean start:
# 1. Stops all Mystic Python processes
# 2. Stops and disables old Windows Redis 3.0.504
# 3. Deletes ALL database files (SQLite)
# 4. Flushes Redis completely
# 5. Clears ALL logs
# 6. Verifies clean state
# ============================================================================

param(
    [switch]$Force,
    [switch]$SkipRedisSwitch
)

$ErrorActionPreference = "Continue"
$repoPath = "C:\Users\lloyd\Mystic-Codebase"
Set-Location $repoPath

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Red
Write-Host "  MYSTIC CLEAN START - NUKE FROM ORBIT" -ForegroundColor Red
Write-Host "  This will DELETE all databases, flush Redis, and clear logs!" -ForegroundColor Red
Write-Host "=" * 70 -ForegroundColor Red
Write-Host ""

if (-not $Force) {
    $confirm = Read-Host "Type 'NUKE' to confirm complete data wipe"
    if ($confirm -ne "NUKE") {
        Write-Host "Aborted. No changes made." -ForegroundColor Yellow
        exit 0
    }
}

# ============================================================================
# PHASE 1: STOP ALL PYTHON PROCESSES
# ============================================================================
Write-Host ""
Write-Host "[1/6] STOPPING ALL PYTHON PROCESSES..." -ForegroundColor Cyan

# Kill all Python processes aggressively
for ($i = 1; $i -le 3; $i++) {
    Write-Host "  Kill pass $i/3..." -ForegroundColor Yellow
    $null = taskkill /F /IM python.exe /T 2>&1
    $null = taskkill /F /IM pythonw.exe /T 2>&1
    Start-Sleep -Seconds 2
}

# Verify
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "  Force killing remaining $($remaining.Count) processes..." -ForegroundColor Red
    Stop-Process -Name python -Force -ErrorAction SilentlyContinue
    Stop-Process -Name pythonw -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "  [OK] All Python processes stopped" -ForegroundColor Green

# ============================================================================
# PHASE 2: SWITCH REDIS (Stop Windows Redis, Use WSL2 Redis)
# ============================================================================
if (-not $SkipRedisSwitch) {
    Write-Host ""
    Write-Host "[2/6] SWITCHING TO WSL2 REDIS..." -ForegroundColor Cyan
    
    # Stop Windows Redis service
    $redisService = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($redisService) {
        if ($redisService.Status -eq "Running") {
            Write-Host "  Stopping Windows Redis service..." -ForegroundColor Yellow
            Stop-Service -Name "Redis" -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
        Write-Host "  Disabling Windows Redis service..." -ForegroundColor Yellow
        Set-Service -Name "Redis" -StartupType Disabled -ErrorAction SilentlyContinue
        Write-Host "  [OK] Windows Redis stopped and disabled" -ForegroundColor Green
    } else {
        Write-Host "  Windows Redis service not found (already removed?)" -ForegroundColor Yellow
    }
    
    # Also stop redis-mystic if it exists
    $redisMystic = Get-Service -Name "redis-mystic" -ErrorAction SilentlyContinue
    if ($redisMystic -and $redisMystic.Status -eq "Running") {
        Stop-Service -Name "redis-mystic" -Force -ErrorAction SilentlyContinue
        Set-Service -Name "redis-mystic" -StartupType Disabled -ErrorAction SilentlyContinue
        Write-Host "  [OK] redis-mystic service stopped and disabled" -ForegroundColor Green
    }
    
    # Start WSL2 Redis
    Write-Host "  Starting WSL2 Redis..." -ForegroundColor Yellow
    wsl -d Ubuntu -e sudo service redis-server start 2>$null
    Start-Sleep -Seconds 2
    
    # Get WSL2 IP and verify Redis is running
    $wslIp = (wsl -d Ubuntu hostname -I 2>$null).Trim()
    if ($wslIp) {
        $pong = redis-cli -h $wslIp ping 2>$null
        if ($pong -eq "PONG") {
            Write-Host "  [OK] WSL2 Redis 7.0.15 responding at $wslIp" -ForegroundColor Green
            $script:wslRedisIp = $wslIp
        } else {
            Write-Host "  [WARNING] Redis not responding at $wslIp" -ForegroundColor Red
        }
    } else {
        Write-Host "  [WARNING] Could not get WSL2 IP" -ForegroundColor Red
    }
} else {
    Write-Host ""
    Write-Host "[2/6] SKIPPING REDIS SWITCH (--SkipRedisSwitch)" -ForegroundColor Yellow
}

# ============================================================================
# PHASE 3: DELETE ALL DATABASE FILES
# ============================================================================
Write-Host ""
Write-Host "[3/6] DELETING ALL DATABASE FILES..." -ForegroundColor Cyan

$dbFiles = @(
    "$repoPath\mystic_trading.db",
    "$repoPath\mystic_trading.db-shm",
    "$repoPath\mystic_trading.db-wal",
    "$repoPath\paper_trading.db",
    "$repoPath\trading.db",
    "$repoPath\ai_live.sqlite",
    "$repoPath\ai_live.sqlite-shm",
    "$repoPath\ai_live.sqlite-wal",
    "$repoPath\data\mystic_trading.db",
    "$repoPath\data\mystic_trading.db-shm",
    "$repoPath\data\mystic_trading.db-wal",
    "$repoPath\data\paper_trading.db",
    "$repoPath\cache.db",
    "$repoPath\cache.db-shm",
    "$repoPath\cache.db-wal"
)

$deletedCount = 0
foreach ($dbFile in $dbFiles) {
    if (Test-Path $dbFile) {
        Remove-Item $dbFile -Force -ErrorAction SilentlyContinue
        Write-Host "  Deleted: $dbFile" -ForegroundColor Yellow
        $deletedCount++
    }
}

# Also find and delete any other .db or .sqlite files
$extraDbs = Get-ChildItem -Path $repoPath -Recurse -Include "*.db","*.sqlite","*.sqlite3" -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -notlike "*venv*" -and $_.FullName -notlike "*node_modules*" }

foreach ($db in $extraDbs) {
    Remove-Item $db.FullName -Force -ErrorAction SilentlyContinue
    Write-Host "  Deleted: $($db.FullName)" -ForegroundColor Yellow
    $deletedCount++
}

Write-Host "  [OK] Deleted $deletedCount database files" -ForegroundColor Green

# ============================================================================
# PHASE 4: FLUSH REDIS COMPLETELY
# ============================================================================
Write-Host ""
Write-Host "[4/6] FLUSHING REDIS..." -ForegroundColor Cyan

try {
    # Get WSL2 Redis IP
    $wslIp = (wsl -d Ubuntu hostname -I 2>$null).Trim()
    if ($wslIp) {
        $flushResult = redis-cli -h $wslIp FLUSHALL 2>$null
        if ($flushResult -eq "OK") {
            Write-Host "  [OK] Redis FLUSHALL completed on $wslIp" -ForegroundColor Green
        } else {
            Write-Host "  [WARNING] Redis flush returned: $flushResult" -ForegroundColor Yellow
        }
        
        # Verify empty
        $dbsize = redis-cli -h $wslIp DBSIZE 2>$null
        Write-Host "  Redis DBSIZE after flush: $dbsize" -ForegroundColor Cyan
    } else {
        Write-Host "  [WARNING] Could not get WSL2 IP for Redis flush" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Could not flush Redis: $_" -ForegroundColor Red
}

# ============================================================================
# PHASE 5: CLEAR ALL LOGS
# ============================================================================
Write-Host ""
Write-Host "[5/6] CLEARING ALL LOGS..." -ForegroundColor Cyan

$logsPath = "$repoPath\logs"
if (Test-Path $logsPath) {
    # Delete all files in logs directory
    $logFiles = Get-ChildItem -Path $logsPath -File -Recurse -ErrorAction SilentlyContinue
    $logCount = $logFiles.Count
    
    foreach ($logFile in $logFiles) {
        Remove-Item $logFile.FullName -Force -ErrorAction SilentlyContinue
    }
    
    # Also clear subdirectories like ai_learning_archives
    $subDirs = @(
        "$logsPath\ai_learning_archives",
        "$logsPath\startup"
    )
    
    foreach ($subDir in $subDirs) {
        if (Test-Path $subDir) {
            Get-ChildItem -Path $subDir -File -Recurse -ErrorAction SilentlyContinue | 
                ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
        }
    }
    
    Write-Host "  [OK] Deleted $logCount log files" -ForegroundColor Green
} else {
    Write-Host "  Logs directory not found" -ForegroundColor Yellow
}

# ============================================================================
# PHASE 6: VERIFICATION
# ============================================================================
Write-Host ""
Write-Host "[6/6] VERIFYING CLEAN STATE..." -ForegroundColor Cyan

$allClean = $true

# Check databases
$remainingDbs = Get-ChildItem -Path $repoPath -Recurse -Include "*.db","*.sqlite","*.sqlite3" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*venv*" -and $_.FullName -notlike "*node_modules*" }

if ($remainingDbs) {
    Write-Host "  [WARNING] Remaining database files:" -ForegroundColor Red
    $remainingDbs | ForEach-Object { Write-Host "    - $($_.FullName)" -ForegroundColor Red }
    $allClean = $false
} else {
    Write-Host "  [OK] No database files remaining" -ForegroundColor Green
}

# Check Redis
$dbsize = redis-cli DBSIZE 2>$null
if ($dbsize -match "keys=0" -or $dbsize -eq "(integer) 0") {
    Write-Host "  [OK] Redis is empty" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Redis still has keys: $dbsize" -ForegroundColor Yellow
}

# Check Python processes
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "  [WARNING] $($pythonProcs.Count) Python processes still running" -ForegroundColor Yellow
    $allClean = $false
} else {
    Write-Host "  [OK] No Python processes running" -ForegroundColor Green
}

# Check logs
$logFiles = Get-ChildItem -Path "$repoPath\logs" -File -Recurse -ErrorAction SilentlyContinue
$logSize = ($logFiles | Measure-Object -Property Length -Sum).Sum
if ($logSize -gt 1000) {
    Write-Host "  [WARNING] Logs still contain $([math]::Round($logSize/1KB, 1)) KB" -ForegroundColor Yellow
} else {
    Write-Host "  [OK] Logs cleared (< 1KB remaining)" -ForegroundColor Green
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
if ($allClean) {
    Write-Host "  CLEAN START COMPLETE - System is ready for fresh start" -ForegroundColor Green
} else {
    Write-Host "  CLEAN START COMPLETED WITH WARNINGS - Review above" -ForegroundColor Yellow
}
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: .\START_MYSTIC.ps1" -ForegroundColor White
Write-Host "  2. Wait for services to initialize" -ForegroundColor White
Write-Host "  3. Verify with: .\VERIFY_SYSTEM_HEALTH.ps1" -ForegroundColor White
Write-Host ""

