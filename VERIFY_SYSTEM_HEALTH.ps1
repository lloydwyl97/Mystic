# ============================================================================
# MYSTIC SYSTEM HEALTH VERIFICATION
# ============================================================================
# Phase 6: Prove "no duplicates, no corruption" in one command
# 
# Checks:
# 1. Writer lock PIDs - who holds each lock
# 2. Process counts - must be 1 per role
# 3. DB health - open lots, positions, recent trades, FIFO consistency
# 4. Redis health - keyspace, freshness
# 5. API health - HTTP 200
# ============================================================================

$ErrorActionPreference = "SilentlyContinue"
$repoPath = "C:\Users\lloyd\Mystic-Codebase"
Set-Location $repoPath

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  MYSTIC SYSTEM HEALTH VERIFICATION" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host "=" * 70 -ForegroundColor Cyan

$allGreen = $true

# ============================================================================
# CHECK 1: WRITER LOCKS
# ============================================================================
Write-Host ""
Write-Host "[1/5] WRITER LOCKS" -ForegroundColor Yellow
Write-Host "-" * 50

$writerRoles = @(
    "writer_lock:signal_generator",
    "writer_lock:orchestrator", 
    "writer_lock:portfolio_engine",
    "writer_lock:live_trading",
    "writer_lock:data_collector"
)

$lockCount = 0
foreach ($role in $writerRoles) {
    $lockValue = redis-cli GET $role 2>$null
    if ($lockValue) {
        Write-Host "  $role" -ForegroundColor Green -NoNewline
        Write-Host " -> $lockValue" -ForegroundColor White
        $lockCount++
    } else {
        Write-Host "  $role" -ForegroundColor Gray -NoNewline
        Write-Host " -> (not held)" -ForegroundColor DarkGray
    }
}

if ($lockCount -gt 0) {
    Write-Host "  [OK] $lockCount writer locks active" -ForegroundColor Green
} else {
    Write-Host "  [INFO] No writer locks held (services may not be running)" -ForegroundColor Yellow
}

# ============================================================================
# CHECK 2: PROCESS COUNTS
# ============================================================================
Write-Host ""
Write-Host "[2/5] PROCESS COUNTS (per role)" -ForegroundColor Yellow
Write-Host "-" * 50

$processPatterns = @{
    "Backend API (uvicorn)" = "uvicorn"
    "Data Collector" = "live_data_collector"
    "AI Signals" = "start_signals"
    "Orchestrator" = "start_orchestrator"
    "Portfolio Engine" = "start_portfolio_engine"
    "Paper AutoBuy" = "start_paper_autobuy"
    "Agent Orchestrator" = "start_agent_orchestrator"
    "Live Trading" = "start_ai_live_trading"
}

$totalProcesses = 0
foreach ($role in $processPatterns.Keys) {
    $pattern = $processPatterns[$role]
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$pattern*" }
    $count = if ($procs) { @($procs).Count } else { 0 }
    $totalProcesses += $count
    
    if ($count -eq 1) {
        Write-Host "  $role" -ForegroundColor Green -NoNewline
        Write-Host " -> $count (OK)" -ForegroundColor Green
    } elseif ($count -eq 0) {
        Write-Host "  $role" -ForegroundColor Yellow -NoNewline
        Write-Host " -> $count (not running)" -ForegroundColor Yellow
    } else {
        Write-Host "  $role" -ForegroundColor Red -NoNewline
        Write-Host " -> $count (DUPLICATE!)" -ForegroundColor Red
        $allGreen = $false
    }
}

Write-Host "  Total Python processes: $totalProcesses" -ForegroundColor Cyan

# ============================================================================
# CHECK 3: DATABASE HEALTH
# ============================================================================
Write-Host ""
Write-Host "[3/5] DATABASE HEALTH" -ForegroundColor Yellow
Write-Host "-" * 50

$dbPath = "$repoPath\mystic_trading.db"
if (Test-Path $dbPath) {
    # Open lots count
    try {
        $openLots = sqlite3 $dbPath "SELECT COUNT(*) FROM portfolio_engine_positions WHERE remaining_qty > 0;" 2>$null
        Write-Host "  Open lots (remaining_qty > 0): $openLots" -ForegroundColor Cyan
    } catch {
        Write-Host "  Open lots: (table not found)" -ForegroundColor Yellow
    }
    
    # Positions count
    try {
        $positions = sqlite3 $dbPath "SELECT COUNT(*) FROM portfolio_engine_positions;" 2>$null
        Write-Host "  Total position records: $positions" -ForegroundColor Cyan
    } catch {
        Write-Host "  Positions: (table not found)" -ForegroundColor Yellow
    }
    
    # Paper trades count
    try {
        $trades = sqlite3 $dbPath "SELECT COUNT(*) FROM paper_trades;" 2>$null
        Write-Host "  Total paper trades: $trades" -ForegroundColor Cyan
    } catch {
        Write-Host "  Paper trades: (table not found)" -ForegroundColor Yellow
    }
    
    # Last 5 trades (brief)
    try {
        Write-Host "  Last 5 trades:" -ForegroundColor Cyan
        $recentTrades = sqlite3 $dbPath "SELECT symbol, side, quantity, price, timestamp FROM paper_trades ORDER BY timestamp DESC LIMIT 5;" 2>$null
        if ($recentTrades) {
            $recentTrades -split "`n" | ForEach-Object { 
                if ($_) { Write-Host "    $_" -ForegroundColor White }
            }
        } else {
            Write-Host "    (no trades)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "    (query failed)" -ForegroundColor Yellow
    }
    
    # FIFO consistency check - negative remaining positions
    try {
        $negative = sqlite3 $dbPath "SELECT COUNT(*) FROM portfolio_engine_positions WHERE remaining_qty < 0;" 2>$null
        if ([int]$negative -gt 0) {
            Write-Host "  [ERROR] FIFO CORRUPTION: $negative positions with negative remaining_qty!" -ForegroundColor Red
            $allGreen = $false
        } else {
            Write-Host "  [OK] FIFO consistency: No negative remaining_qty" -ForegroundColor Green
        }
    } catch {
        Write-Host "  FIFO check: (table not found)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Database not found: $dbPath" -ForegroundColor Yellow
    Write-Host "  (This is OK if doing clean start)" -ForegroundColor Gray
}

# ============================================================================
# CHECK 4: REDIS HEALTH  
# ============================================================================
Write-Host ""
Write-Host "[4/5] REDIS HEALTH" -ForegroundColor Yellow
Write-Host "-" * 50

# Get WSL2 Redis IP
$wslIp = (wsl -d Ubuntu hostname -I 2>$null).Trim()
$redisHost = if ($wslIp) { $wslIp } else { "localhost" }

# Ping test
$ping = redis-cli -h $redisHost PING 2>$null
if ($ping -eq "PONG") {
    Write-Host "  [OK] Redis responding: PONG (host: $redisHost)" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Redis not responding at $redisHost!" -ForegroundColor Red
    $allGreen = $false
}

# Version
$version = redis-cli -h $redisHost INFO server 2>$null | Select-String "redis_version"
if ($version) {
    Write-Host "  Version: $($version -replace 'redis_version:', '')" -ForegroundColor Cyan
}

# Keyspace size
$dbsize = redis-cli -h $redisHost DBSIZE 2>$null
Write-Host "  Keyspace: $dbsize" -ForegroundColor Cyan

# Market data freshness
$lastUpdate = redis-cli -h $redisHost GET "market:last_update" 2>$null
if ($lastUpdate) {
    $updateTime = [DateTime]::Parse($lastUpdate)
    $age = (Get-Date) - $updateTime
    if ($age.TotalSeconds -lt 60) {
        Write-Host "  [OK] Market data fresh: $([math]::Round($age.TotalSeconds, 1))s ago" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Market data stale: $([math]::Round($age.TotalMinutes, 1)) minutes ago" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Market data: (no timestamp)" -ForegroundColor Gray
}

# Check for any error/stale keys
$errorKeys = redis-cli KEYS "*error*" 2>$null
$errorCount = if ($errorKeys) { ($errorKeys -split "`n" | Where-Object { $_ }).Count } else { 0 }
Write-Host "  Error-related keys: $errorCount" -ForegroundColor Cyan

# ============================================================================
# CHECK 5: API HEALTH
# ============================================================================
Write-Host ""
Write-Host "[5/5] API HEALTH" -ForegroundColor Yellow
Write-Host "-" * 50

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -ErrorAction Stop
    if ($health.status -eq "ok") {
        Write-Host "  [OK] API /health: status=ok" -ForegroundColor Green
        Write-Host "  Timestamp: $($health.timestamp)" -ForegroundColor Cyan
    } else {
        Write-Host "  [WARNING] API /health returned: $($health.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] API not responding: $_" -ForegroundColor Red
    $allGreen = $false
}

# Try portfolio status if available
try {
    $portfolio = Invoke-RestMethod -Uri "http://localhost:8000/api/portfolio/status" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($portfolio) {
        Write-Host "  Portfolio cash: `$$($portfolio.cash_balance)" -ForegroundColor Cyan
        Write-Host "  Open positions: $($portfolio.open_positions_count)" -ForegroundColor Cyan
    }
} catch {
    # Endpoint may not exist, that's OK
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
if ($allGreen) {
    Write-Host "  ALL CHECKS PASSED - System is healthy" -ForegroundColor Green
} else {
    Write-Host "  SOME CHECKS FAILED - Review issues above" -ForegroundColor Red
}
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

